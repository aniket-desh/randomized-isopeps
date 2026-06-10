import numpy as np
import numpy.linalg as la 
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None
try:
    import torch
except Exception:
    torch = None
import random

from .MPS import MPS
from .misc import maxlinkdim,check_randomized_apply
from .linalg import *
from .stopping import Cutoff
from .incrementalqr import *
from .rounding import * 


"""
READ ME (IMPORTANT):
In this file you will find all contraction algorithms used throughout the paper. Each algorithm comes as a pair, one written using np.einsum
(indicated by the _einsum suffix) and one written in pure numpy that directly compiles to BLAS matrix multiplication. In our experience, we have found 
that this explicit version of the contraction algorithm is up to 10x faster than the native einsum version due to several reasons:

In general, einsum versions of the algorithms are easier to read, and are recommended to review before parsing the BLAS versions
"""

#================================================ 
#          Blas Contraction Techniques 
#================================================ 
def mps_mpo_blas(mps, mpo, round_type = "standard",stop=Cutoff(1e-14),r=None,num_iters=5,oversample=5,l=None,final_round=False):
    new_mps = []

    # site = np.einsum("dDl,lX->dDX",mpo[0],mps[0])
    mpo_reshaped = mpo[0].reshape(mpo[0].shape[0]*mpo[0].shape[1], mpo[0].shape[2])    # Reshape to (dD,l)
    site = mpo_reshaped @ mps[0]                                                       # Resulting shape: (dD,X)
    site = site.reshape(mpo[0].shape[0], mpo[0].shape[1]*mps[0].shape[1])              # Reshape to (d,DX)
    new_mps.append(site)

    for i in range(1, mps.N-1):
        # site = np.einsum("DdEl,XlY->DXdEY",mpo[i],mps[i])
        mpo_reshaped = mpo[i].reshape(-1, mpo[i].shape[3])                             # Reshape to (DdE,l)
        mps_transposed = mps[i].transpose(1, 0, 2)                                     # Transpose to (l,X,Y)
        mps_reshaped = mps_transposed.reshape(mps_transposed.shape[0], -1)             # Reshape to (l,XY)
        site = mpo_reshaped @ mps_reshaped                                             # Resulting shape (DdE,XY)
        site = site.reshape(mpo[i].shape[0], mpo[i].shape[1],                          # Reshape to (D,d,E,X,Y)
                            mpo[i].shape[2], mps[i].shape[0], 
                            mps[i].shape[2])
        site = site.transpose(0, 3, 1, 2, 4)                                           # Transpose to (D,X,d,E,Y)
        site = site.reshape(site.shape[0]*site.shape[1], 
                            site.shape[2],
                            site.shape[3]*site.shape[4])                               # Reshape to (DE,d,XY)
        new_mps.append(site)

    # site = np.tensordot(mpo[-1], mps[-1], (2, 1))
    # site = np.einsum("Ddl,Xl->XDd",mpo[-1],mps[-1])
    mpo_reshaped = mpo[-1].reshape(mpo[-1].shape[0]*mpo[-1].shape[1], -1)              # Reshape to (Dd,l)
    site = mpo_reshaped @ mps[-1].T                                                    # Resulting shape (Dd,X)
    site = site.reshape(mpo[-1].shape[0], mpo[-1].shape[1], mps[-1].shape[0])          # Reshape to (D,d,X)
    site = site.transpose(0, 2, 1)                                                     # Transpose to (D,X,d)
    site = site.reshape(site.shape[0]*site.shape[1], site.shape[2])                    # Reshape to (DX,d)
    new_mps.append(site)


    new_mps = MPS(new_mps)
    if stop.is_truncation():
        if round_type == "standard":
            new_mps.round(stop=stop)
        elif round_type == "dass_blas":
            dass_round_blas(new_mps,stop=stop)
        elif round_type == "dass":
            dass_round(new_mps,stop=stop)
        elif round_type == "orth_then_rand_blas":
            orth_then_rand_blas(new_mps,r)
        elif round_type == "rand_then_orth_blas": #TODO: Strange runtime behavior for r=None
            rand_then_orth_blas(new_mps,r,finalround=final_round,stop=stop)
            # new_mps.display_tensors()
        elif round_type == "nystrom_blas":
            
            nystrom_round_blas(new_mps,l,stop=stop)
        else:
            raise ValueError(f"Unknown rounding type: {round_type}")
            
    return new_mps

def random_contraction(
    H,
    psi,
    stop=Cutoff(1e-6),
    sketchdim=1,
    sketchincrement=1,
    finalround=None,
    accuracychecks=False,
):
    # ===========================================================================
    # 1. Initialization
    # ===========================================================================
    n = len(H)

    if n != psi.N:
        raise ValueError("lengths of MPO and MPS do not match")

    if n == 1:
        raise NotImplementedError("MPO-MPS product for n=1 has not been implemented yet")
        # return [H[0] * psi[0]]

    psi_out = psi.N * [None]
    errorcount=0
    maxdim = stop.maxdim
    outputdim = stop.outputdim
    mindim = stop.mindim
    cutoff = stop.cutoff
    global_dtype = H[0].dtype
    if outputdim is None: # Adaptive determination of sketch dimension
        if maxdim is None:
            maxdim = maxlinkdim(H) * maxlinkdim(psi)
        mindim = max(mindim, 1)
    else:
        maxdim = outputdim
        mindim = outputdim
        sketchdim = outputdim
    
    envs = []
    cap = None
    cap_dim = 1

    visible_dim = H[0].shape[0]

    reshaped_psis = [psi[k].transpose(1, 0, 2).reshape(psi[k].shape[0]*psi[k].shape[1], psi[k].shape[2]) for k in range(1,n-1)]
    reshaped_H = [H[k].transpose(0, 2, 3, 1).reshape(-1, H[k].shape[1]) for k in range(1,n-1)]

    reshaped_psis2 = [psi[j].reshape(psi[j].shape[0], -1) for j in range (1,n-1)]
    reshaped_H2= [ H[j].transpose(1,2,0,3).reshape(-1,H[j].shape[0]*H[j].shape[3]) for j in range (1,n-1)]
    
    

    for j in reversed(range(1, n)):
        # print("J= ",j)
        # ===========================================================================
        # 2. Dimension Handling
        # ===========================================================================
        if j == n - 1:
            prod_bond_dims = (
                H[j].shape[0] * psi[j].shape[0]
            )  # product of left bond
        else:
            prod_bond_dims = max(
                (H[j].shape[0] * psi[j].shape[0]),
                (H[j].shape[2] * psi[j].shape[2]),
            )  # either left or right

        current_maxdim = min(prod_bond_dims, maxdim, visible_dim * cap_dim)
        current_mindim = min(mindim, current_maxdim)
        current_sketchdim = max(min(sketchdim, current_maxdim), current_mindim)

        sketches_complete = 0

        if j == n - 1:
            sketch = np.zeros((visible_dim, current_sketchdim),dtype='complex')
        else:
            sketch = np.zeros((visible_dim, cap_dim, current_sketchdim),dtype='complex')

        while True:
            # ===========================================================================
            # 3. env formation
            # ===========================================================================  
                          
            for idx in range(len(envs), current_sketchdim):
                random_vectors= [np.random.randn(visible_dim) for i in range(0,j)]
                env = []
                #x = np.random.randn(visible_dim) # consider precompute
                #temp= np.einsum('ijk,i->jk',H[0],x)
                H_reshaped = H[0].reshape(H[0].shape[0], -1)
                result = H_reshaped.T @ random_vectors[0]
                temp = result.reshape(H[0].shape[1], H[0].shape[2])
                env.append(temp @ psi[0])
                
                for k in range(1, j):
                    #x = np.random.randn(visible_dim) # consider precompute
                    #temp = np.einsum('ijkl,j->ikl', H[k], x)
                    #H_transposed = H[k].transpose(0, 2, 3, 1)
                    #H_reshaped = H_transposed.reshape(-1, H[k].shape[1])
                    #H_reshaped = H[k].transpose(0, 2, 3, 1).reshape(-1, H[k].shape[1])
                    
                    #result = np.dot(reshaped_H[k-1], random_vectors[k])
                    result=reshaped_H[k-1]@random_vectors[k]
                    temp = result.reshape(H[k].shape[0], H[k].shape[2], H[k].shape[3])

                    #temp = np.einsum('ijk,il->jkl', temp, env[k - 1])
                    
                    temp_reshaped = temp.reshape(temp.shape[0], -1) 
                    env_reshaped = env[k - 1].reshape(env[k - 1].shape[0], -1) 
                    result = temp_reshaped.T @ env_reshaped 
                    temp = result.reshape(temp.shape[1], temp.shape[2], env[k - 1].shape[1])
                    
                    #temp = np.tensordot(temp, psi[k], axes=([1, 2], [1, 0]))
                    #temp = np.einsum('ijk,kjl->il', temp, psi[k])
                    
                    temp = temp.reshape(temp.shape[0], -1) 
                    temp = temp @ reshaped_psis[k-1]

                    env.append(temp)
                envs.append(env)
            # ===========================================================================
            # 4. sketch  formation
            # ===========================================================================
            
            for idx in range(sketches_complete, current_sketchdim):
                if j == n - 1:
                    # C[n-2] (Hψ)[n-1]
                    temp = envs[idx][j - 1] @ psi[j]
    
                    #temp = np.einsum('ijk,ik->j',H[j],temp)
                    temp = (H[j].reshape(H[j].shape[1], -1) @ temp.reshape(-1)).reshape(H[j].shape[1])

                    sketch[:, idx] = temp
                else:
                    # C[j-1] (Hψ)[j] S[j+1]
                    #temp=np.einsum('ij,jkl->ikl',envs[idx][j - 1],psi[j])
                    #psij_reshaped = psi[j].reshape(psi[j].shape[0], -1)
                    
                    temp = envs[idx][j - 1] @ reshaped_psis2[j-1]
                    temp = temp.reshape(-1, psi[j].shape[1], psi[j].shape[2])

                    #temp=np.einsum('ijlk,ikn->jln',H[j], temp)
                    #Hj_transposed = H[j].transpose(1,2,0,3)
                    #Hj_reshaped = Hj_transposed.reshape(-1,Hj_transposed.shape[2]*Hj_transposed.shape[3])
                    temp_reshaped = temp.reshape(temp.shape[0]*temp.shape[1],-1)
                    result = reshaped_H2[j-1] @ temp_reshaped
                    temp = result.reshape(H[j].shape[1],H[j].shape[2],temp.shape[2])
                    # print(H[j].transpose(1, 2, 0, 3).reshape(-1, H[j].shape[2] * H[j].shape[3]).shape,temp.reshape(temp.shape[0] * temp.shape[2], -1).shape)
                    # temp = H[j].transpose(1, 2, 0, 3).reshape(-1, H[j].shape[2] * H[j].shape[3]) @ temp.reshape(temp.shape[0] * temp.shape[2], -1)
                    # temp = temp.reshape(H[j].shape[0], H[j].shape[2], -1).transpose(1, 0, 2)
                    
                    #temp = np.einsum('ijk,ljk->il',temp,cap)
                    cap_transposed = cap.transpose(1,2,0)
                    cap_reshaped = cap_transposed.reshape(-1,cap_transposed.shape[2])
                    temp_reshaped = temp.reshape(-1,temp.shape[1]*temp.shape[2])
                    temp = temp_reshaped @ cap_reshaped
                    #temp = temp.reshape(temp.shape[0],-1) @ temp.transpose(1,2,0).reshape(-1,temp.shape[2])

                    sketch[:, :, idx] = temp
            sketches_complete = current_sketchdim
            # ===========================================================================
            # 5. QR decomposition
            # ===========================================================================
            if j == n - 1:
                Q, R = np.linalg.qr(sketch)
            else:
                temp = sketch.reshape(cap_dim * visible_dim, current_sketchdim)
                Q, R = np.linalg.qr(temp)
                Q = Q.reshape((visible_dim, cap_dim, current_sketchdim))
            
            if not (outputdim is None):
                done = True
            else:
                # ===========================================================================
                # 7. error calc
                # ===========================================================================
                done = False
                if current_sketchdim == current_maxdim:
                    done = True
                else:
                    errorcount+=1
                    norm_est = np.linalg.norm(sketch) / np.sqrt(current_sketchdim)
                    G = np.linalg.inv(R.T)

                    err_est = np.sqrt(sum(np.linalg.norm(G, axis=0) ** (-2)) / current_sketchdim)
                    done = err_est <= cutoff * norm_est and current_sketchdim >= current_mindim
            # print("error estimation", err_est)
            # print("norm estimation",norm_est)
            # print("Validity of condition 1:",(err_est <= cutoff * norm_est and current_sketchdim >= current_mindim))
            # print("Validity of condition 2:",(current_sketchdim == current_maxdim))
            # ===========================================================================
            # 9. cap construction
            # ===========================================================================
            if done:
                if j == n - 1:
                    psi_out[j] = Q.transpose(1, 0)
                else:
                    psi_out[j] = Q.transpose(2, 0, 1)
                cap_dim = current_sketchdim

                if j == n - 1:
                    #temp=np.einsum("ij,kil->jkl",np.conj(Q), H[j])
                    temp = np.conj(Q).T @ H[j].transpose(1, 0, 2).reshape(H[j].shape[1], -1)
                    temp = temp.reshape(Q.shape[1], H[j].shape[0], H[j].shape[2])
                    
                    #cap =np.einsum("ijk,lk->ijl",temp, psi[j])
                    temp_ = temp.reshape(-1, temp.shape[-1])
                    psi_j_ = psi[j].reshape(psi[j].shape[0], -1)
                    cap = np.dot(temp_, psi_j_.T).reshape(temp.shape[0], temp.shape[1], psi[j].shape[0])
                else:
      
                    #temp = np.tensordot(np.conj(Q), cap, axes=(1, 0))
                    Q_reshaped = np.conj(Q).transpose(0,2,1)
                    Q_reshaped = Q_reshaped.reshape(-1, Q_reshaped.shape[2])
                    cap_reshaped = cap.reshape(cap.shape[0],-1)
                    temp = (Q_reshaped @ cap_reshaped)
                    temp = temp.reshape(Q.shape[0], Q.shape[2], cap.shape[1], cap.shape[2])


                    #temp = np.einsum("ijkl,mikn->jlmn",temp, H[j])
                    temp_transposed = temp.transpose(1, 3, 0, 2) 
                    Hj_transposed = H[j].transpose(1, 2, 0, 3)
                    temp_reshaped = temp_transposed.reshape(-1,temp_transposed.shape[2]*temp_transposed.shape[3])
                    Hj_reshaped = Hj_transposed.reshape( Hj_transposed.shape[0]*Hj_transposed.shape[1],-1,)
                    result = temp_reshaped @ Hj_reshaped
                    temp = result.reshape(temp.shape[1], temp.shape[3], H[j].shape[0], H[j].shape[3])
                    
                    #cap = np.einsum("ijkl,mlj->ikm",temp, psi[j])
                    temp_reshaped = temp.transpose(0, 2, 1, 3)
                    temp_reshaped=temp_reshaped.reshape(-1,temp_reshaped.shape[2]*temp_reshaped.shape[3])
                    psi_j_reshaped = psi[j].transpose(2, 1, 0)
                    psi_j_reshaped=psi_j_reshaped.reshape(psi_j_reshaped.shape[0]*psi_j_reshaped.shape[1], -1)
                    cap = (temp_reshaped @ psi_j_reshaped)
                    cap = cap.reshape(temp.shape[0], temp.shape[2], psi[j].shape[0])

                if accuracychecks:
                    check_randomized_apply(H, psi, cap, psi_out, j+1, verbose=False, cutoff=((100*cutoff) if (maxdim is None) else np.inf))
                break

            # ===========================================================================
            # 10. sketch correction
            # ===========================================================================
            current_sketchdim = min(current_maxdim, current_sketchdim + sketchincrement)
            old_sketch = sketch

            if j == n - 1:
                sketch = np.zeros(
                    (
                        old_sketch.shape[0],
                        current_sketchdim,
                    ),dtype='complex'
                )
                sketch[:, : old_sketch.shape[1]] = old_sketch
            else:
               
                sketch = np.zeros(
                    (
                        old_sketch.shape[0],
                        old_sketch.shape[1],
                        current_sketchdim,
                    ),dtype='complex'
                )
                sketch[:, :, : old_sketch.shape[2]] = old_sketch

    temp = np.einsum('ijk,lk',cap, psi[0])
    #print(H[0].shape,temp.shape)
    psi_out[0] = np.einsum('ijk,ljk->il',H[0],temp)
    #psi_out[0] = np.tensordot(H[0], temp, axes=([1,2], [1,2]))
    #print(psi_out[0].shape)
    # mps_round_left(psi_out,cutoff,maxdim)
    #print(errorcount)
    psi_out= MPS(psi_out, canform="Left")
    if not (finalround is None):
        psi_out.round(stop = finalround)
        
    return psi_out

def random_contraction_inc(
    H,
    psi,
    stop=Cutoff(1e-6),
    sketchdim=1,
    sketchincrement=1,
    finalround=None,
    accuracychecks=False,
    cpp=True,
    debug=True
):
    # ===========================================================================
    # 1. Initialization
    # ===========================================================================
    n = len(H)

    if n != psi.N:
        raise ValueError("lengths of MPO and MPS do not match")

    if n == 1:
        raise NotImplementedError("MPO-MPS product for n=1 has not been implemented yet")
        # return [H[0] * psi[0]]

    psi_out = psi.N * [None]
    maxdim = stop.maxdim
    outputdim = stop.outputdim
    mindim = stop.mindim
    cutoff = stop.cutoff

    if outputdim is None:  # Adaptive determination of sketch dimension
        if maxdim is None:
            maxdim = maxlinkdim(H) * maxlinkdim(psi)
        mindim = max(mindim, 1)
    else:
        maxdim = outputdim
        mindim = outputdim
        sketchdim = outputdim
    # if debug:
    #     error_counts = []
    envs = []
    cap = None
    cap_dim = 1

    visible_dim = H[0].shape[0]

    reshaped_psis = [psi[k].transpose(1, 0, 2).reshape(psi[k].shape[0] * psi[k].shape[1], psi[k].shape[2]) for k in range(1, n - 1)]
    reshaped_H = [H[k].transpose(0, 2, 3, 1).reshape(-1, H[k].shape[1]) for k in range(1, n - 1)]

    for j in reversed(range(1, n)):
        errorcount = 0
        # ===========================================================================
        # 2. Dimension Handling
        # ===========================================================================
        if j == n - 1:
            prod_bond_dims = H[j].shape[0] * psi[j].shape[0]  # product of left bond
        else:
            prod_bond_dims = max(H[j].shape[0] * psi[j].shape[0], H[j].shape[2] * psi[j].shape[2])  # either left or right

        current_maxdim = min(prod_bond_dims, maxdim, visible_dim * cap_dim)
        current_mindim = min(mindim, current_maxdim)
        current_sketchdim = max(min(sketchdim, current_maxdim), current_mindim)

        sketches_complete = 0

        if j == n - 1:
            sketch = np.zeros((visible_dim, 0), dtype='complex')  # TODO: resolve typing
        else:
            sketch = np.zeros((visible_dim, cap_dim, 0), dtype='complex')
        
        while True:
            # ===========================================================================
            # 3. env formation
            # ===========================================================================                
            # envs = form_environments(H, psi,envs, j, visible_dim, reshaped_H, reshaped_psis,current_sketchdim)
            for idx in range(len(envs), current_sketchdim):
                random_vectors= [np.random.randn(visible_dim) for i in range(0,j)]
                env = []
                #temp= np.einsum('ijk,i->jk',H[0],x)
                H_reshaped = H[0].reshape(H[0].shape[0], -1)
                result = H_reshaped.T @ random_vectors[0]
                temp = result.reshape(H[0].shape[1], H[0].shape[2])
                env.append(temp @ psi[0])
                
                for k in range(1, j):
                    #temp = np.einsum('ijkl,j->ikl', H[k], x)
                    result=reshaped_H[k-1]@random_vectors[k]
                    temp = result.reshape(H[k].shape[0], H[k].shape[2], H[k].shape[3])

                    #temp = np.einsum('ijk,il->jkl', temp, env[k - 1])
                    temp_reshaped = temp.reshape(temp.shape[0], -1) 
                    env_reshaped = env[k - 1].reshape(env[k - 1].shape[0], -1) 
                    result = temp_reshaped.T @ env_reshaped 
                    temp = result.reshape(temp.shape[1], temp.shape[2], env[k - 1].shape[1])
                    
                    #temp = np.einsum('ijk,kjl->il', temp, psi[k])
                    temp = temp.reshape(temp.shape[0], -1) 
                    temp = temp @ reshaped_psis[k-1]

                    env.append(temp)
                envs.append(env)
            # ===========================================================================
            # 4. sketch  formation
            # ===========================================================================
            # sketch = form_sketch(sketch,envs,sketches_complete,current_sketchdim,H,psi,reshaped_psis2,reshaped_H2,j,cap)
            all_env = np.stack([envs[idx][j-1] for idx in range(sketches_complete, current_sketchdim)], axis=0)
            #print(current_sketchdim,sketches_complete)
            if j == psi.N - 1:
                # temp = np.einsum('ZDX,Xl->ZDl',all_env,psi[j])
                prod = all_env.reshape(-1,all_env.shape[2]) @ psi[j] 
                temp = prod.reshape(all_env.shape[0],all_env.shape[1],psi[j].shape[1])
                
                # temp = np.einsum('ZDl,Ddl->dZ',temp,H[j])
                H_transposed = H[j].transpose(0,2,1) #Trans to D,l,d
                prod =  temp.reshape(temp.shape[0],-1) @ H_transposed.reshape(-1,H_transposed.shape[-1]) #Resulting shape Z,d
                temp = prod.T
            else:
                # temp = np.einsum('ZDX,XdY->ZDdY',all_env,psi[j])
                prod = all_env.reshape(-1,all_env.shape[-1]) @ psi[j].reshape(psi[j].shape[0],-1) # resulting shape(ZD,dY)
                temp = prod.reshape(all_env.shape[0],all_env.shape[1],psi[j].shape[1],psi[j].shape[2]) #reshape to (Z,D,d,Y)
                
                # temp = np.einsum('ZDlY,DdEl->ZdEY',temp,H[j])
                temp_transposed =  temp.transpose(0,3,1,2) #trans to ZYDl
                H_transposed = H[j].transpose(0,3,1,2) #Trans to DldE
                prod = temp_transposed.reshape(-1,temp_transposed.shape[2]*temp_transposed.shape[3]) @ H_transposed.reshape(H_transposed.shape[0]*H_transposed.shape[1],-1) #Resulting shape (ZY,dE) 
                temp = prod.reshape(temp.shape[0],temp.shape[3],H[j].shape[1],H[j].shape[2]) #Reshape to ZYdE
                temp = temp.transpose(0,2,3,1) #trans to ZdEy
                
                # temp = np.einsum('ZdEY,PEY->dPZ',temp,cap) # (4, 2, 20, 20) (2, 20, 20)
                cap_transposed = cap.transpose(1,2,0) #trans to EYP
                prod = temp.reshape(-1,temp.shape[2]*temp.shape[3]) @ cap_transposed.reshape(-1,cap_transposed.shape[-1]) #Resulting shape Zdp
                temp = prod.reshape(temp.shape[0],temp.shape[1],cap.shape[0]) #Reshape to Z,d,p
                temp = temp.transpose(1,2,0)
                
            sketch = np.concatenate((sketch, temp), axis=-1)
            # ===========================================================================
            # 5. QR decomposition
            # ===========================================================================
            if j == n - 1:
                if errorcount == 0:
                    qr = IncrementalQR(sketch, use_cpp_if_available=cpp)
                else:
                    qr.append(sketch[:, current_sketchdim:sketches_complete])
            else:
                if errorcount == 0:
                    temp = sketch[:, :, sketches_complete:current_sketchdim].reshape(cap_dim * visible_dim, -1)
                    qr = IncrementalQR(temp, use_cpp_if_available=cpp)
                else:
                    temp = sketch[:, :, sketches_complete:current_sketchdim].reshape(cap_dim * visible_dim, -1)
                    qr.append(temp)

            sketches_complete = current_sketchdim

            if outputdim is not None:  # If fixed dimension cutoff
                done = True
                
            else:
                # ===========================================================================
                # 7. error calc
                # ===========================================================================
                done = False
                if current_sketchdim == current_maxdim:
                    done = True
                else:
                    err_est = qr.error_estimate()
                    errorcount += 1
                    norm_est = np.linalg.norm(sketch.reshape(-1)) / np.sqrt(current_sketchdim)
                    done = err_est <= cutoff * norm_est and current_sketchdim >= current_mindim
                    
            # ==========================================================================
            # 9. cap construction
            # ===========================================================================

            if done:
                # if debug:
                #     error_counts.append(errorcount)
                Q = qr.get_q()
                if j == n - 1:
                    psi_out[j] = Q.transpose(1, 0)
                else:
                    Q = Q.reshape((visible_dim, cap_dim, current_sketchdim))
                    psi_out[j] = Q.transpose(2, 0, 1)
                cap_dim = current_sketchdim

                if j == n - 1:
                    #temp=np.einsum("ij,kil->jkl",np.conj(Q), H[j])
                    temp = np.conj(Q).T @ H[j].transpose(1, 0, 2).reshape(H[j].shape[1], -1)
                    temp = temp.reshape(Q.shape[1], H[j].shape[0], H[j].shape[2])
                    
                    #cap =np.einsum("ijk,lk->ijl",temp, psi[j])
                    temp_ = temp.reshape(-1, temp.shape[-1])
                    psi_j_ = psi[j].reshape(psi[j].shape[0], -1)
                    cap = np.dot(temp_, psi_j_.T).reshape(temp.shape[0], temp.shape[1], psi[j].shape[0])
                else:
                    #temp = np.tensordot(np.conj(Q), cap, axes=(1, 0))
                    Q_reshaped = np.conj(Q).transpose(0,2,1)
                    Q_reshaped = Q_reshaped.reshape(-1, Q_reshaped.shape[2])
                    cap_reshaped = cap.reshape(cap.shape[0],-1)
                    temp = (Q_reshaped @ cap_reshaped)
                    temp = temp.reshape(Q.shape[0], Q.shape[2], cap.shape[1], cap.shape[2])

                    #temp = np.einsum("ijkl,mikn->jlmn",temp, H[j])
                    temp_transposed = temp.transpose(1, 3, 0, 2) 
                    Hj_transposed = H[j].transpose(1, 2, 0, 3)
                    temp_reshaped = temp_transposed.reshape(-1,temp_transposed.shape[2]*temp_transposed.shape[3])
                    Hj_reshaped = Hj_transposed.reshape(Hj_transposed.shape[0]*Hj_transposed.shape[1],-1)
                    result = temp_reshaped @ Hj_reshaped
                    temp = result.reshape(temp.shape[1], temp.shape[3], H[j].shape[0], H[j].shape[3])
                    
                    #cap = np.einsum("ijkl,mlj->ikm",temp, psi[j])
                    temp_reshaped = temp.transpose(0, 2, 1, 3)
                    temp_reshaped=temp_reshaped.reshape(-1,temp_reshaped.shape[2]*temp_reshaped.shape[3])
                    psi_j_reshaped = psi[j].transpose(2, 1, 0)
                    psi_j_reshaped=psi_j_reshaped.reshape(psi_j_reshaped.shape[0]*psi_j_reshaped.shape[1], -1)
                    cap = (temp_reshaped @ psi_j_reshaped)
                    cap = cap.reshape(temp.shape[0], temp.shape[2], psi[j].shape[0])

                if accuracychecks:
                    check_randomized_apply(H, psi, cap, psi_out, j+1, verbose=False, cutoff=((100*cutoff) if (maxdim is None) else np.Inf))
                break
            
            # ===========================================================================
            # 10. sketch correction
            # ===========================================================================
            current_sketchdim = min(current_maxdim, current_sketchdim + sketchincrement)

    # temp = np.einsum('pDX,lX',cap, psi[0])
    temp = cap.reshape(-1,cap.shape[-1]) @ psi[0].T #Resulting shape pDl
    temp = temp.reshape(cap.shape[0],cap.shape[1],psi[0].shape[0])
    psi_out[0] = np.einsum('pDl,dDl->pd',H[0],temp)
    psi_out= MPS(psi_out, canform="Left")
    
    if not (finalround is None):
        psi_out.round(stop = finalround)
    # if debug:
    #     return psi_out,error_counts

    return psi_out
    
def density_matrix(mpo, mps, stop=Cutoff(1e-14), maxdim=None, normalize=False):
    """
    Einsum Indexing convention: 
     ______
    |      |    _______
    |      |-Z-|mps_c_j|-W-...
    |      |   |_______|
    |      |    __|l___
    |      |-F-|mpo_c_j|-G-...
    |      |   |_______|
    | L[j] |      |d  
    |      |
    |      |    __|d___
    |      |-D-| mpo_j |-E-...
    |      |   |_______|
    |      |    __|l___
    |      |-X-| mps_j |-Y-...
    |______|   |_______|
    
    Comments:
    - In cases where the same index is duplicated in a contraction ie: dxd density matrix the 
      surrogate index will be promoted one letter: dxd -> exd.
    - truncation variable produced by eig is k.
    """
    # ==========================================
    # 0. [Initialization]
    # ==========================================
    if len(mpo) != len(mps):
        raise ValueError("MPO and MPS must have the same length.")
    if maxdim is None:
        requested_maxdim = maxlinkdim(mpo) * maxlinkdim(mps)


    n = len(mpo)
    mps_out = mps.copy()
    mps_c = mps.dagger()
    mpo_c = mpo.dagger()
    L = [None] * (n - 1)


    # ==========================================
    # 1. [Environment Tensor Construction]          
    # 1. [Environment Tensor Construction]          
    # ==========================================
    
    # L[0] = np.einsum('dDl,lX->dDX', mpo[0], mps[0]) 
    mpo_reshaped = mpo[0].reshape(mpo[0].shape[0] * mpo[0].shape[1], -1)                    # Reshape to (dD, l)
    mps_reshaped = mps[0].reshape(-1, mps[0].shape[1])                                      # Reshape to (l, X)
    temp = mpo_reshaped @ mps_reshaped                                      
    L[0] = temp.reshape(mpo[0].shape[0], mpo[0].shape[1], mps[0].shape[1])   

    # temp = np.einsum('dFl,lZ->ZFd', mpo_c[0], mps_c[0])
    mpo_reshaped = mpo_c[0].reshape(mpo_c[0].shape[0] * mpo_c[0].shape[1], -1)              # Reshape to (dF, l)
    mps_reshaped = mps_c[0].reshape(-1, mps_c[0].shape[1])                                  # Reshape to (l, Z)
    temp = mpo_reshaped @ mps_reshaped                                        
    temp = temp.reshape(mpo_c[0].shape[0], mpo_c[0].shape[1], mps_c[0].shape[1])            # Reshape to (d, F, Z)
    temp = temp.transpose(2, 1, 0)                                                          # Transpose to (Z, F, d)

    # L[0] = np.einsum('dDX,ZFd->XDFZ', L[0], temp)
    L_transposed = L[0].transpose(2, 1, 0)                                                  # Transpose to (X, D, d)
    L_reshaped = L_transposed.reshape(-1, L_transposed.shape[2])                            # Reshape to (XD, d)
    temp_transposed = temp.transpose(2, 1, 0)                                               # Transpose to (d, F, Z)
    temp_reshaped = temp_transposed.reshape(temp_transposed.shape[0], -1)                   # Reshape to (d, FZ)
    L[0] = L_reshaped @ temp_reshaped                                                       # Resulting shape (XD, FZ)
    L[0] = L[0].reshape(L_transposed.shape[0], L_transposed.shape[1], 
                        temp_transposed.shape[1], temp_transposed.shape[2])                 # Reshape to (X, D, F, Z)

    for j in range(1, n-1):
        # L[j] = np.einsum('XDFZ,ZlW->XDFlW', L[j-1], mps_c[j])
        L_reshaped = L[j-1].reshape(-1, L[j-1].shape[3])                                    # Reshape to (XDF, Z)
        mps_reshaped = mps_c[j].reshape(mps_c[j].shape[0], -1)                              # Reshape to (Z, lW)
        intermediate = L_reshaped @ mps_reshaped                                     
        intermediate_reshaped = intermediate.reshape(L[j-1].shape[0], L[j-1].shape[1], 
                                                    L[j-1].shape[2], mps_c[j].shape[1], 
                                                    mps_c[j].shape[2])                      # Reshape to (X, D, F, l, W)
        L[j] = intermediate_reshaped.transpose(0, 1, 2, 3, 4)                               # Transpose to (XDFlW)

        # L[j] = np.einsum('XDFlW,FdGl->XDdGW', L[j], mpo_c[j])
        L_transposed = L[j].transpose(0, 1, 4, 2, 3)                                        # Transpose to (X, D, W, F, l)
        mpo_transposed = mpo_c[j].transpose(0, 3, 1, 2)                                     # Transpose to (F, l, d, G)
        L_reshaped = L_transposed.reshape(L_transposed.shape[0] * L_transposed.shape[2] * L_transposed.shape[3], -1)   # Reshape to (XDW, Fl)
        mpo_reshaped = mpo_transposed.reshape(-1, mpo_transposed.shape[2] * mpo_transposed.shape[3])                   # Reshape to (Fl, dG)
        L[j] = L_reshaped @ mpo_reshaped                                                    # Resulting shape (XDW, dG)
        L[j] = L[j].reshape(L_transposed.shape[0], L_transposed.shape[1], L_transposed.shape[2], 
                            mpo_transposed.shape[2], mpo_transposed.shape[3])               # Reshape to (X, D, W, d, G)
        L[j] = L[j].transpose(0, 1, 3, 4, 2)                                                # Transpose to (X, D, d, G, W)

        
        # L[j] = np.einsum('XDdGW,DdEl->XlEGW', L[j], mpo[j]) 
        L_transposed = L[j].transpose(0, 3, 4, 1, 2)                                        # Transpose to (X, G, W, D, d)
        L_reshaped = L_transposed.reshape(L_transposed.shape[0] * L_transposed.shape[1] * L_transposed.shape[2], -1)  # Reshape to (XGW, Dd)
        mpo_reshaped = mpo[j].reshape(mpo[j].shape[0] * mpo[j].shape[1], mpo[j].shape[2] * mpo[j].shape[3])           # Reshape to (Dd, El)
        L[j] = L_reshaped @ mpo_reshaped                                                    # Resulting shape (XGW, El)
        L[j] = L[j].reshape(L_transposed.shape[0], L_transposed.shape[1], L_transposed.shape[2], 
                            mpo[j].shape[2], mpo[j].shape[3])                               # Reshape to (X, G, W, E, l)
        L[j] = L[j].transpose(0, 4, 3, 1, 2)                                                # Transpose to (X, l, E, G, W)
        
        # L[j] = np.einsum('XlEGW,XlY->YEGW', L[j], mps[j]) 
        L_transposed = L[j].transpose(2,3,4,0,1)                                            # Transpose to ( E,G,W,X,l)
        L_reshaped = L_transposed.reshape(-1,L_transposed.shape[3]*L_transposed.shape[4])   # Reshape to (EGW,Xl)
        mps_reshaped = mps[j].reshape(mps[j].shape[0]*mps[j].shape[1],mps[j].shape[2])      # Reshape to (Xl,Y)
        L[j] = L_reshaped @ mps_reshaped                                                    # Resulting shape (EGW, Y)
        L[j] = L[j].reshape(L_transposed.shape[0],L_transposed.shape[1],L_transposed.shape[2],-1) # Reshape to (E,G,W,Y)
        L[j] = L[j].transpose(3,0,1,2) #Tranpose to (Y,E,G,W)
    
    # L[0] = np.einsum('dDl,lX->dDX', mpo[0], mps[0]) 
    mpo_reshaped = mpo[0].reshape(mpo[0].shape[0] * mpo[0].shape[1], -1)                    # Reshape to (dD, l)
    mps_reshaped = mps[0].reshape(-1, mps[0].shape[1])                                      # Reshape to (l, X)
    temp = mpo_reshaped @ mps_reshaped                                      
    L[0] = temp.reshape(mpo[0].shape[0], mpo[0].shape[1], mps[0].shape[1])   

    # temp = np.einsum('dFl,lZ->ZFd', mpo_c[0], mps_c[0])
    mpo_reshaped = mpo_c[0].reshape(mpo_c[0].shape[0] * mpo_c[0].shape[1], -1)              # Reshape to (dF, l)
    mps_reshaped = mps_c[0].reshape(-1, mps_c[0].shape[1])                                  # Reshape to (l, Z)
    temp = mpo_reshaped @ mps_reshaped                                        
    temp = temp.reshape(mpo_c[0].shape[0], mpo_c[0].shape[1], mps_c[0].shape[1])            # Reshape to (d, F, Z)
    temp = temp.transpose(2, 1, 0)                                                          # Transpose to (Z, F, d)

    # L[0] = np.einsum('dDX,ZFd->XDFZ', L[0], temp)
    L_transposed = L[0].transpose(2, 1, 0)                                                  # Transpose to (X, D, d)
    L_reshaped = L_transposed.reshape(-1, L_transposed.shape[2])                            # Reshape to (XD, d)
    temp_transposed = temp.transpose(2, 1, 0)                                               # Transpose to (d, F, Z)
    temp_reshaped = temp_transposed.reshape(temp_transposed.shape[0], -1)                   # Reshape to (d, FZ)
    L[0] = L_reshaped @ temp_reshaped                                                       # Resulting shape (XD, FZ)
    L[0] = L[0].reshape(L_transposed.shape[0], L_transposed.shape[1], 
                        temp_transposed.shape[1], temp_transposed.shape[2])                 # Reshape to (X, D, F, Z)

    for j in range(1, n-1):
        # L[j] = np.einsum('XDFZ,ZlW->XDFlW', L[j-1], mps_c[j])
        L_reshaped = L[j-1].reshape(-1, L[j-1].shape[3])                                    # Reshape to (XDF, Z)
        mps_reshaped = mps_c[j].reshape(mps_c[j].shape[0], -1)                              # Reshape to (Z, lW)
        intermediate = L_reshaped @ mps_reshaped                                     
        intermediate_reshaped = intermediate.reshape(L[j-1].shape[0], L[j-1].shape[1], 
                                                    L[j-1].shape[2], mps_c[j].shape[1], 
                                                    mps_c[j].shape[2])                      # Reshape to (X, D, F, l, W)
        L[j] = intermediate_reshaped.transpose(0, 1, 2, 3, 4)                               # Transpose to (XDFlW)

        # L[j] = np.einsum('XDFlW,FdGl->XDdGW', L[j], mpo_c[j])
        L_transposed = L[j].transpose(0, 1, 4, 2, 3)                                        # Transpose to (X, D, W, F, l)
        mpo_transposed = mpo_c[j].transpose(0, 3, 1, 2)                                     # Transpose to (F, l, d, G)
        L_reshaped = L_transposed.reshape(L_transposed.shape[0] * L_transposed.shape[2] * L_transposed.shape[3], -1)   # Reshape to (XDW, Fl)
        mpo_reshaped = mpo_transposed.reshape(-1, mpo_transposed.shape[2] * mpo_transposed.shape[3])                   # Reshape to (Fl, dG)
        L[j] = L_reshaped @ mpo_reshaped                                                    # Resulting shape (XDW, dG)
        L[j] = L[j].reshape(L_transposed.shape[0], L_transposed.shape[1], L_transposed.shape[2], 
                            mpo_transposed.shape[2], mpo_transposed.shape[3])               # Reshape to (X, D, W, d, G)
        L[j] = L[j].transpose(0, 1, 3, 4, 2)                                                # Transpose to (X, D, d, G, W)

        
        # L[j] = np.einsum('XDdGW,DdEl->XlEGW', L[j], mpo[j]) 
        L_transposed = L[j].transpose(0, 3, 4, 1, 2)                                        # Transpose to (X, G, W, D, d)
        L_reshaped = L_transposed.reshape(L_transposed.shape[0] * L_transposed.shape[1] * L_transposed.shape[2], -1)  # Reshape to (XGW, Dd)
        mpo_reshaped = mpo[j].reshape(mpo[j].shape[0] * mpo[j].shape[1], mpo[j].shape[2] * mpo[j].shape[3])           # Reshape to (Dd, El)
        L[j] = L_reshaped @ mpo_reshaped                                                    # Resulting shape (XGW, El)
        L[j] = L[j].reshape(L_transposed.shape[0], L_transposed.shape[1], L_transposed.shape[2], 
                            mpo[j].shape[2], mpo[j].shape[3])                               # Reshape to (X, G, W, E, l)
        L[j] = L[j].transpose(0, 4, 3, 1, 2)                                                # Transpose to (X, l, E, G, W)
        
        # L[j] = np.einsum('XlEGW,XlY->YEGW', L[j], mps[j]) 
        L_transposed = L[j].transpose(2,3,4,0,1)                                            # Transpose to ( E,G,W,X,l)
        L_reshaped = L_transposed.reshape(-1,L_transposed.shape[3]*L_transposed.shape[4])   # Reshape to (EGW,Xl)
        mps_reshaped = mps[j].reshape(mps[j].shape[0]*mps[j].shape[1],mps[j].shape[2])      # Reshape to (Xl,Y)
        L[j] = L_reshaped @ mps_reshaped                                                    # Resulting shape (EGW, Y)
        L[j] = L[j].reshape(L_transposed.shape[0],L_transposed.shape[1],L_transposed.shape[2],-1) # Reshape to (E,G,W,Y)
        L[j] = L[j].transpose(3,0,1,2) #Tranpose to (Y,E,G,W)

    # ==========================================
    # 2. [First Density Matrix]
    # ==========================================

    # rho = np.einsum('XDFZ,Zl->XDFl', L[n-2], mps_c[n-1])
    L_reshaped = L[n-2].reshape(-1, L[n-2].shape[3])                                        # Reshape to (XDF, Z)
    mps_reshaped = mps_c[n-1].reshape(mps_c[n-1].shape[0], -1)                              # Reshape to (Z, l)
    intermediate = L_reshaped @ mps_reshaped                                   
    rho = intermediate.reshape(L[n-2].shape[0], L[n-2].shape[1], L[n-2].shape[2], mps_c[n-1].shape[1])  # Reshape to (X, D, F, l)

    # rho = np.einsum('XDFl,Fdl->XDd', rho, mpo_c[n-1])
    rho_reshaped = rho.reshape(rho.shape[0] * rho.shape[1], rho.shape[2] * rho.shape[3])    # Reshape to (XD, Fl)
    mpo_transposed = mpo_c[n-1].transpose(0, 2, 1)                                          # Transpose to (F, l, d)
    mpo_reshaped = mpo_transposed.reshape(mpo_transposed.shape[0] * mpo_transposed.shape[1], -1)      # Reshape to (Fl, d)
    temp = rho_reshaped @ mpo_reshaped                                                      # Resulting shape (XD, d)
    rho = temp.reshape(rho.shape[0], rho.shape[1], -1)                                      # Reshape to (X, D, d)

    # rho = np.einsum('XDd,Xl->lDd', rho, mps[n-1])
    rho_reshaped = rho.reshape(rho.shape[0], -1)                                            # Reshape to (X, Dd)
    mps_reshaped = mps[n-1].reshape(mps[n-1].shape[0], -1)                                  # Reshape to (X, l)
    intermediate = rho_reshaped.T @ mps_reshaped                                                 
    rho = intermediate.T.reshape(mps[n-1].shape[1], rho.shape[1], rho.shape[2])             # Reshape to (l, D, d)
    
    # rho = np.einsum('lDd,Dol->od', rho, mpo[n-1])
    rho_transposed = rho.transpose(2, 1, 0)                                                       # Transpose to (d, D, l)
    rho_reshaped = rho_transposed.reshape(-1, rho_transposed.shape[1] * rho_transposed.shape[2])  # Reshape to (d, Dl)
    mpo_transposed = mpo[n-1].transpose(0, 2, 1)                                                  # Transpose to (D, l, o)
    mpo_reshaped = mpo_transposed.reshape(mpo_transposed.shape[0] * mpo_transposed.shape[1], -1)  # Reshape to (Dl, o)
    rho = rho_reshaped @ mpo_reshaped                                                             # Resulting shape (d, o)
    rho = rho.T                                                                                   # Transpose to (o, d)

    # rho = np.einsum('XDFZ,Zl->XDFl', L[n-2], mps_c[n-1])
    L_reshaped = L[n-2].reshape(-1, L[n-2].shape[3])                                        # Reshape to (XDF, Z)
    mps_reshaped = mps_c[n-1].reshape(mps_c[n-1].shape[0], -1)                              # Reshape to (Z, l)
    intermediate = L_reshaped @ mps_reshaped                                   
    rho = intermediate.reshape(L[n-2].shape[0], L[n-2].shape[1], L[n-2].shape[2], mps_c[n-1].shape[1])  # Reshape to (X, D, F, l)

    # rho = np.einsum('XDFl,Fdl->XDd', rho, mpo_c[n-1])
    rho_reshaped = rho.reshape(rho.shape[0] * rho.shape[1], rho.shape[2] * rho.shape[3])    # Reshape to (XD, Fl)
    mpo_transposed = mpo_c[n-1].transpose(0, 2, 1)                                          # Transpose to (F, l, d)
    mpo_reshaped = mpo_transposed.reshape(mpo_transposed.shape[0] * mpo_transposed.shape[1], -1)      # Reshape to (Fl, d)
    temp = rho_reshaped @ mpo_reshaped                                                      # Resulting shape (XD, d)
    rho = temp.reshape(rho.shape[0], rho.shape[1], -1)                                      # Reshape to (X, D, d)

    # rho = np.einsum('XDd,Xl->lDd', rho, mps[n-1])
    rho_reshaped = rho.reshape(rho.shape[0], -1)                                            # Reshape to (X, Dd)
    mps_reshaped = mps[n-1].reshape(mps[n-1].shape[0], -1)                                  # Reshape to (X, l)
    intermediate = rho_reshaped.T @ mps_reshaped                                                 
    rho = intermediate.T.reshape(mps[n-1].shape[1], rho.shape[1], rho.shape[2])             # Reshape to (l, D, d)
    
    # rho = np.einsum('lDd,Dol->od', rho, mpo[n-1])
    rho_transposed = rho.transpose(2, 1, 0)                                                       # Transpose to (d, D, l)
    rho_reshaped = rho_transposed.reshape(-1, rho_transposed.shape[1] * rho_transposed.shape[2])  # Reshape to (d, Dl)
    mpo_transposed = mpo[n-1].transpose(0, 2, 1)                                                  # Transpose to (D, l, o)
    mpo_reshaped = mpo_transposed.reshape(mpo_transposed.shape[0] * mpo_transposed.shape[1], -1)  # Reshape to (Dl, o)
    rho = rho_reshaped @ mpo_reshaped                                                             # Resulting shape (d, o)
    rho = rho.T                                                                                   # Transpose to (o, d)

    # ==========================================
    # 3. [Eigendecomposition]
    # ==========================================
    _, U, _ = truncated_eigendecomposition(rho, stop)
    mps_out[n-1] = U.T 

    # ==========================================
    # 4. [Cap Construction II]
    # ==========================================
    # M_top = np.einsum("dk,Gdl->Gkl", U, mpo_c[n-1])  
    Ut = U.T                                                                                       # Transpose to (k, d)
    mpo_transposed = mpo_c[n-1].transpose(1, 0, 2)                                                 # Transpose to (d, G, l)
    mpo_reshaped = mpo_transposed.reshape(-1, mpo_transposed.shape[1] * mpo_transposed.shape[2])   # Reshape to (d, GL)
    M_top = Ut @ mpo_reshaped                                                                      # Resulting shape (k, GL)
    M_top = M_top.reshape(-1, mpo_transposed.shape[1], mpo_transposed.shape[2])                    # Reshape to (k, G, l)
    M_top = M_top.transpose(1, 0, 2)                                                               # Transpose to (G, k, l)

    # M_top = np.einsum("Gkl,Wl->WGk", M_top, mps_c[n-1]) 
    M_top_reshaped = M_top.reshape(M_top.shape[0] * M_top.shape[1], -1)                            # Reshape to (Gk, l)
    mps_transposed = mps_c[n-1].T                                                                  # Transpose to (l, W)
    temp = M_top_reshaped @ mps_transposed                                                         # Resulting shape (Gk, W)
    M_top = temp.reshape(M_top.shape[0], M_top.shape[1], -1)                                       # Reshape to (G, k, W)
    M_top = M_top.transpose(2, 0, 1)                                                               # Transpose to (W, G, k)

    _, U, _ = truncated_eigendecomposition(rho, stop)
    mps_out[n-1] = U.T 

    # ==========================================
    # 4. [Cap Construction II]
    # ==========================================
    # M_top = np.einsum("dk,Gdl->Gkl", U, mpo_c[n-1])  
    Ut = U.T                                                                                       # Transpose to (k, d)
    mpo_transposed = mpo_c[n-1].transpose(1, 0, 2)                                                 # Transpose to (d, G, l)
    mpo_reshaped = mpo_transposed.reshape(-1, mpo_transposed.shape[1] * mpo_transposed.shape[2])   # Reshape to (d, GL)
    M_top = Ut @ mpo_reshaped                                                                      # Resulting shape (k, GL)
    M_top = M_top.reshape(-1, mpo_transposed.shape[1], mpo_transposed.shape[2])                    # Reshape to (k, G, l)
    M_top = M_top.transpose(1, 0, 2)                                                               # Transpose to (G, k, l)

    # M_top = np.einsum("Gkl,Wl->WGk", M_top, mps_c[n-1]) 
    M_top_reshaped = M_top.reshape(M_top.shape[0] * M_top.shape[1], -1)                            # Reshape to (Gk, l)
    mps_transposed = mps_c[n-1].T                                                                  # Transpose to (l, W)
    temp = M_top_reshaped @ mps_transposed                                                         # Resulting shape (Gk, W)
    M_top = temp.reshape(M_top.shape[0], M_top.shape[1], -1)                                       # Reshape to (G, k, W)
    M_top = M_top.transpose(2, 0, 1)                                                               # Transpose to (W, G, k)

    for j in reversed(range(1, n-1)):
        # top = np.einsum("WGk,ZlW->ZlGk", M_top, mps_c[j])
        M_top_transposed = M_top.transpose(1, 2, 0)                                                # Transpose to (G, k, W)
        M_top_reshaped = M_top_transposed.reshape(-1, M_top_transposed.shape[2])                   # Reshape to (Gk, W)
        mps_transposed = mps_c[j].transpose(2, 0, 1)                                               # Transpose to (W, Z, l)
        mps_reshaped = mps_transposed.reshape(mps_transposed.shape[0], -1)                         # Reshape to (W, Zl)
        top = M_top_reshaped @ mps_reshaped                                                        # Resulting shape (Gk, Zl)
        top = top.T                                                                                # Transpose to (Zl, Gk)
        top = top.reshape(mps_transposed.shape[1], mps_transposed.shape[2], 
                        M_top_transposed.shape[0], M_top_transposed.shape[1])                      # Reshape to (Z, l, G, k)

        # top = np.einsum("ZlGk,FdGl->ZFdk", top, mpo_c[j])
        top_transposed = top.transpose(0, 3, 2, 1)                                                 # Transpose to (Z, k, G, l)
        top_reshaped = top_transposed.reshape(top_transposed.shape[0] * top_transposed.shape[1], -1)  # Reshape to (Zk, Gl)
        mpo_transposed = mpo_c[j].transpose(2, 3, 0, 1)                                            # Transpose to (G, l, F, d)
        mpo_reshaped = mpo_transposed.reshape(mpo_transposed.shape[0] * mpo_transposed.shape[1], -1)  # Reshape to (Gl, Fd)
        top = top_reshaped @ mpo_reshaped                                                          # Resulting shape (Zk, Fd)
        top = top.reshape(top_transposed.shape[0], top_transposed.shape[1], 
                        mpo_transposed.shape[2], mpo_transposed.shape[3])                          # Reshape to (Z, k, F, d)
        top = top.transpose(0, 2, 3, 1)                                                            # Transpose to (Z, F, d, k)
        bottom = np.conj(top)
        
        # ==========================================
        # 5. [Density Matrix j]
        # ==========================================
        # rho = np.einsum("XDFZ,XDdk->kdFZ", L[j-1], bottom) 
        L_transposed = L[j-1].transpose(2,3,0,1) # Transpose to (F,Z,X,D)
        L_reshaped = L_transposed.reshape(L_transposed.shape[0]*L_transposed.shape[1],-1) #Reshape to (FZ,XD)
        bottom_reshaped = bottom.reshape(bottom.shape[0]*bottom.shape[1],-1) #Reshape to (XD ,dk)
        temp = L_reshaped @ bottom_reshaped # Resulting shape (FZ,dk)
        rho=temp.reshape(L_transposed.shape[0],L_transposed.shape[1],bottom.shape[2],bottom.shape[3]) #Reshape to (F,Z,d,k)
        rho = rho.transpose(3,2,0,1) #Transpose to (k,d,F,Z)
        
        # rho = np.einsum("kdFZ,ZFej->kdje", rho, top)
        rho_reshaped = rho.reshape(-1, rho.shape[2] * rho.shape[3])                                # Reshape to (kd, FZ)
        top_transposed = top.transpose(1, 0, 2, 3)                                                 # Transpose to (F, Z, e, j)
        top_reshaped = top_transposed.reshape(-1, top_transposed.shape[2] * top_transposed.shape[3])  # Reshape to (FZ, ej)
        temp = rho_reshaped @ top_reshaped                                                         # Resulting shape (kd, ej)
        rho = temp.reshape(rho.shape[0], rho.shape[1], top.shape[2], top.shape[3])                 # Reshape to (k, d, e, j)
        rho = rho.transpose(0, 1, 3, 2)                                                            # Transpose to (k, d, j, e)
         
        k_old = rho.shape[0]
        physdim = rho.shape[1]
        rho = rho.reshape(rho.shape[0] * rho.shape[1], rho.shape[2] * rho.shape[3])
        _, U, _ = truncated_eigendecomposition(rho, stop)
        U = U.reshape(k_old,physdim,U.shape[1])
        
        # U = np.einsum("kdl->ldk",U)
        U = U.transpose(2, 1, 0)  # Transpose the dimensions from (k, d, l) to (l, d, k)
        mps_out[j] = U

        # ==========================================
        # 6. [Cap Construction j]
        # ==========================================
        # M_top = np.einsum("ZFdk,ldk->ZFl", top, U)
        top_reshaped = top.reshape(top.shape[0], top.shape[1], -1)                    # Reshape to (ZF, dk)
        U_reshaped = U.reshape(U.shape[0], -1)                                        # Reshape to (l, dk)
        intermediate = top_reshaped @ U_reshaped.T                                    
        M_top = intermediate.reshape(top.shape[0], top.shape[1], U.shape[0])          # Reshape to (Z, F, l)

    # mps_out[0] = np.einsum('lZ,ZFk->lFk', mps_c[0], M_top)
    mps_reshaped = mps_c[0].reshape(-1, mps_c[0].shape[1])                                    # Reshape to (l, Z)
    M_top_reshaped = M_top.reshape(M_top.shape[0], -1)                                        # Reshape to (Z, Fk)
    intermediate = mps_reshaped @ M_top_reshaped                                             
    mps_out[0] = intermediate.reshape(mps_c[0].shape[0], M_top.shape[1], M_top.shape[2])      # Reshape to (l, F, k)

    # mps_out[0] = np.einsum('lFk,dFl->dk', mps_out[0], mpo_c[0])
    mps_out_transposed = mps_out[0].transpose(2, 1, 0)                                        # Transpose to (k, F, l)
    mps_out_reshaped = mps_out_transposed.reshape(-1, mps_out_transposed.shape[1] * mps_out_transposed.shape[2])  # Reshape to (k, Fl)
    mpo_transposed = mpo_c[0].transpose(1, 2, 0)                                              # Transpose to (F, l, d)
    mpo_reshaped = mpo_transposed.reshape(-1, mpo_transposed.shape[2])                        # Reshape to (Fl, d)
    mps_out[0] = mps_out_reshaped @ mpo_reshaped                                              # Resulting shape (k, d)
    mps_out[0] = mps_out[0].T                                                                 # Transpose to (d, k)

    
    return MPS(mps_out)

def zipup(mpo, mps, stop=Cutoff(1e-14), finalround=False, conditioning=False):
    
    # Einsum conventions:
    #   MPS[i] bond dimension X_i            = X
    #   MPS[i] bond dimension X_{i+1}        = Y
    #   MPO[i] bond dimension D_i            = D
    #   MPO[i] bond dimension D_{i+1}        = E
    #   MPO[i] physical dimension d_(top)    = d
    #   MPO[i] physical dimension d_(bottom) = l
    #   DX rank trunction p_(current)        = p
    #   DX rank trunction p_(previous)       = x
    
    if conditioning:
        mpo.canonize_left()
        mps.canonize_left()

    n = mpo.N
    mps_out = [None] * n
    physdim = mpo[0].shape[0]

    # =============================== First tensor ===============================
    C = np.tensordot(mpo[0], mps[0], axes=(2, 0))
    mps_out[0], S, Vt = truncated_svd(np.reshape(C, (C.shape[0], C.shape[1] * C.shape[2])), stop=stop)
    Z = np.diag(S) @ Vt
    Z = Z.reshape(Z.shape[0], mpo[1].shape[0], mps[1].shape[0])

    # =============================== Middle tensors =============================
    for i in range(1, n - 2):
        #C = np.einsum("pDX,XlY->pDlY", Z, mps[i])
        C = Z @ mps[i].reshape(mps[i].shape[0], -1)
        C = C.reshape(Z.shape[0], mpo[i].shape[0], mps[i].shape[1], mps[i].shape[2])
        
        #C = np.einsum('pDlY,DdEl->pdEY', C, mpo[i])
        C_transposed = C.transpose(0, 3, 1, 2)                                                     # Transpose to (p, Y, D, l)
        mpo_transposed = mpo[i].transpose(0,3, 1, 2)                                               # Transpose to (D, l, d, E)

        C_reshaped = C_transposed.reshape(-1, C_transposed.shape[2]*C_transposed.shape[3])         # Reshape to (pY, Dl)
        mpo_reshaped = mpo_transposed.reshape(mpo_transposed.shape[0]*mpo_transposed.shape[1],-1)  # Reshape to (Dl, dE)

        temp = (C_reshaped @ mpo_reshaped)
        temp=temp.reshape(C.shape[0], C.shape[3],mpo[i].shape[1], mpo[i].shape[2])                 # Reshape to (p, Y, d, E)
        C=temp.transpose(0,2,3,1)                                                                  # Transpose to (p, d, E, Y)
                
        U, S, Vt = truncated_svd(C.reshape(C.shape[0] * C.shape[1], C.shape[2] * C.shape[3]), stop=stop)
        U = U.reshape(C.shape[0], physdim, U.shape[1])
        mps_out[i] = U
        Z = np.diag(S) @ Vt
        Z = Z.reshape(Z.shape[0], mpo[i + 1].shape[0], mps[i + 1].shape[0])

    # =============================== Last tensor ================================
    
    #C = np.einsum("pDX,XlY->pDlY", Z, mps[n-2])
    Z_reshaped = Z.reshape(Z.shape[0], Z.shape[1], -1)                                             # Reshape to (p, D, X)
    mps_reshaped = mps[n-2].reshape(mps[n-2].shape[0], -1)                                         # Reshape to (X, lY)
    temp = Z_reshaped @ mps_reshaped 
    C = temp.reshape(Z.shape[0], Z.shape[1], mps[n-2].shape[1], mps[n-2].shape[2])                 # Reshape to (p, D, l, Y)
    
    #C = np.einsum("pDlY,DdEl->pdEY", C, mpo[n-2])
    C_transposed = C.transpose(0, 3, 1, 2)                                                         # Transpose to (p, Y, D, l)
    mpo_transposed = mpo[n-2].transpose(0, 3, 1, 2)                                                # Transpose to (D, l, d, E)
    C_reshaped = C_transposed.reshape(-1, C_transposed.shape[2]*C_transposed.shape[3])             # Reshape to (pY, Dl)
    mpo_reshaped = mpo_transposed.reshape(mpo_transposed.shape[0]*mpo_transposed.shape[1], -1)     # Reshape to (Dl, dE)
    temp = C_reshaped @ mpo_reshaped 
    temp = temp.reshape(C.shape[0], C.shape[3], mpo[n-2].shape[1], mpo[n-2].shape[2])              # Reshape to (p, Y, d, E)
    C = temp.transpose(0, 2, 3, 1)                                                                 # Transpose to (p, d, E, Y)
    
    #C = np.einsum("pdDX,Xl->pdDl", C, mps[n-1])
    C_reshaped = C.reshape(C.shape[0], C.shape[1], C.shape[2], -1)                                 # Reshape to (p, d, D, X)
    mps_reshaped = mps[n-1].reshape(-1, mps[n-1].shape[1])                                         # Reshape to (X, l)
    temp = C_reshaped @ mps_reshaped 
    C = temp.reshape(C.shape[0], C.shape[1], C.shape[2], mps[n-1].shape[1])                        # Reshape to (p, d, D, l)
    
    #C = np.einsum("pdDl,Dul->pdu", C, mpo[n-1])
    mpo_transposed = mpo[n-1].transpose(0, 2, 1)                                                   # Transpose to (D, l, u)
    C_reshaped = C.reshape(-1, C.shape[2]*C.shape[3])                                              # Reshape to (pd, Dl)
    mpo_reshaped = mpo_transposed.reshape(mpo_transposed.shape[0]*mpo_transposed.shape[1],-1)      # Reshape to (Dl, u)
    temp = C_reshaped @ mpo_reshaped
    C = temp.reshape(C.shape[0], C.shape[1], mpo[n-1].shape[1])                                    # Reshape to (p, d, u)

    #print(C.shape)
    U, S, Vt = truncated_svd(np.reshape(C, (C.shape[0] * C.shape[1], C.shape[1])), stop=stop)
    mps_out[n-2] = np.reshape(U, (C.shape[0], C.shape[2], U.shape[1]))
    Z = np.diag(S) @ Vt
    mps_out[-1] = Z

    mps_out = MPS(mps_out)
    if finalround:
        mps_out.round(stop=stop)

    return MPS(mps_out)

def zipup_randomsvd(mpo, mps,num_iters=1,power_iteration=0,oversample=5, stop=Cutoff(1e-14),finalround=False, conditioning=False):
    if conditioning:
        mpo.canonize_left()
        mps.canonize_left()

    n = mpo.N
    mps_out = [None] * n
    physdim = mpo[0].shape[0]

    # =============================== First tensor ===============================
    C = np.tensordot(mpo[0], mps[0], axes=(2, 0))
    mps_out[0], S, Vt = random_truncated_svd(np.reshape(C, (C.shape[0], C.shape[1] * C.shape[2])),num_iters,oversample,stop=stop)
    Z = np.diag(S) @ Vt
    Z = Z.reshape(Z.shape[0], mpo[1].shape[0], mps[1].shape[0])

    # =============================== Middle tensors =============================
    for i in range(1, n - 2):
        #C = np.einsum("pDX,XlY->pDlY", Z, mps[i])
        C = Z @ mps[i].reshape(mps[i].shape[0], -1)
        C = C.reshape(Z.shape[0], mpo[i].shape[0], mps[i].shape[1], mps[i].shape[2])
        
        #C = np.einsum('pDlY,DdEl->pdEY', C, mpo[i])
        C_transposed = C.transpose(0, 3, 1, 2)                                                     # Transpose to (p, Y, D, l)
        mpo_transposed = mpo[i].transpose(0,3, 1, 2)                                               # Transpose to (D, l, d, E)

        C_reshaped = C_transposed.reshape(-1, C_transposed.shape[2]*C_transposed.shape[3])         # Reshape to (pY, Dl)
        mpo_reshaped = mpo_transposed.reshape(mpo_transposed.shape[0]*mpo_transposed.shape[1],-1)  # Reshape to (Dl, dE)

        temp = (C_reshaped @ mpo_reshaped)
        temp=temp.reshape(C.shape[0], C.shape[3],mpo[i].shape[1], mpo[i].shape[2])                 # Reshape to (p, Y, d, E)
        C=temp.transpose(0,2,3,1)                                                                  # Transpose to (p, d, E, Y)
                
        U, S, Vt = random_truncated_svd(C.reshape(C.shape[0] * C.shape[1], C.shape[2] * C.shape[3]),num_iters,oversample, stop=stop)
        U = U.reshape(C.shape[0], physdim, U.shape[1])
        mps_out[i] = U
        Z = np.diag(S) @ Vt
        Z = Z.reshape(Z.shape[0], mpo[i + 1].shape[0], mps[i + 1].shape[0])

    # =============================== Last tensor ================================
    
    #C = np.einsum("pDX,XlY->pDlY", Z, mps[n-2])
    Z_reshaped = Z.reshape(Z.shape[0], Z.shape[1], -1)                                             # Reshape to (p, D, X)
    mps_reshaped = mps[n-2].reshape(mps[n-2].shape[0], -1)                                         # Reshape to (X, lY)
    temp = Z_reshaped @ mps_reshaped 
    C = temp.reshape(Z.shape[0], Z.shape[1], mps[n-2].shape[1], mps[n-2].shape[2])                 # Reshape to (p, D, l, Y)
    
    #C = np.einsum("pDlY,DdEl->pdEY", C, mpo[n-2])
    C_transposed = C.transpose(0, 3, 1, 2)                                                         # Transpose to (p, Y, D, l)
    mpo_transposed = mpo[n-2].transpose(0, 3, 1, 2)                                                # Transpose to (D, l, d, E)
    C_reshaped = C_transposed.reshape(-1, C_transposed.shape[2]*C_transposed.shape[3])             # Reshape to (pY, Dl)
    mpo_reshaped = mpo_transposed.reshape(mpo_transposed.shape[0]*mpo_transposed.shape[1], -1)     # Reshape to (Dl, dE)
    temp = C_reshaped @ mpo_reshaped 
    temp = temp.reshape(C.shape[0], C.shape[3], mpo[n-2].shape[1], mpo[n-2].shape[2])              # Reshape to (p, Y, d, E)
    C = temp.transpose(0, 2, 3, 1)                                                                 # Transpose to (p, d, E, Y)
    
    #C = np.einsum("pdDX,Xl->pdDl", C, mps[n-1])
    C_reshaped = C.reshape(C.shape[0], C.shape[1], C.shape[2], -1)                                 # Reshape to (p, d, D, X)
    mps_reshaped = mps[n-1].reshape(-1, mps[n-1].shape[1])                                         # Reshape to (X, l)
    temp = C_reshaped @ mps_reshaped 
    C = temp.reshape(C.shape[0], C.shape[1], C.shape[2], mps[n-1].shape[1])                        # Reshape to (p, d, D, l)
    
    #C = np.einsum("pdDl,Dul->pdu", C, mpo[n-1])
    mpo_transposed = mpo[n-1].transpose(0, 2, 1)                                                   # Transpose to (D, l, u)
    C_reshaped = C.reshape(-1, C.shape[2]*C.shape[3])                                              # Reshape to (pd, Dl)
    mpo_reshaped = mpo_transposed.reshape(mpo_transposed.shape[0]*mpo_transposed.shape[1],-1)      # Reshape to (Dl, u)
    temp = C_reshaped @ mpo_reshaped
    C = temp.reshape(C.shape[0], C.shape[1], mpo[n-1].shape[1])                                    # Reshape to (p, d, u)

    #print(C.shape)
    U, S, Vt = random_truncated_svd(np.reshape(C, (C.shape[0] * C.shape[1], C.shape[1])),num_iters,oversample, stop=stop)
    mps_out[n-2] = np.reshape(U, (C.shape[0], C.shape[2], U.shape[1]))
    Z = np.diag(S) @ Vt
    mps_out[-1] = Z

    mps_out = MPS(mps_out)
    if finalround:
        mps_out.round(stop=stop)

    return MPS(mps_out)

def fit(mpo, mps, max_sweeps=2, stop=Cutoff(1e-14), random_tensor=np.random.randn, guess=None):
    """
    Einsum Indexing convention L->R:
     ______                 ______
    |      |    _______    |      |   
    |      |-Z-|mps_c_j|-W-|      |
    |      |   |_______|   |      |
    |      |    __|d___    |      |
    | L[j] |-D-| mpo_j |-E-|R[N-j]|
    |      |   |_______|   |      |
    |      |    __|l___    |      |
    |      |-X-| mps_j |-Y-|      |
    |______|   |_______|   |______|
    """
    
    def right_sweep(mps, mpo, L, final_site=None,stop=Cutoff(1e-14)):
        mps_out = [None] * mps.N
        R = [None] * (mps.N - 2)

        if final_site is None:
            # site = np.einsum("XjY,Yl->Xjl", mps[-2], mps[-1]) 
            mps_reshaped = mps[-2].reshape(-1,mps[-2].shape[-1]) #Reshape to (Xj,Y)
            temp = mps_reshaped @ mps[-1] #Resulting shape (Xj,l)
            site = temp.reshape(mps[-2].shape[0],mps[-2].shape[1],temp.shape[-1]) #Reshape to (X,j,l)
            
            # site = np.einsum("Xjl,Edl->XjEd", site, mpo[-1])
            site_reshaped =site.reshape(-1,site.shape[-1]) #Reshape to (Xj,l)
            mpo_transposed = mpo[-1].transpose(2,0,1) #Transpose to (l,E,d)
            mpo_reshaped = mpo_transposed.reshape(mpo[-1].shape[0],-1) #Reshape to (l,Ed)
            temp = site_reshaped @ mpo_reshaped #Resulting shape (Xj,Ed)
            site = temp.reshape(site.shape[0],site.shape[1],mpo[-1].shape[0],mpo[-1].shape[1])
            
            
            # site = np.einsum("XjEk,DdEj->XDkd", site, mpo[-2])
            site_transposed = site.transpose(0,3,1,2) #Transpose to (X,k,j,E)
            site_reshaped = site_transposed.reshape(-1,site_transposed.shape[2]*site_transposed.shape[3]) #reshape to (Xk,jE)
            mpo_transposed = mpo[-2].transpose(3,2,0,1) #transpose to (j,E,D,d)
            mpo_reshaped =  mpo_transposed.reshape(-1,mpo_transposed.shape[2]*mpo_transposed.shape[3]) #reshape to (jE,Dd)
            temp = site_reshaped @ mpo_reshaped #resulting shape (Xk,Dd)
            temp = temp.reshape(site.shape[0],site.shape[3],mpo[-2].shape0,mpo[-2].shape[1]) #Reshape to (X,k,D,d)
            site = temp.transpose(0,2,1,3) #Transpose to (X,D,k,d)
            
            # site = np.einsum("pDX,XDkd->pdk", L[-1], site)
            L_transposed = L[k].transpose(0,2,1) #Transpose to p,X,D
            L_reshaped = L_transposed.reshape(L_transposed.shape[0],-1) #Reshape to (p,XD)
            site_reshaped = site.reshape (-1,site.shape[2]*site.shape[3] )#Reshape to (XD,kd)
            temp = L_reshaped @ site_reshaped #resulting shape p,kd
            temp = temp.reshape(temp.shape[0],site.shape[2],site.shape[3]) #Reshape to (p,k,d)
            site = temp.transpose(0,2,1) #Transpose to (p,d,k)
            
        else:
            site = final_site

        U, S, Vt = truncated_svd(site.reshape(site.shape[0] * site.shape[1], site.shape[2]), stop=stop)
        mps_out[-1] = (Vt).reshape(U.shape[1], mps[-1].shape[1])

        # R[-1] = np.einsum("Ddl,Xl->XDd", mpo[-1], mps[-1])
        mpo_reshaped = mpo[-1].reshape(-1,mpo[-1].shape[-1]) #reshape to (Dd,l)
        temp = mpo_reshaped @ mps[-1].T #Resulting shape (Dd,X)
        temp = temp.reshape (mpo[-1].shape[0],mps[-1].shape[1],temp.shape[1])#Reshape to (D,d,X)
        R[-1] = temp.transpose (2,0,1) # Transpose to (X,D,d)
        
        # R[-1] = np.einsum("XDd,pd->XDp", R[-1], mps_out[-1])
        R_reshaped = R[-1].reshape(-1,R[-1].shape[-1]) #Reshape to (XD,d)
        temp = R_reshaped @ mps_out[-1].T #Resulting shape (XD,p)
        R[-1] = temp.reshape (R[-1].shape[0],R[-1].shape[1],temp.shape[-1]) #Reshape to (X,D,p)

        # Middle sites.
        for k in range(mps.N - 2, 1, -1):
            # site = np.einsum("XDp,YlX->YlDp", R[k - 1], mps[k])
            R_transposed = R[k-1].transpose(1,2,0) # transpose to (D,p,X)
            R_reshaped = R_transposed.reshape (-1,R_transposed.shape[-1]) #Reshape to (Dp,X)
            mps_transposed =mps[k].transpose(2,0,1) #Transpose to (X,Y,l)
            mps_reshaped = mps_transposed.reshape(mps_transposed.shape[0],-1) #Reshape to (X, Yl)
            temp = R_reshaped @ mps_reshaped #Resulting shape (Dp,Yl)
            temp = temp.reshape(R[k-1].shape[1],R[k-1].shape[2],mps[k].shape[0],mps[k].shape[1]) #Reshape to (D,p,Y,l)
            site = temp.transpose(2,3,0,1) #Transpose to (Y,l,D,p)
            
            # site = np.einsum("YlDp,EdDl->YEdp", site, mpo[k])
            site_transposed = site.transpose(0,3,1,2) #Transpose to (Y,p,l,D)
            site_reshaped = site_transposed.reshape(-1,site_transposed.shape[2]*site_transposed.shape[3]) #Reshape to (Yp,lD)
            mpo_transposed = mpo[k].transpose (3,2,0,1) #Transpose to (l,D,E,d)
            mpo_reshaped = mpo_transposed.reshape(-1,mpo_transposed.shape[2]*mpo_transposed.shape[3]) #Reshape to (lD,Ed)
            temp = site_reshaped @ mpo_reshaped #resulting shape (Yp,Ed)
            temp = temp.reshape(site.shape[0],site.shape[3],mpo[k].shape[0],mpo[k].shape[1]) #reshape to (Y,p,E,d)
            site = temp.transpose (0,2,3,1) #Transpose to ( Y,E,d,p)
            
            # site = np.einsum("YEdp,ZlY->ZlEdp", site, mps[k - 1])
            site_transposed = site.transpose(1,2,3,0) #Transpose to (E,d,p,Y)
            site_reshaped = site_transposed.reshape(-1,site_transposed.shape[-1]) #Reshape to (Edp,Y)
            mps_transposed = mps[k-1].transpose(2,0,1) #transpose to (Y,Z,l)
            mps_reshaped = mps_transposed.reshape(mps_transposed.shape[0],-1) #reshape to (Y,Zl)
            temp = site_reshaped @ mps_reshaped #resulting shape (Edp,Zl)
            temp = temp.reshape(site.shape[1],site.shape[2],site.shape[3],mps[k-1].shape[0],mps[k-1].shape[1]) #Reshape to (E,d,p,Z,l)
            site = temp.transpose(3,4,0,1,2) #Transpose to (Z,l,E,d,p)
            
            # site = np.einsum("ZlEdp,FkEl->ZFkdp", site, mpo[k - 1])
            site_transposed = site.transpose(0,3,4,1,2) #Transpose to (Z,d,p,l,E)
            site_reshaped = site_transposed.reshape(-1,site_transposed.shape[3]*site_transposed.shape[4]) # Reshape to (Zdp,lE)
            mpo_transposed = mpo[k-1].transpose(3,2,0,1) #Transpose to (l,E,F,k)
            mpo_reshaped = mpo_transposed.reshape(-1,mpo_transposed.shape[2]*mpo_transposed.shape[3]) #Reeshape to (le,Fk)
            temp = site_reshaped @ mpo_reshaped #Resulting shape (Zdp,Fk)
            temp = temp.reshape(site.shape[0],site.shape[3],site.shape[4],mpo[k-1].shape[0],mpo[k-1].shape[1]) #Reshape to (Z,d,p,F,k)
            site = temp.transpose (0,3,4,1,2) #Transpose to (Z,F,k,d,p)
            
     
            # check = np.einsum("ZFkdp,qFZ->qkdp", site, L[k - 2])
            site_transposed = site.transpose(2,3,4,1,0) #Transpose (k,d,p,F,Z)
            site_reshaped = site_transposed.reshape(-1,site_transposed.shape[3]*site_transposed.shape[4]) #reshape to (kdp,FZ)
            L_transposed = L[k-2].transpose(1,2,0) #Transpose to (F,K,q)
            L_reshaped = L_transposed.reshape(-1,L_transposed.shape[2]) #Reshape to (FK,q)
            temp = site_reshaped @ L_reshaped # Resulting shape (kdp,q)
            temp = temp.reshape(site.shape[2],site.shape[3],site.shape[4],L[k-2].shape[0]) # (k,d,p,q)
            site = temp.transpose(3,0,1,2) #Transpose to (q,k,d,p)

            U, S, Vt = truncated_svd(site.reshape(site.shape[0] * site.shape[1], site.shape[2] * site.shape[3]), stop=stop)
            mps_out[k] = (Vt).reshape(U.shape[1], mps[-1].shape[1], site.shape[3])

            
            # R[k - 2] = np.einsum("XDp,qdp->XDdq", R[k - 1], mps_out[k])
            R_reshaped = R[k-1].reshape(-1,R[k-1].shape[-1]) #Reshape to (XD,p)
            mps_transposed = mps_out[k].transpose(2,0,1) #Transpose to (p,q,d)
            mps_reshaped = mps_transposed.reshape(mps_transposed.shape[0],-1) #Reshape to (p,qd)
            temp = R_reshaped @ mps_reshaped #Resulting shape (XD,qd)
            temp = temp.reshape(R[k-1].shape[0],R[k-1].shape[1],mps_out[k].shape[0],mps_out[k].shape[1]) #Reshape to (X,D,q,d)
            R[k-2] = temp.transpose(0,1,3,2) #Transpose tp (X,D,d,q)
            
            # R[k - 2] = np.einsum("XDdq,EdDl->XlEq", R[k - 2], mpo[k])
            R_transposed = R[k-2].transpose(0,3,1,2) #Transpose to (X,q,D,d)
            R_reshaped = R_transposed.reshape(-1,R_transposed.shape[2]*R_transposed.shape[3]) #Reshape to (Xq,Dd)
            mpo_transposed = mpo[k].transpose(2,1,0,3) # Transpose to (D,d, E,l)
            mpo_reshaped = mpo_transposed.reshape (-1, mpo_transposed.shape[2]*mpo_transposed.shape[3]) #Reshape tp (Dd,El)
            temp = R_reshaped @ mpo_reshaped #Resulting shape (Xq,El)
            temp = temp.reshape(R[k-2].shape[0],R[k-2].shape[3],mpo[k].shape[0],mpo[k].shape[3]) #Reshape tp (X,q,E,l)
            R[k-2] = temp.transpose(0,3,2,1) #Transpose to (X,l,E,q)
            
            # R[k - 2] = np.einsum("XlEq,YlX->YEq", R[k - 2], mps[k])
            R_transposed = R[k-2].transpose(2,3,0,1) #Transpose to (E,q,X,l)
            R_reshaped = R_transposed.reshape(-1,R_transposed.shape[2]*R_transposed.shape[3]) #Reshape to (Eq,Xl)
            mps_transposed = mps[k].transpose(2,1,0) #Transpose to (X,l,Y)
            mpo_reshaped = mps_transposed.reshape(-1,mps_transposed.shape[-1]) #reshape to (Xl,Y)
            temp = R_reshaped @mpo_reshaped # Resulting shape (Eq,Y)
            temp = temp.reshape(R[k-2].shape[2],R[k-2].shape[3],mps[k].shape[0]) #Reshape to (E,q,Y)
            R[k-2] = temp.transpose (2,0,1) #Transpose to (Y,E,q)

        # site = np.einsum("lX,XkY->lkY",mps[0],mps[1])
        mps_reshaped = mps[1].reshape(mps[1].shape[0],-1) #Reshape to ( X,kY)
        temp = mps[0] @ mps_reshaped #Resulting shape (l,kY)
        site = temp.reshape(mps[0].shape[0],mps[1].shape[1],mps[1].shape[2]) #Reshape to (l,k,Y)
        
        # site = np.einsum("lkY,dDl->dDkY",site,mpo[0])
        site_transposed = site.transpose(1,2,0) #Transpose to (k,Y,l)
        site_reshaped = site_transposed.reshape (-1,site_transposed.shape[-1]) #Reshape to (kY,l)
        mpo_transposed = mpo[0].transpose(2,0,1) #Transpose to (l,d,D)
        mpo_reshaped = mpo_transposed.reshape(mpo_transposed.shape[0],-1)#Reshape to (l,dD)
        temp = site_reshaped @ mpo_reshaped #resulting shape (kY,dD)
        temp = temp.reshape(site.shape[1],site.shape[2],mpo[0].shape[0],mpo[0].shape[1]) #Reshape to (k,Y,d,D)
        site = temp.transpose(2,3,0,1) #Transpose to (d,D,k,Y)
        
        # site = np.einsum("kDlY,DdEl->kdEY",site,mpo[1])
        site_transposed = site.transpose(0,3,1,2) #Transpose to (k,Y,D,l)
        mpo_transposed = mpo[1].transpose(0,3,1,2) #Transpose to (D,l,d,E)
        site_reshaped = site_transposed.reshape(site_transposed.shape[0]*site_transposed.shape[1],-1) #Reshape to (kY,Dl)
        mpo_reshaped = mpo_transposed.reshape(mpo_transposed.shape[0]*mpo_transposed.shape[1],-1) #Reshape tp (Dl,dE)
        temp = site_reshaped @ mpo_reshaped #Resulting shape (kY,dE)
        temp = temp.reshape(site.shape[0],site.shape[3],mpo[1].shape[1],mpo[1].shape[2]) #Reshape to (k,Y,d,E)
        site = temp.transpose(0,2,3,1) #Transpose to (k,d,E,Y)
        
        # final_site = np.einsum("kdEY,YEq->kdq",site,R[0])
        site_reshaped= site.reshape(site.shape[0]*site.shape[1],-1) #reshape to (kd,EY)
        R_transposed = R[0].transpose(1,0,2) #Transpose to ( E,Y,q)
        R_reshaped = R_transposed.reshape(-1,R_transposed.shape[-1]) #Reshape to (EY,q)
        temp  = site_reshaped @ R_reshaped #Reusulting shape (kd,q)
        final_site = temp.reshape(site.shape[0],site.shape[1],temp.shape[-1]) #Reshape tp (k,d,q)
        
        U, S, Vt = truncated_svd(final_site.reshape(final_site.shape[0], final_site.shape[1] * final_site.shape[2]), stop=stop)
        mps_out[1] = Vt.reshape(Vt.shape[0], mps[0].shape[0], R[0].shape[2])
        mps_out[0] = U @ np.diag(S)

        return R, final_site, MPS(mps_out)
    
    def left_sweep(mps, mpo, R, final_site=None,stop=Cutoff(1e-14)):
            mps_out = [None] * mps.N
            L = [None] * (mps.N - 2)

            # First Local Site from the left 
            if final_site is None:
                # site = np.einsum("lX,XjY->ljY", mps[0], mps[1])
                mps_reshaped = mps[1].reshape(mps[1].shape[0],-1)# Reshape to (X,jY)
                temp = mps[0] @ mps_reshaped                        #Resulting shape (l,jY)
                site = temp.reshape(temp.shape[0],mps[1].shape[1],mps[1].shape[2])# Reshape to l,j,Y
                
                # site = np.einsum("ljY,dDl->dDjY", site, mpo[0])
                site_transposed = site.transpose(1,2,0) #Transpose to (j,Y,l)
                site_reshaped = site_transposed.reshape(-1,site_transposed.shape[-1]) # reshape to (jY,l)
                mpo_transposed = mpo[0].transpose(2,0,1) # Transpose to (l,d,D)
                mpo_reshaped = mpo_transposed.reshape (mpo_transposed.shape[0],-1) #reshape to (l,dD)
                temp = site_reshaped @ mpo_reshaped #Resulting shape (jY,dD)
                temp = temp.reshape (site.shape[1],site.shape[2],mpo[0].shape[0],mpo[0].shape[1]) #Reshape to (j,Y,d,D)
                site = temp.transpose (2,3,0,1) # Transpose to (d,D,j,Y)
            
                # site = np.einsum("dDjY,DkEj->dkEY", site, mpo[1]) #TODO fix me 
                site_transposed = site.transpose(0,3,1,2) #Transpose to (d,Y,D,j)
                mpo_transposed = mpo[1].transpose(0,3,1,2) #Transpose to (D,j,k,E)
                site_reshaped = site_transposed.reshape(-1,site_transposed.shape[2]*site_transposed.shape[3]) #Reshape to (dY,Dj)
                mpo_reshaped = mpo_transposed.reshape(-1,mpo_transposed.shape[2]*mpo_transposed.shape[3]) #Reshape to (Dj,kE)
                temp = site_reshaped @ mpo_reshaped #Resulting shape (dY,kE)
                temp = temp.reshape(site.shape[0],site.shape[3],mpo[1].shape[1],mpo[1].shape[2]) #Reshape to (d,Y,k,E)
                site = temp.transpose(0,2,3,1) # transpose to (d,k,E,Y)
                
                # site = np.einsum("dkEY,YEW->dkW", site, R[0])
                site_transposed = site.transpose(0,1,3,2) #transpose to (d,k,Y,E)
                site_reshaped = site_transposed.reshape(-1,site_transposed.shape[2]*site_transposed.shape[3]) #Reshape to (dk,YE)
                R_reshaped = R[0].reshape(R[0].shape[0]*R[0].shape[1],-1)# Reshape to (YE,W)
                temp = site_reshaped @ R_reshaped #Resulting shape (dk,W)
                site = temp.reshape(site.shape[0],site.shape[1],R[0].shape[2])
                
            else:
                site = final_site
            U, S, Vt = truncated_svd(site.reshape(site.shape[0], site.shape[1] * site.shape[2]), stop=stop)
            mps_out[0] = U

            # L[0] = np.einsum("dp,dDl->pDl", U, mpo[0])
            mpo_reshaped =mpo[0].reshape(mpo[0].shape[0],-1) #reshape to (d,Dl)
            temp = U.T @ mpo_reshaped #Resulting shape (p,Dl)
            L[0] = temp.reshape(temp.shape[0],mpo[0].shape[1],mpo[0].shape[2])
            
            # L[0] = np.einsum("pDl,lX->pDX", L[0], mps[0])
            L_reshaped = L[0].reshape(-1,L[0].shape[2]) #Reshape to (pD,l)
            temp = L_reshaped @ mps[0] #Resulting shape (pD,X)
            L[0] = temp.reshape(L[0].shape[0],L[0].shape[1],mps[0].shape[1]) # Reshape to (p,D,X)

            # Sweep through the remaining sites
            for k in range(1, len(mps) - 2):
                # site = np.einsum("pDX,XlY->pDlY", L[k - 1], mps[k]) 
                L_reshaped = L[k-1].reshape(-1,L[k-1].shape[2]) # Reshape to (pD,X)
                mps_reshaped = mps[k].reshape(mps[k].shape[0],-1) #Reshape to (X,lY)
                temp = L_reshaped @mps_reshaped #Resulting shape (pD,lY)
                site= temp.reshape (L[k-1].shape[0],L[k-1].shape[1],mps[k].shape[1],mps[k].shape[2]) #Reshape to (p,D,l,Y)
                
                # site = np.einsum("pDlY,DdEl->pdEY", site, mpo[k])
                site_transposed = site.transpose(0,3,1,2) #Transpose to ( p,Y,D,l)
                site_reshaped = site_transposed.reshape(-1,site_transposed.shape[2]*site_transposed.shape[3]) #Reshape to ( pY,Dl)
                mpo_transposed = mpo[k].transpose(0,3,1,2) #Transpose to (D,l,d,E)
                mpo_reshaped = mpo_transposed.reshape(mpo_transposed.shape[0]*mpo_transposed.shape[1],-1) # Reshape to (Dl,dE)
                temp = site_reshaped @ mpo_reshaped #Resulting shape (pY,dE)
                temp = temp.reshape(site.shape[0],site.shape[3],mpo[k].shape[1],mpo[k].shape[2]) #Reshape to ( p,Y,d,E)
                site = temp.transpose ( 0,2,3,1) #Trannspose to (p,d,E,Y)
                
                
                # site = np.einsum("pdEY,YlZ->pdElZ", site, mps[k + 1])
                site_reshaped = site.reshape(-1,site.shape[-1]) #Reshape to (pdE,Y)
                mps_reshaped = mps[k+1].reshape(mps[k+1].shape[0],-1) #reshape to (Y,lZ)
                temp = site_reshaped @ mps_reshaped #Resulting shape (pdE,lZ)
                site = temp.reshape(site.shape[0],site.shape[1],site.shape[2],mps[k+1].shape[1],mps[k+1].shape[2]) #Reshape to (p,d,E,l,Z)
                
                
                # site = np.einsum("pdElZ,EkFl->pdkFZ", site, mpo[k + 1])
                site_transposed = site.transpose(0,1,4,2,3) #Transpose to (p,d,Z,E,l)
                site_reshaped = site_transposed.reshape(-1,site_transposed.shape[3]*site_transposed.shape[4]) #Reshape to (pdZ,El)
                mpo_transposed = mpo[k+1].transpose(0,3,1,2) #Transpose to (E,l,k,F)
                mpo_reshaped = mpo_transposed.reshape(-1,mpo_transposed.shape[2]*mpo_transposed.shape[3]) #Reshape to (El,kF)
                temp = site_reshaped @ mpo_reshaped #Resulting shape (pdZ,kF)
                temp = temp.reshape(site.shape[0],site.shape[1],site.shape[4],mpo[k+1].shape[1],mpo[k+1].shape[2])#Reshape to (p,d,Z,k,F)
                site = temp.transpose(0,1,3,4,2) #Transpose to (p,d,k,F,Z)
                
                # site = np.einsum("pdkFZ,ZFW->pdkW", site, R[k])
                site_reshaped = site.reshape(-1,site.shape[3]*site.shape[4]) #Reshape to (pdk,FZ)
                R_transposed = R[k].transpose(1,0,2) # Transpose to (Z,F,W)
                R_reshaped  = R_transposed.reshape(-1,R_transposed.shape[-1]) #Reshape to (ZF,W)
                temp = site_reshaped @ R_reshaped #Resulting shape (pdk,W)
                site = temp.reshape(site.shape[0],site.shape[1],site.shape[2],R[k].shape[-1]) #Reshape to (p,d,k,W)
                
                U, _, _ = truncated_svd(site.reshape(site.shape[0] * site.shape[1], site.shape[2] * site.shape[3]), stop=stop)
                mps_out[k] = U.reshape(site.shape[0], mpo[k].shape[1], U.shape[1])

                # L[k] = np.einsum("pDX,pdq->qdDX", L[k - 1], mps_out[k])
                L_transposed = L[k-1].transpose(1,2,0) #Transpose to (D,X,l)
                L_reshaped = L_transposed.reshape(-1,L_transposed.shape[-1] )#Reshape to (Dx,p)
                mps_reshaped = mps_out[k].reshape(mps_out[k].shape[0],-1) #Reshape to (p,dq)
                temp = L_reshaped @ mps_reshaped #Resulting shape (DX,dq)
                temp = temp.reshape(L[k-1].shape[1],L[k-1].shape[2],mps_out[k].shape[1],mps_out[k].shape[-1]) #Reshape to (D,X,d,q)
                L[k] = temp.transpose(3,2,0,1) #Transpose to (q,d,D,X)
                
                # L[k] = np.einsum("qdDX,DdEl->qElX", L[k], mpo[k])
                L_transposed = L[k].transpose(0,3,2,1) #Transpose to (q,X,D,d)
                L_reshaped = L_transposed.reshape(-1,L_transposed.shape[2]*L_transposed.shape[3] )#Reshape to (qX,Dd)
                mpo_reshaped = mpo[k].reshape(-1,mpo[k].shape[2]*mpo[k].shape[3]) #Reshape to (Dd,El)
                temp = L_reshaped @  mpo_reshaped #resulting shape (qX,El)
                temp = temp.reshape(L[k].shape[0],L[k].shape[3],mpo[k].shape[2],mpo[k].shape[3]) #Reshape to (q,X,E,l)
                L[k] = temp.transpose(0,2,3,1) #Transpose to (q,E,l,X)
                
                # L[k] = np.einsum("qElX,XlY->qEY", L[k], mps[k])
                L_reshaped = L[k].reshape(-1,L[k].shape[2]*L[k].shape[3]) #Reshape to (qE,lX)
                mps_transposed = mps[k].transpose(1,0,2) # Transpose to (l,X,Y)
                mps_reshaped = mps_transposed.reshape(-1,mps_transposed.shape[-1]) #Reshape to (lX,Y)
                temp = L_reshaped @mps_reshaped #Resulting shape (qE,Y)
                L[k]= temp.reshape(L[k].shape[0],L[k].shape[1],mps[k].shape[2])

            # Final two sites
            
            # site = np.einsum("XlY,Yk->Xlk",mps[-2],mps[-1])
            mps_reshaped = mps[-2].reshape(-1,mps[-2].shape[-1]) #Reshape to (Xl,Y)
            temp = mps_reshaped  @ mps[-1] #Resulting shape (Xl,k)
            site = temp.reshape(mps[-2].shape[0],mps[-2].shape[1],mps[1].shape[1]) #Reshape to (X,l,k)
   
            
            # site = np.einsum("Xlk,Edk->XlEd",site,mpo[-1])
            site_reshaped = site.reshape(-1,site.shape[-1]) #Reshape to (Xl,k)
            mpo_transposed = mpo[-1].transpose(2,0,1) #Transpose to (k,E,d)
            mpo_reshaped = mpo_transposed.reshape(mpo_transposed.shape[0],-1) #Reshape to (k,Ed)
            temp = site_reshaped @ mpo_reshaped #Resulting shape (Xl,Ed)
            site = temp.reshape(site.shape[0],site.shape[1],mpo[-1].shape[0],mpo[-1].shape[1]) #Reshape to (Xl,Ed)
            
            # site = np.einsum("XlEd,DkEl->XDdk",site,mpo[-2])
            site_transposed = site.transpose(0,3,1,2) #Tranpose to (X,d,l,E)
            mpo_transposed = mpo[-2].transpose(3,2,0,1) #Transpose to (l,E,D,k)
            site_reshaped = site_transposed.reshape(site_transposed.shape[0]*site_transposed.shape[1],-1) #Reshape to (Xd,lE)
            mpo_reshaped = mpo_transposed.reshape(mpo_transposed.shape[0]*mpo_transposed.shape[1],-1) #Reshape to (lE,Dk)
            temp = site_reshaped @mpo_reshaped #Resulting shape (Xd,Dk)
            temp = temp.reshape(site.shape[0],site.shape[3],mpo[-2].shape[0],mpo[-2].shape[3]) #Reshape to (X,d,D,k)
            site = temp.transpose(0,2,1,3) #Transpose to (X,D,d,k)
            
            # final_site = np.einsum("qDX,XDdk->qdk",L[k],site)
            L_reshaped = L[k].reshape(L[k].shape[0],-1) #Reshape to (q,DX)
            site_transposed = site.transpose(1,0,2,3) #Transpose to (D,X,d,k)
            site_reshaped = site_transposed.reshape(site_transposed.shape[0]*site_transposed.shape[1],-1) #Reshape to (DX,dk)
            temp = L_reshaped @ site_reshaped #Resulting shape (q,dk)
            final_site = temp.reshape(temp.shape[0],site.shape[2],site.shape[3])
            
            U, S, Vt = truncated_svd(final_site.reshape(final_site.shape[0] * final_site.shape[1], final_site.shape[2]), stop=stop)
            mps_out[-2] = U.reshape(final_site.shape[0], mps[-2].shape[1], final_site.shape[2])
            mps_out[-1] = (np.diag(S) @ Vt).reshape(final_site.shape[2], mps[-1].shape[1])

            return L, final_site, MPS(mps_out)

    def compute_left_envs(mps, mpo, guess): #TODO: Convert to einsums if needed ie starting from right
            L = [None] * (mps.N - 2)
            # -------- Left environments --------
            L[0] = np.einsum("dDl,dZ->ZDl", mpo[0], guess[0])
            L[0] = np.einsum("ZDl,lX->ZDX", L[0], mps[0])

            for i in range(1, mps.N - 2):
                L[i] = np.einsum("ZDX,XlY->ZDlY", L[i - 1], mps[i])
                L[i] = np.einsum("ZDlY,DdEl->ZdEY", L[i], mpo[i])
                L[i] = np.einsum("ZdEY,ZdW->WEY", L[i], guess[i])
            return L

    def compute_right_envs(mps, mpo, guess):
            # Only needed if starting from left
            R = [None] * (mps.N - 2)
            # -------- Right environments --------
            
            # R[-1] = np.einsum("Ddl,Zd->ZDl", mpo[-1], guess[-1])
            mpo_transposed = mpo[-1].transpose(0,2,1)                               # Transpose to (D,l,d)
            mpo_reshaped = mpo_transposed.reshape(-1,mpo_transposed.shape[-1])      # Reshape to (Dl,d)
            temp = mpo_reshaped @ guess[-1].T                                       # Resulting shape to (Dl,Z)
            temp = temp.reshape(mpo[-1].shape[0],mpo[-1].shape[1],temp.shape[-1])   # Reshape to (D,l,Z)
            R[-1] = temp.transpose(2,0,1)                                           # Transpose to (Z,D,L)
            
            
            #R[-1] = np.einsum("ZDl,Xl->XDZ", R[-1], mps[-1])
            R_reshaped = R[-1].reshape(-1,R[-1].shape[-1])                           # Reshape to (ZD,l)
            temp = R_reshaped @ mps[-1].T                                            # Resulting shape (ZD,X)
            temp = temp.reshape(R[-1].shape[0],R[-1].shape[1],mps[-1].shape[0])      # Reshape to (Z,D,X)
            R[-1]=temp.transpose(2,1,0)
            

            for i in range(mps.N - 2, 1, -1):
                # R[i - 2] = np.einsum("XDZ,YlX->YlDZ", R[i - 1], mps[i])
                R_transposed = R[i-1].transpose(1,2,0)                              # Tranpose to (D,Z,X)
                R_reshaped = R_transposed.reshape(-1,R_transposed.shape[-1])        #Reshape (DZ,X)
                mps_transposed = mps[i].transpose(2,0,1)                            #Transpose to (X,Y,l)
                mps_reshaped = mps_transposed.reshape(mps_transposed.shape[0],-1)   # Reshape to (X,Yl)
                temp = R_reshaped @ mps_reshaped                                    #Resulting shape (DZ,Yl)
                temp = temp.reshape(R[i-1].shape[1],R[i-1].shape[2],mps[i].shape[0],mps[i].shape[1]) #Reshape to (D,Z,Y,l)
                R[i - 2] = temp.transpose(2,3,0,1)
                
                # R[i - 2] = np.einsum("YlDZ,EdDl->YEdZ", R[i - 2], mpo[i])
                R_transposed = R[i-2].transpose(0,3,1,2)                            # Transpose to (Y,Z,l,D)
                R_reshaped = R_transposed.reshape(-1,R_transposed.shape[2]*R_transposed.shape[3]) #Reshape to (YZ,lD)
                mpo_transposed = mpo[i].transpose(3,2,0,1)                          # Transpose to (l,D,E,d)
                mpo_reshaped = mpo_transposed.reshape(mpo_transposed.shape[0]*mpo_transposed.shape[1],-1) # reshape to (ld,Ed)
                temp = R_reshaped @ mpo_reshaped                                # Resulting shape (YZ,Ed)
                temp = temp.reshape(R[i-2].shape[0],R[i-2].shape[3],mpo[i].shape[0],mpo[i].shape[1]) #reshape to (Y,Z,E,d)
                R[i-2] = temp.transpose(0,2,3,1)
                
                # R[i - 2] = np.einsum("YEdZ,WdZ->YEW", R[i - 2], guess[i])
                R_reshaped = R[i-2].reshape(-1,R[i-2].shape[2]*R[i-2].shape[3])         # reshape to (YE,dZ)
                guess_transposed = guess[i].transpose(1,2,0)                            #Transpose to (d,Z,W)
                guess_reshaped = guess_transposed.reshape(-1,guess_transposed.shape[-1])#Reshape to (dZ,W)
                temp = R_reshaped @ guess_reshaped                                      #Resulting shape (YE,W)
                R[i-2] = temp.reshape(R[i-2].shape[0],R[i-2].shape[1],guess[i].shape[0])#Reshape to (Y,E,W)
            return R
        
    # Form a random MPS |ψB> of bond dimensionm 
    # states= [np.random.randn(mps[1].shape[0]) for i in range(mps.N)]
    # guess=MPS([np.reshape(states[0],(len(states[0]),1))] + [np.reshape(states[i],(1,len(states[i]),1)) for i in range (1,len(states)-1)] + [np.reshape(states[-1],(1,len(states[-1])))])
    
    if guess is None:
        guess = MPS.random_mps(n=mps.N, m=mps[0].shape[1], d=mps[0].shape[0], random_tensor=random_tensor)
    elif guess == "input":
        guess = mps.copy()

    # orthogonalize it to have any arbitrary orthogonality center.
    guess.canonize_left()

    R = compute_right_envs(mps, mpo, guess)


    final_site = None
    for sweep_count in range(max_sweeps):
        L, final_site, mps_approx = left_sweep(mps, mpo, R, stop=stop,final_site=final_site)
        R, final_site, mps_approx = right_sweep(mps, mpo, L, stop=stop,final_site=final_site)

    mps_approx.canform == "Left"
    return mps_approx

#================================================ 
#       Non Blas Contraction techniques
#================================================ 
def mps_mpo(mps, mpo, round_type = "standard",stop=Cutoff(1e-14),r=None,num_iters=5,oversample=5,l=None,final_round=False):
    new_mps = []
 
    site = np.tensordot(mpo[0], mps[0], axes=(2, 0))
    new_mps.append(
        np.reshape(site, (site.shape[0], site.shape[1] * site.shape[2]))
    )

    for i in range(1, len(mps) - 1):
        site = np.tensordot(mpo[i], mps[i], axes=(3, 1))
        site = np.moveaxis(site, 3, 1)
        site = np.reshape(
            site,
            (
                site.shape[0] * site.shape[1],
                site.shape[2],
                site.shape[3] * site.shape[4],
            ),
        )

        new_mps.append(site)
    site = np.tensordot(mpo[-1], mps[-1], (2, 1))
    site = np.moveaxis(site, 2, 1)
    site = np.reshape(site, (site.shape[0] * site.shape[1], site.shape[2]))

    new_mps.append(site)

    new_mps = MPS(new_mps)
    if stop.is_truncation():
        if round_type == "standard":
            new_mps.round(stop=stop)
        elif round_type == "dass_blas":
            dass_round_blas(new_mps,stop=stop)
        elif round_type == "dass":
            dass_round(new_mps,stop=stop)
        elif round_type == "orth_then_rand_blas":
            orth_then_rand_blas(mps,r)
        elif round_type == "rand_then_orth_blas":
            rand_then_orth_blas(mps,r,finalround=final_round)
        elif round_type == "nystrom_blas":
            nystrom_round_blas(new_mps,l,stop=stop)
        else:
            raise ValueError(f"Unknown rounding type: {round_type}")
            
    return new_mps

def density_matrix_einsum(mpo, mps, stop=Cutoff(1e-14), maxdim=None, normalize=False):
    """
    Einsum Indexing convention: 
     ______
    |      |    _______
    |      |-Z-|mps_c_j|-W-...
    |      |   |_______|
    |      |    __|l___
    |      |-F-|mpo_c_j|-G-...
    |      |   |_______|
    | L[j] |      |d  
    |      |
    |      |    __|d___
    |      |-D-| mpo_j |-E-...
    |      |   |_______|
    |      |    __|l___
    |      |-X-| mps_j |-Y-...
    |______|   |_______|
    
    Comments:
    - In cases where the same index is duplicated in a contraction ie: dxd density matrix the 
      surrogate index will be promoted one letter: dxd -> exd.
    - truncation variable produced by eig is k.
    """
    # ==========================================
    # 0. [Initialization]
    # ==========================================
    if len(mpo) != len(mps):
        raise ValueError("MPO and MPS must have the same length.")
    if maxdim is None:
        requested_maxdim = maxlinkdim(mpo) * maxlinkdim(mps)

    n = len(mpo)
    mps_out = mps.copy()
    mps_c = mps.dagger()
    mpo_c = mpo.dagger()
    physdim = mpo[0].shape[0]
    L = [None] * (n - 1)

    # ==========================================
    # 1. [Environment Tensor Construction]          
    # ==========================================
    L[0] = np.einsum('dDl,lX->dDX', mpo[0], mps[0]) 
    temp = np.einsum('dFl,lZ->ZFd', mpo_c[0], mps_c[0])
    L[0] = np.einsum('dDX,ZFd->XDFZ', L[0], temp)

    for j in range(1, n-1):
        L[j] = np.einsum('XDFZ,ZlW->XDFlW', L[j-1], mps_c[j])
        L[j] = np.einsum('XDFlW,FdGl->XDdGW', L[j], mpo_c[j])
        L[j] = np.einsum('XDdGW,DdEl->XlEGW', L[j], mpo[j])
        L[j] = np.einsum('XlEGW,XlY->YEGW', L[j], mps[j])

    # ==========================================
    # 2. [First Density Matrix]
    # ==========================================
    rho = np.einsum('XDFZ,Zl->XDFl', L[n-2], mps_c[n-1])
    rho = np.einsum('XDFl,Fdl->XDd', rho, mpo_c[n-1])
    rho = np.einsum('XDd,Xl->lDd', rho, mps[n-1])
    rho = np.einsum('lDd,Dol->od', rho, mpo[n-1])
    # ==========================================
    # 3. [Eigendecomposition]
    # ==========================================
    _, U, _ = truncated_eigendecomposition(rho, stop)
    # print(U.shape)
    mps_out[n-1] = U.T 

    # ==========================================
    # 4. [Cap Construction II]
    # ==========================================
    M_top = np.einsum("dk,Gdl->Gkl", U, mpo_c[n-1])
    M_top = np.einsum("Gkl,Wl->WGk", M_top, mps_c[n-1])
    
    for j in reversed(range(1, n-1)):
        top = np.einsum("WGk,ZlW->ZlGk", M_top, mps_c[j])  
        top = np.einsum("ZlGk,FdGl->ZFdk", top, mpo_c[j])
        bottom = np.conj(top)

        # ==========================================
        # 5. [Density Matrix j]
        # ==========================================
        rho = np.einsum("XDFZ,XDdk->kdFZ", L[j-1], bottom)
        rho = np.einsum("kdFZ,ZFej->kdje", rho, top) #consider ej
        k_old = rho.shape[0]
        physdim = rho.shape[1]
        rho = rho.reshape(rho.shape[0] * rho.shape[1], rho.shape[2] * rho.shape[3])
        _, U, _ = truncated_eigendecomposition(rho, stop)
        U = U.reshape(k_old,physdim,U.shape[1])
        U = np.einsum("kdl->ldk",U)
        mps_out[j] = U

        # ==========================================
        # 6. [Cap Construction j]
        # ==========================================
        M_top = np.einsum("ZFdk,ldk->ZFl", top, U)
        # M_bot = np.einsum("XDdk,ldk->XDl", bottom, Ut)

    mps_out[0] = np.einsum('lZ,ZFk->lFk', mps_c[0], M_top) 
    mps_out[0] = np.einsum('lFk,dFl->dk', mps_out[0], mpo_c[0])       
    return MPS(mps_out)

def zipup_einsum(mpo, mps, stop=Cutoff(1e-14), finalround=False, conditioning=False):
    # Einsum conventions:
    #   MPS[i] bond dimension X_i            = X
    #   MPS[i] bond dimension X_{i+1}        = Y
    #   MPO[i] bond dimension D_i            = D
    #   MPO[i] bond dimension D_{i+1}        = E
    #   MPO[i] physical dimension d_(top)    = d
    #   MPO[i] physical dimension d_(bottom) = l
    #   DX rank trunction p_(current)        = p
    #   DX rank trunction p_(previous)       = x
    
    if conditioning:
        mpo.canonize_left()
        mps.canonize_left()

    n = mpo.N
    mps_out = [None] * n
    physdim = mpo[0].shape[0]

    # =============================== First tensor ===============================
    C = np.tensordot(mpo[0], mps[0], axes=(2, 0))
    mps_out[0], S, Vt = truncated_svd(np.reshape(C, (C.shape[0], C.shape[1] * C.shape[2])), stop=stop)
    Z = np.diag(S) @ Vt
    Z = Z.reshape(Z.shape[0], mpo[1].shape[0], mps[1].shape[0])

    # =============================== Middle tensors =============================
    for i in range(1, n - 2):
        # Absorb MPOS site and MPS site into Z
        C = np.einsum("pDX,XlY->pDlY", Z, mps[i])
        C = np.einsum('pDlY,DdEl->pdEY', C, mpo[i])
        U, S, Vt = truncated_svd(C.reshape(C.shape[0] * C.shape[1], C.shape[2] * C.shape[3]), stop=stop)
        U = U.reshape(C.shape[0], physdim, U.shape[1])
        mps_out[i] = U
        Z = np.diag(S) @ Vt
        Z = Z.reshape(Z.shape[0], mpo[i + 1].shape[0], mps[i + 1].shape[0])

    # =============================== Last tensor ================================
    C = np.einsum("pDX,XlY->pDlY", Z, mps[n-2])
    C = np.einsum("pDlY,DdEl->pdEY", C, mpo[n-2])
    C = np.einsum("pdDX,Xl->pdDl", C, mps[n-1])
    C = np.einsum("pdDl,Dul->pdu", C, mpo[n-1])
    U, S, Vt = truncated_svd(np.reshape(C, (C.shape[0] * C.shape[1], C.shape[1])), stop=stop)
    mps_out[n-2] = np.reshape(U, (C.shape[0], C.shape[2], U.shape[1]))
    Z = np.diag(S) @ Vt
    mps_out[-1] = Z

    mps_out = MPS(mps_out)
    if finalround:
        mps_out.round(stop=stop)

    return MPS(mps_out)

def fit_einsum(mpo, mps, max_sweeps=10, stop=Cutoff(1e-14), random_tensor=np.random.randn):
    """
    Einsum Indexing convention L->R:
     ______                 ______
    |      |    _______    |      |   
    |      |-Z-|mps_c_j|-W-|      |
    |      |   |_______|   |      |
    |      |    __|d___    |      |
    | L[j] |-D-| mpo_j |-E-|R[N-j]|
    |      |   |_______|   |      |
    |      |    __|l___    |      |
    |      |-X-| mps_j |-Y-|      |
    |______|   |_______|   |______|
    """
    
    def right_sweep(mps, mpo, L, final_site=None,stop=Cutoff(1e-14)):
        mps_out = [None] * mps.N
        R = [None] * (mps.N - 2)

        if final_site is None:
            base = np.einsum("XjY,Yl->Xjl", mps[-2], mps[-1])
            site = np.einsum("Xjl,Edl->XjEd", base, mpo[-1])
            site = np.einsum("XjEk,DdEj->XDkd", site, mpo[-2])
            site = np.einsum("pDX,XDkd->pdk", L[-1], site)
        else:
            site = final_site

        U, S, Vt = truncated_svd(site.reshape(site.shape[0] * site.shape[1], site.shape[2]), stop=stop)
        mps_out[-1] = (Vt).reshape(U.shape[1], mps[-1].shape[1])

        R[-1] = np.einsum("Ddl,Xl->XDd", mpo[-1], mps[-1])
        R[-1] = np.einsum("XDd,pd->XDp", R[-1], mps_out[-1])

        # Middle sites.
        for k in range(mps.N - 2, 1, -1):
            site = np.einsum("XDp,YlX->YlDp", R[k - 1], mps[k])
            site = np.einsum("YlDp,EdDl->YEdp", site, mpo[k])
            site = np.einsum("YEdp,ZlY->ZlEdp", site, mps[k - 1])
            site = np.einsum("ZlEdp,FkEl->ZFkdp", site, mpo[k - 1])  # five tensor?
            # print(site.shape,L[k-1].shape)
            # print(k)
            # print(site.shape,L[k-1].shape)
            site = np.einsum("ZFkdp,qFZ->qkdp", site, L[k - 2])

            U, S, Vt = truncated_svd(site.reshape(site.shape[0] * site.shape[1], site.shape[2] * site.shape[3]), stop=stop)
            mps_out[k] = (Vt).reshape(U.shape[1], mps[-1].shape[1], site.shape[3])

            
            R[k - 2] = np.einsum("XDp,qdp->XDdq", R[k - 1], mps_out[k])
            R[k - 2] = np.einsum("XDdq,EdDl->XlEq", R[k - 2], mpo[k])
            R[k - 2] = np.einsum("XlEq,YlX->YEq", R[k - 2], mps[k])

        
        site = np.einsum("lX,XkY->lkY",mps[0],mps[1])
        site = np.einsum("lkY,dDl->dDkY",site,mpo[0])
        site = np.einsum("kDlY,DdEl->kdEY",site,mpo[1])
        final_site = np.einsum("kdEY,YEq->kdq",site,R[0])

        U, S, Vt = truncated_svd(final_site.reshape(final_site.shape[0], final_site.shape[1] * final_site.shape[2]), stop=stop)
        mps_out[1] = Vt.reshape(Vt.shape[0], mps[0].shape[0], R[0].shape[2])
        mps_out[0] = U @ np.diag(S)

        return R, final_site, MPS(mps_out)
    
    def left_sweep(mps, mpo, R, final_site=None,stop=Cutoff(1e-14)):
            mps_out = [None] * mps.N
            L = [None] * (mps.N - 2)

            # First Local Site from the left (H_0ψ_0)(H_1ψ_1)R_0  of size (d,d,X)
            if final_site is None:
                site = np.einsum("lX,XjY->ljY", mps[0], mps[1])
                site = np.einsum("ljY,dDl->dDjY", site, mpo[0])
                site = np.einsum("dDjY,DkEj->dkEY", site, mpo[1])
                site = np.einsum("dkEY,YEW->dkW", site, R[0])
            else:
                site = final_site
            U, S, Vt = truncated_svd(site.reshape(site.shape[0], site.shape[1] * site.shape[2]), stop=stop)
            mps_out[0] = U

            L[0] = np.einsum("dp,dDl->pDl", U, mpo[0])
            L[0] = np.einsum("pDl,lX->pDX", L[0], mps[0])

            # Sweep through the remaining sites
            for k in range(1, len(mps) - 2):
                site = np.einsum("pDX,XlY->pDlY", L[k - 1], mps[k])
                site = np.einsum("pDlY,DdEl->pdEY", site, mpo[k])
                site = np.einsum("pdEY,YlZ->pdElZ", site, mps[k + 1])
                site = np.einsum("pdElZ,EkFl->pdkFZ", site, mpo[k + 1])
                site = np.einsum("pdkFZ,ZFW->pdkW", site, R[k])

                U, _, _ = truncated_svd(site.reshape(site.shape[0] * site.shape[1], site.shape[2] * site.shape[3]), stop=stop)
                mps_out[k] = U.reshape(site.shape[0], mpo[k].shape[1], U.shape[1])

                L[k] = np.einsum("pDX,pdq->qdDX", L[k - 1], mps_out[k])
                L[k] = np.einsum("qdDX,DdEl->qElX", L[k], mpo[k])
                L[k] = np.einsum("qElX,XlY->qEY", L[k], mps[k])

            # Final two sites
            site = np.einsum("XlY,Yk->Xlk",mps[-2],mps[-1])
            site = np.einsum("Xlk,Edk->XlEd",site,mpo[-1])
            site = np.einsum("XlEd,DkEl->XDdk",site,mpo[-2])
            final_site = np.einsum("qDX,XDdk->qdk",L[k],site)

            U, S, Vt = truncated_svd(final_site.reshape(final_site.shape[0] * final_site.shape[1], final_site.shape[2]), stop=stop)
            mps_out[-2] = U.reshape(final_site.shape[0], mps[-2].shape[1], final_site.shape[2])
            mps_out[-1] = (np.diag(S) @ Vt).reshape(final_site.shape[2], mps[-1].shape[1])

            return L, final_site, MPS(mps_out)

    def compute_left_envs(mps, mpo, guess):
            L = [None] * (mps.N - 2)
            # -------- Left environments --------
            L[0] = np.einsum("dDl,dZ->ZDl", mpo[0], guess[0])
            L[0] = np.einsum("ZDl,lX->ZDX", L[0], mps[0])

            for i in range(1, mps.N - 2):
                L[i] = np.einsum("ZDX,XlY->ZDlY", L[i - 1], mps[i])
                L[i] = np.einsum("ZDlY,DdEl->ZdEY", L[i], mpo[i])
                L[i] = np.einsum("ZdEY,ZdW->WEY", L[i], guess[i])
            return L

    def compute_right_envs(mps, mpo, guess):
            # Only needed if starting from left
            R = [None] * (mps.N - 2)
            # -------- Right environments --------
            R[-1] = np.einsum("Ddl,Zd->ZDl", mpo[-1], guess[-1])
            R[-1] = np.einsum("ZDl,Xl->XDZ", R[-1], mps[-1])

            for i in range(mps.N - 2, 1, -1):
                R[i - 2] = np.einsum("XDZ,YlX->YlDZ", R[i - 1], mps[i])
                R[i - 2] = np.einsum("YlDZ,EdDl->YEdZ", R[i - 2], mpo[i])
                R[i - 2] = np.einsum("YEdZ,WdZ->YEW", R[i - 2], guess[i])
            return R
        
    # Form a random MPS |ψB> of bond dimension m
    guess = MPS.random_mps(n=mps.N, m=mps[0].shape[1], d=mps[0].shape[0], random_tensor=random_tensor)

    # orthogonalize it to have any arbitrary orthogonality center.
    guess.canonize_left()

    R = compute_right_envs(mps, mpo, guess)

    sweep_count = 0
    final_site = None
    for sweep_count in range(max_sweeps):
    # for sweep_count in tqdm(range(max_sweeps), desc="Sweeping Progress"):
        if sweep_count % 2 == 0:
            L, final_site, mps_approx = left_sweep(mps, mpo, R, stop=stop,final_site=final_site)
        else:
            R, final_site, mps_approx = right_sweep(mps, mpo, L,stop-stop, final_site=final_site)

    return mps_approx

#================================================
#            Rand Apply variants 
#================================================

#Random contraction (tensordots only)
def rand_apply_tensordot(
    H,
    psi,
    stop=Cutoff(1e-6),
    sketchdim=1,
    sketchincrement=1,
    finalround=None,
    accuracychecks=False,
):
    # ===========================================================================
    # 1. Initialization
    # ===========================================================================
    n = len(H)

    if n != psi.N:
        raise ValueError("lengths of MPO and MPS do not match")

    if n == 1:
        raise NotImplementedError("MPO-MPS product for n=1 has not been implemented yet")
        # return [H[0] * psi[0]]

    psi_out = psi.N * [None]

    maxdim = stop.maxdim
    outputdim = stop.outputdim
    mindim = stop.mindim
    cutoff = stop.cutoff

    if outputdim is None: # Adaptive determination of sketch dimension
        if maxdim is None:
            maxdim = maxlinkdim(H) * maxlinkdim(psi)
        mindim = max(mindim, 1)
    else:
        maxdim = outputdim
        mindim = outputdim
        sketchdim = outputdim
    
    envs = []
    cap = None
    cap_dim = 1

    visible_dim = H[0].shape[0]

    for j in reversed(range(1, n)):
        # print("J= ",j)
        # ===========================================================================
        # 2. Dimension Handling
        # ===========================================================================
        if j == n - 1:
            prod_bond_dims = (
                H[j].shape[0] * psi[j].shape[0]
            )  # product of left bond
        else:
            prod_bond_dims = max(
                (H[j].shape[0] * psi[j].shape[0]),
                (H[j].shape[2] * psi[j].shape[2]),
            )  # either left or right

        current_maxdim = min(prod_bond_dims, maxdim, visible_dim * cap_dim)
        current_mindim = min(mindim, current_maxdim)
        current_sketchdim = max(min(sketchdim, current_maxdim), current_mindim)

        sketches_complete = 0

        if j == n - 1:
            sketch = np.zeros((visible_dim, current_sketchdim),dtype='complex')
        else:
            sketch = np.zeros((visible_dim, cap_dim, current_sketchdim),dtype='complex')
        while True:
            # ===========================================================================
            # 3. env formation
            # ===========================================================================
            for idx in range(len(envs), current_sketchdim):
                env = []
                for k in range(0, j):
                    tensor_k = H[k]
                    if k == 0:
                        x = np.random.randn(tensor_k.shape[0])
                        temp = np.tensordot(tensor_k, x, axes=(0, 0))
                        temp = np.tensordot(temp, psi[0], axes=(1, 0))
                        env.append(temp)
                    else:
                        x = np.random.randn(tensor_k.shape[1])
                        temp = np.tensordot(tensor_k, x, axes=(1, 0))
                        temp = np.tensordot(temp, env[k - 1], axes=(0, 0))
                        temp = np.tensordot(temp, psi[k], axes=([1, 2], [1, 0]))
                        env.append(temp)
                envs.append(env)

            # ===========================================================================
            # 4. sketch  formation
            # ===========================================================================
            for idx in range(sketches_complete, current_sketchdim):
                if j == n - 1:
                    # C[n-2] (Hψ)[n-1]
                    temp = np.tensordot(envs[idx][j - 1], psi[j], axes=(1, 0))
                    temp = np.tensordot(H[j], temp, axes=([0, 2], [0, 1]))
                    sketch[:, idx] = temp
                else:
                    # C[j-1] (Hψ)[j] S[j+1]
                    temp = np.tensordot(envs[idx][j - 1], psi[j], axes=(1, 0))
                    temp = np.tensordot(H[j], temp, axes=([0, 3], [0, 1]))
                    temp = np.tensordot(temp, cap, axes=([1, 2], [1, 2]))
                    sketch[:, :, idx] = temp

            sketches_complete = current_sketchdim

            # ===========================================================================
            # 5. QR decomposition
            # ===========================================================================
            if j == n - 1:
                Q, R = np.linalg.qr(sketch)
            else:
                temp = sketch.reshape(cap_dim * visible_dim, current_sketchdim)
                Q, R = np.linalg.qr(temp)
                Q = Q.reshape((visible_dim, cap_dim, current_sketchdim))
            
            if not (outputdim is None):
                done = True
            else:
                # ===========================================================================
                # 7. error calc
                # ===========================================================================
                done = False
                if current_sketchdim == current_maxdim:
                    done = True
                else:
                    norm_est = np.linalg.norm(sketch) / np.sqrt(current_sketchdim)
                    G = np.linalg.inv(R.T)

                    err_est = np.sqrt(sum(np.linalg.norm(G, axis=0) ** (-2)) / current_sketchdim)
                    done = err_est <= cutoff * norm_est and current_sketchdim >= current_mindim
            
            # print("error estimation", err_est)
            # print("norm estimation",norm_est)
            # print("Validity of condition 1:",(err_est <= cutoff * norm_est and current_sketchdim >= current_mindim))
            # print("Validity of condition 2:",(current_sketchdim == current_maxdim))
            
            # ===========================================================================
            # 9. cap construction
            # ===========================================================================
            if done:
                if j == n - 1:
                    psi_out[j] = Q.transpose(1, 0)
                else:
                    psi_out[j] = Q.transpose(2, 0, 1)
                cap_dim = current_sketchdim

                if j == n - 1:
                    temp = np.tensordot(np.conj(Q), H[j], axes=(0, 1))
                    cap = np.tensordot(temp, psi[j], axes=(2, 1))
                else:
                    temp = np.tensordot(np.conj(Q), cap, axes=(1, 0))
                    temp = np.tensordot(temp, H[j], axes=([0, 2], [1, 2]))
                    cap = np.tensordot(temp, psi[j], axes=([3, 1], [1, 2]))
                if accuracychecks:
                    check_randomized_apply(H, psi, cap, psi_out, j+1, verbose=False, cutoff=((100*cutoff) if (maxdim is None) else np.Inf))
                break

            # ===========================================================================
            # 10. sketch correction
            # ===========================================================================
            current_sketchdim = min(current_maxdim, current_sketchdim + sketchincrement)
            old_sketch = sketch

            if j == n - 1:
                sketch = np.zeros(
                    (
                        old_sketch.shape[0],
                        current_sketchdim,
                    ),dtype='complex'
                )
                sketch[:, : old_sketch.shape[1]] = old_sketch
            else:
               
                sketch = np.zeros(
                    (
                        old_sketch.shape[0],
                        old_sketch.shape[1],
                        current_sketchdim,
                    ),dtype='complex'
                )
                sketch[:, :, : old_sketch.shape[2]] = old_sketch

    temp = np.tensordot(cap, psi[0], axes=(2, 1))
    psi_out[0] = np.tensordot(H[0], temp, axes=([1,2], [1,2]))
    # mps_round_left(psi_out,cutoff,maxdim)
    psi_out= MPS(psi_out, canform="Left")
    if not (finalround is None):
        psi_out.round(stop = finalround)
    return psi_out
    
#Random contraction (einsums only)
def rand_apply_einsum(
    H,
    psi,
    stop=Cutoff(1e-6),
    sketchdim=1,
    sketchincrement=1,
    finalround=None,
    accuracychecks=False,
):
    # ===========================================================================
    # 1. Initialization
    # ===========================================================================
    n = len(H)

    if n != psi.N:
        raise ValueError("lengths of MPO and MPS do not match")

    if n == 1:
        raise NotImplementedError("MPO-MPS product for n=1 has not been implemented yet")
        # return [H[0] * psi[0]]

    psi_out = psi.N * [None]

    maxdim = stop.maxdim
    outputdim = stop.outputdim
    mindim = stop.mindim
    cutoff = stop.cutoff

    if outputdim is None: # Adaptive determination of sketch dimension
        if maxdim is None:
            maxdim = maxlinkdim(H) * maxlinkdim(psi)
        mindim = max(mindim, 1)
    else:
        maxdim = outputdim
        mindim = outputdim
        sketchdim = outputdim
    
    envs = []
    cap = None
    cap_dim = 1

    visible_dim = H[0].shape[0]

    for j in reversed(range(1, n)):
        # print("J= ",j)
        # ===========================================================================
        # 2. Dimension Handling
        # ===========================================================================
        if j == n - 1:
            prod_bond_dims = (
                H[j].shape[0] * psi[j].shape[0]
            )  # product of left bond
        else:
            prod_bond_dims = max(
                (H[j].shape[0] * psi[j].shape[0]),
                (H[j].shape[2] * psi[j].shape[2]),
            )  # either left or right

        current_maxdim = min(prod_bond_dims, maxdim, visible_dim * cap_dim)
        current_mindim = min(mindim, current_maxdim)
        current_sketchdim = max(min(sketchdim, current_maxdim), current_mindim)

        sketches_complete = 0

        if j == n - 1:
            sketch = np.zeros((visible_dim, current_sketchdim),dtype='complex')
        else:
            sketch = np.zeros((visible_dim, cap_dim, current_sketchdim),dtype='complex')

        while True:
            # ===========================================================================
            # 3. env formation
            # ===========================================================================                
            for idx in range(len(envs), current_sketchdim):
                env = []
                x = np.random.randn(visible_dim) # consider precompute
                temp= np.einsum('ijk,i->jk',H[0],x)
                env.append(temp @ psi[0])
                for k in range(1, j):
                    x = np.random.randn(visible_dim) # consider precompute
                    temp = np.einsum('ijkl,j->ikl', H[k], x)
                    temp = np.einsum('ijk,il->jkl', temp, env[k - 1])
                    temp = np.einsum('ijk,kjl->il', temp, psi[k])
                    env.append(temp)
                envs.append(env)

            # ===========================================================================
            # 4. sketch  formation
            # ===========================================================================
            for idx in range(sketches_complete, current_sketchdim):
                if j == n - 1:
                    # C[n-2] (Hψ)[n-1]
                    temp = envs[idx][j - 1] @ psi[j]
                    temp = np.einsum('ijk,ik->j',H[j],temp)
                    sketch[:, idx] = temp
                else:
                    # C[j-1] (Hψ)[j] S[j+1]
                    temp=np.einsum('ij,jkl->ikl',envs[idx][j - 1],psi[j])
                    temp=np.einsum('ijlk,ikn->jln',H[j], temp)
                    temp = np.einsum('ijk,ljk->il',temp,cap)
                    sketch[:, :, idx] = temp


            sketches_complete = current_sketchdim

            # ===========================================================================
            # 5. QR decomposition
            # ===========================================================================
            if j == n - 1:
                Q, R = np.linalg.qr(sketch)
            else:
                temp = sketch.reshape(cap_dim * visible_dim, current_sketchdim)
                Q, R = np.linalg.qr(temp)
                Q = Q.reshape((visible_dim, cap_dim, current_sketchdim))
            
            if not (outputdim is None):
                done = True
            else:
                # ===========================================================================
                # 7. error calc
                # ===========================================================================
                done = False
                if current_sketchdim == current_maxdim:
                    done = True
                else:
                    norm_est = np.linalg.norm(sketch) / np.sqrt(current_sketchdim)
                    G = np.linalg.inv(R.T)

                    err_est = np.sqrt(sum(np.linalg.norm(G, axis=0) ** (-2)) / current_sketchdim)
                    done = err_est <= cutoff * norm_est and current_sketchdim >= current_mindim
            
            # print("error estimation", err_est)
            # print("norm estimation",norm_est)
            # print("Validity of condition 1:",(err_est <= cutoff * norm_est and current_sketchdim >= current_mindim))
            # print("Validity of condition 2:",(current_sketchdim == current_maxdim))
            
            # ===========================================================================
            # 9. cap construction
            # ===========================================================================
            if done:
                if j == n - 1:
                    psi_out[j] = Q.transpose(1, 0)
                else:
                    psi_out[j] = Q.transpose(2, 0, 1)
                cap_dim = current_sketchdim

                if j == n - 1:
                    temp=np.einsum("ij,kil->jkl",np.conj(Q), H[j])
                    cap =np.einsum("ijk,lk->ijl",temp, psi[j])
                else:
                    #temp = np.einsum('ij,kjl->jkl',np.conj(Q),H[j])

                    temp = np.tensordot(np.conj(Q), cap, axes=(1, 0))
                    temp = np.einsum("ijkl,mikn->jlmn",temp, H[j])
                    cap = np.einsum("ijkl,mlj->ikm",temp, psi[j])
                if accuracychecks:
                    check_randomized_apply(H, psi, cap, psi_out, j+1, verbose=False, cutoff=((100*cutoff) if (maxdim is None) else np.Inf))
                break

            # ===========================================================================
            # 10. sketch correction
            # ===========================================================================
            current_sketchdim = min(current_maxdim, current_sketchdim + sketchincrement)
            old_sketch = sketch

            if j == n - 1:
                sketch = np.zeros(
                    (
                        old_sketch.shape[0],
                        current_sketchdim,
                    ),dtype='complex'
                )
                sketch[:, : old_sketch.shape[1]] = old_sketch
            else:
               
                sketch = np.zeros(
                    (
                        old_sketch.shape[0],
                        old_sketch.shape[1],
                        current_sketchdim,
                    ),dtype='complex'
                )
                sketch[:, :, : old_sketch.shape[2]] = old_sketch

    temp = np.einsum('ijk,lk',cap, psi[0])
    #print(H[0].shape,temp.shape)
    psi_out[0] = np.einsum('ijk,ljk->il',H[0],temp)
    #psi_out[0] = np.tensordot(H[0], temp, axes=([1,2], [1,2]))
    #print(psi_out[0].shape)
    # mps_round_left(psi_out,cutoff,maxdim)
    psi_out= MPS(psi_out, canform="Left")
    if not (finalround is None):
        psi_out.round(stop = finalround)
    return psi_out

#Random contraction (einsums + tensordot)
def rand_apply_hybrid(
    H,
    psi,
    stop=Cutoff(1e-6),
    sketchdim=1,
    sketchincrement=1,
    finalround=None,
    accuracychecks=False,
):
    # ===========================================================================
    # 1. Initialization
    # ===========================================================================
    n = len(H)

    if n != psi.N:
        raise ValueError("lengths of MPO and MPS do not match")

    if n == 1:
        raise NotImplementedError("MPO-MPS product for n=1 has not been implemented yet")
        # return [H[0] * psi[0]]

    psi_out = psi.N * [None]

    maxdim = stop.maxdim
    outputdim = stop.outputdim
    mindim = stop.mindim
    cutoff = stop.cutoff

    if outputdim is None: # Adaptive determination of sketch dimension
        if maxdim is None:
            maxdim = maxlinkdim(H) * maxlinkdim(psi)
        mindim = max(mindim, 1)
    else:
        maxdim = outputdim
        mindim = outputdim
        sketchdim = outputdim
    
    envs = []
    cap = None
    cap_dim = 1

    visible_dim = H[0].shape[0]

    for j in reversed(range(1, n)):
        # print("J= ",j)
        # ===========================================================================
        # 2. Dimension Handling
        # ===========================================================================
        if j == n - 1:
            prod_bond_dims = (
                H[j].shape[0] * psi[j].shape[0]
            )  # product of left bond
        else:
            prod_bond_dims = max(
                (H[j].shape[0] * psi[j].shape[0]),
                (H[j].shape[2] * psi[j].shape[2]),
            )  # either left or right

        current_maxdim = min(prod_bond_dims, maxdim, visible_dim * cap_dim)
        current_mindim = min(mindim, current_maxdim)
        current_sketchdim = max(min(sketchdim, current_maxdim), current_mindim)

        sketches_complete = 0

        if j == n - 1:
            sketch = np.zeros((visible_dim, current_sketchdim),dtype='complex')
        else:
            sketch = np.zeros((visible_dim, cap_dim, current_sketchdim),dtype='complex')

        while True:
            # ===========================================================================
            # 3. env formation
            # ===========================================================================             
            for idx in range(len(envs), current_sketchdim):
                env = []
                random_vectors = np.random.randn(j,visible_dim)
                temp= np.einsum('ijk,i->jk',H[0],random_vectors[0])
                env.append(temp @ psi[0])
                for k in range(1, j):
                    temp = np.einsum('ijkl,j->ikl', H[k], random_vectors[k])
                    temp = np.einsum('ijk,il->jkl', temp, env[k - 1])
                    temp = np.einsum('ijk,kjl->il', temp, psi[k])
                    env.append(temp)
                envs.append(env)

            # ===========================================================================
            # 4. sketch  formation
            # ===========================================================================
            for idx in range(sketches_complete, current_sketchdim):
                if j == n - 1:
                    # C[n-2] (Hψ)[n-1]
                    temp = envs[idx][j - 1] @ psi[j]
                    temp = np.einsum('ijk,ik->j',H[j],temp)
                    sketch[:, idx] = temp
                else:
                    # C[j-1] (Hψ)[j] S[j+1]
                    temp=np.einsum('ij,jkl->ikl',envs[idx][j - 1],psi[j])
                    temp=np.einsum('ijlk,ikn->jln',H[j], temp)
                    temp = np.einsum('ijk,ljk->il',temp,cap)
                    sketch[:, :, idx] = temp


            sketches_complete = current_sketchdim

            # ===========================================================================
            # 5. QR decomposition
            # ===========================================================================
            if j == n - 1:
                Q, R = np.linalg.qr(sketch)
            else:
                temp = sketch.reshape(cap_dim * visible_dim, current_sketchdim)
                Q, R = np.linalg.qr(temp)
                Q = Q.reshape((visible_dim, cap_dim, current_sketchdim))
            
            if not (outputdim is None):
                done = True
            else:
                # ===========================================================================
                # 7. error calc
                # ===========================================================================
                done = False
                if current_sketchdim == current_maxdim:
                    done = True
                else:
                    norm_est = np.linalg.norm(sketch) / np.sqrt(current_sketchdim)
                    G = np.linalg.inv(R.T)

                    err_est = np.sqrt(sum(np.linalg.norm(G, axis=0) ** (-2)) / current_sketchdim)
                    done = err_est <= cutoff * norm_est and current_sketchdim >= current_mindim
            
            # print("error estimation", err_est)
            # print("norm estimation",norm_est)
            # print("Validity of condition 1:",(err_est <= cutoff * norm_est and current_sketchdim >= current_mindim))
            # print("Validity of condition 2:",(current_sketchdim == current_maxdim))
            
            # ===========================================================================
            # 9. cap construction
            # ===========================================================================
            if done:
                if j == n - 1:
                    psi_out[j] = Q.transpose(1, 0)
                else:
                    psi_out[j] = Q.transpose(2, 0, 1)
                cap_dim = current_sketchdim

                if j == n - 1:
                    temp=np.einsum("ij,kil->jkl",np.conj(Q), H[j])
                    cap =np.einsum("ijk,lk->ijl",temp, psi[j])
                else:
                    #temp = np.einsum('ij,kjl->jkl',np.conj(Q),H[j])

                    temp = np.tensordot(np.conj(Q), cap, axes=(1, 0))
                    temp = np.tensordot(temp, H[j], axes=([0, 2], [1, 2]))
                    cap = np.tensordot(temp, psi[j], axes=([3, 1], [1, 2]))
                if accuracychecks:
                    check_randomized_apply(H, psi, cap, psi_out, j+1, verbose=False, cutoff=((100*cutoff) if (maxdim is None) else np.Inf))
                break

            # ===========================================================================
            # 10. sketch correction
            # ===========================================================================
            current_sketchdim = min(current_maxdim, current_sketchdim + sketchincrement)
            old_sketch = sketch

            if j == n - 1:
                sketch = np.zeros(
                    (
                        old_sketch.shape[0],
                        current_sketchdim,
                    ),dtype='complex'
                )
                sketch[:, : old_sketch.shape[1]] = old_sketch
            else:
               
                sketch = np.zeros(
                    (
                        old_sketch.shape[0],
                        old_sketch.shape[1],
                        current_sketchdim,
                    ),dtype='complex'
                )
                sketch[:, :, : old_sketch.shape[2]] = old_sketch

    temp = np.einsum('ijk,lk',cap, psi[0])
    #print(H[0].shape,temp.shape)
    psi_out[0] = np.einsum('ijk,ljk->il',H[0],temp)
    #psi_out[0] = np.tensordot(H[0], temp, axes=([1,2], [1,2]))
    #print(psi_out[0].shape)
    # mps_round_left(psi_out,cutoff,maxdim)
    psi_out= MPS(psi_out, canform="Left")
    if not (finalround is None):
        psi_out.round(stop = finalround)
    return psi_out

#Random contraction (torch) (torch.tensordot)
def rand_apply_torch_tensordot(
    H,
    psi,
    stop=Cutoff(1e-6),
    sketchdim=1,
    sketchincrement=1,
    finalround=None,
    accuracychecks=False,
):
    # ===========================================================================
    # 1. Initialization
    # ===========================================================================
    n = len(H)

    if n != psi.N:
        raise ValueError("lengths of MPO and MPS do not match")

    if n == 1:
        raise NotImplementedError("MPO-MPS product for n=1 has not been implemented yet")
        # return [H[0] * psi[0]]

    psi_out = psi.N * [None]

    maxdim = stop.maxdim
    outputdim = stop.outputdim
    mindim = stop.mindim
    cutoff = stop.cutoff

    if outputdim is None: # Adaptive determination of sketch dimension
        if maxdim is None:
            maxdim = maxlinkdim(H) * maxlinkdim(psi)
        mindim = max(mindim, 1)
    else:
        maxdim = outputdim
        mindim = outputdim
        sketchdim = outputdim
    
    envs = []
    cap = None
    cap_dim = 1

    visible_dim = H[0].shape[0]

    for j in reversed(range(1, n)):
        # print("J= ",j)
        # ===========================================================================
        # 2. Dimension Handling
        # ===========================================================================
        if j == n - 1:
            prod_bond_dims = (
                H[j].shape[0] * psi[j].shape[0]
            )  # product of left bond
        else:
            prod_bond_dims = max(
                (H[j].shape[0] * psi[j].shape[0]),
                (H[j].shape[2] * psi[j].shape[2]),
            )  # either left or right

        current_maxdim = min(prod_bond_dims, maxdim, visible_dim * cap_dim)
        current_mindim = min(mindim, current_maxdim)
        current_sketchdim = max(min(sketchdim, current_maxdim), current_mindim)

        sketches_complete = 0

        if j == n - 1:
            sketch = torch.zeros((visible_dim, current_sketchdim),dtype=torch.complex128)
        else:
            sketch = torch.zeros((visible_dim, cap_dim, current_sketchdim),dtype=torch.complex128)

        while True:
            # ===========================================================================
            # 3. env formation
            # ===========================================================================  
            for idx in range(len(envs), current_sketchdim):
                env = []
                x=torch.randn(visible_dim,dtype=H[0].dtype)
                temp = torch.tensordot(H[0], x, dims=([0], [0]))
                env.append(temp @ psi[0])
                for k in range(1, j):
                    x=torch.randn(visible_dim,dtype=H[0].dtype)
                    temp = torch.tensordot(H[k], x, dims=([1], [0]))
                    temp = torch.tensordot(temp, env[k - 1], dims=([0], [0]))
                    temp = torch.tensordot(temp, psi[k], dims=([1, 2], [1, 0]))
                    env.append(temp)
                envs.append(env)              
            # ===========================================================================
            # 4. sketch  formation
            # ===========================================================================
           
            for idx in range(sketches_complete, current_sketchdim):
                if j == n - 1:
                    # C[n-2] (Hψ)[n-1]
                    temp = torch.tensordot(envs[idx][j - 1], psi[j], dims=([1], [0]))
                    temp = torch.tensordot(H[j], temp, dims=([0, 2], [0, 1]))
                    sketch[:, idx] = temp
                else:
                    # C[j-1] (Hψ)[j] S[j+1]
                    temp = torch.tensordot(envs[idx][j - 1], psi[j], dims=([1], [0]))
                    temp = torch.tensordot(H[j], temp, dims=([0, 3], [0, 1]))
                    temp = torch.tensordot(temp, cap, dims=([1, 2], [1, 2]))
                    sketch[:, :, idx] = temp
            sketches_complete = current_sketchdim

            # ===========================================================================
            # 5. QR decomposition
            # ===========================================================================
            if j == n - 1:
                Q, R = torch.linalg.qr(sketch)
            else:
                temp = sketch.reshape(cap_dim * visible_dim, current_sketchdim)
                Q, R = torch.linalg.qr(temp)
                Q = Q.reshape((visible_dim, cap_dim, current_sketchdim))
            
            if not (outputdim is None):
                done = True
            else:
                # ===========================================================================
                # 7. error calc
                # ===========================================================================
                done = False
                if current_sketchdim == current_maxdim:
                    done = True
                else:
                    norm_est = np.linalg.norm(sketch) / np.sqrt(current_sketchdim)
                    G = torch.linalg.inv(R.T)

                    err_est = np.sqrt(sum(np.linalg.norm(G, axis=0) ** (-2)) / current_sketchdim)
                    done = err_est <= cutoff * norm_est and current_sketchdim >= current_mindim
            
            # print("error estimation", err_est)
            # print("norm estimation",norm_est)
            # print("Validity of condition 1:",(err_est <= cutoff * norm_est and current_sketchdim >= current_mindim))
            # print("Validity of condition 2:",(current_sketchdim == current_maxdim))
            
            # ===========================================================================
            # 9. cap construction
            # ===========================================================================
            if done:
                if j == n - 1:
                    psi_out[j] = Q.permute(1, 0)
                else:
                    psi_out[j] = Q.permute(2, 0, 1)
                cap_dim = current_sketchdim

                if j == n - 1:
                    temp = torch.tensordot(Q.conj(), H[j], dims=([0], [1]))
                    cap = torch.tensordot(temp, psi[j], dims=([2], [1]))
                else:
                    temp = torch.tensordot(Q.conj(), cap, dims=([1], [0]))
                    temp = torch.tensordot(temp, H[j], dims=([0, 2], [1, 2]))
                    cap = torch.tensordot(temp, psi[j], dims=([3, 1], [1, 2]))
                if accuracychecks:
                    check_randomized_apply(H, psi, cap, psi_out, j+1, verbose=False, cutoff=((100*cutoff) if (maxdim is None) else np.Inf))
                break
            # ===========================================================================
            # 10. sketch correction
            # ===========================================================================
            current_sketchdim = min(current_maxdim, current_sketchdim + sketchincrement)
            old_sketch = sketch

            if j == n - 1:
                sketch = torch.zeros(
                    (
                        old_sketch.shape[0],
                        current_sketchdim,
                    ), dtype=torch.complex128  # Use torch.complex64 if less precision is required
                )
                sketch[:, : old_sketch.shape[1]] = torch.from_numpy(old_sketch)
            else:
                sketch = torch.zeros(
                    (
                        old_sketch.shape[0],
                        old_sketch.shape[1],
                        current_sketchdim,
                    ), dtype=torch.complex128  # Use torch.complex64 if less precision is required
                )
                sketch[:, :, : old_sketch.shape[2]] = old_sketch

    temp = torch.tensordot(cap, psi[0], dims=([2], [1]))
    psi_out[0] = torch.tensordot(H[0], temp, dims=([1, 2], [1, 2]))
    #psi_out[0] = np.tensordot(H[0], temp, axes=([1,2], [1,2]))
    #print(psi_out[0].shape)
    # mps_round_left(psi_out,cutoff,maxdim)
    psi_out= MPS(psi_out, canform="Left")
    if not (finalround is None):
        psi_out.round(stop = finalround)
    return psi_out

#Random contraction (torch) (torch.einsum)
def rand_apply_torch_einsum(
    H,
    psi,
    stop=Cutoff(1e-6),
    sketchdim=1,
    sketchincrement=1,
    finalround=None,
    accuracychecks=False,
):
    # ===========================================================================
    # 1. Initialization
    # ===========================================================================
    n = len(H)

    if n != psi.N:
        raise ValueError("lengths of MPO and MPS do not match")

    if n == 1:
        raise NotImplementedError("MPO-MPS product for n=1 has not been implemented yet")
        # return [H[0] * psi[0]]

    psi_out = psi.N * [None]

    maxdim = stop.maxdim
    outputdim = stop.outputdim
    mindim = stop.mindim
    cutoff = stop.cutoff

    if outputdim is None: # Adaptive determination of sketch dimension
        if maxdim is None:
            maxdim = maxlinkdim(H) * maxlinkdim(psi)
        mindim = max(mindim, 1)
    else:
        maxdim = outputdim
        mindim = outputdim
        sketchdim = outputdim
    
    envs = []
    cap = None
    cap_dim = 1

    visible_dim = H[0].shape[0]

    for j in reversed(range(1, n)):
        # print("J= ",j)
        # ===========================================================================
        # 2. Dimension Handling
        # ===========================================================================
        if j == n - 1:
            prod_bond_dims = (
                H[j].shape[0] * psi[j].shape[0]
            )  # product of left bond
        else:
            prod_bond_dims = max(
                (H[j].shape[0] * psi[j].shape[0]),
                (H[j].shape[2] * psi[j].shape[2]),
            )  # either left or right

        current_maxdim = min(prod_bond_dims, maxdim, visible_dim * cap_dim)
        current_mindim = min(mindim, current_maxdim)
        current_sketchdim = max(min(sketchdim, current_maxdim), current_mindim)

        sketches_complete = 0

        if j == n - 1:
            sketch = torch.zeros((visible_dim, current_sketchdim),dtype=torch.complex128)
        else:
            sketch = torch.zeros((visible_dim, cap_dim, current_sketchdim),dtype=torch.complex128)

        while True:
            # ===========================================================================
            # 3. env formation
            # ===========================================================================                
            for idx in range(len(envs), current_sketchdim):
                env = []
                x=torch.randn(visible_dim,dtype=H[0].dtype)
                #x = np.random.randn(visible_dim) # consider precompute
                temp= torch.einsum('ijk,i->jk',H[0],x)
                env.append(temp @ psi[0])
                for k in range(1, j):
                    # if H[k].is_complex():
                    #     x = torch.randn(visible_dim, dtype=torch.complex128)
                    # else:
                    #     x = torch.randn(visible_dim, dtype=H[k].dtype)
                    #x = np.random.randn(visible_dim) # consider precompute
                    x=torch.randn(visible_dim,dtype=H[k].dtype)
                    temp = torch.einsum('ijkl,j->ikl', H[k], x)
                    temp = torch.einsum('ijk,il->jkl', temp, env[k - 1])
                    temp = torch.einsum('ijk,kjl->il', temp, psi[k])
                    env.append(temp)
                envs.append(env)

            # ===========================================================================
            # 4. sketch  formation
            # ===========================================================================
            for idx in range(sketches_complete, current_sketchdim):
                if j == n - 1:
                    # C[n-2] (Hψ)[n-1]
                    temp = envs[idx][j - 1] @ psi[j]
                    temp = torch.einsum('ijk,ik->j',H[j],temp)
                    sketch[:, idx] = temp
                else:
                    # C[j-1] (Hψ)[j] S[j+1]
                    temp=torch.einsum('ij,jkl->ikl',envs[idx][j - 1],psi[j])
                    temp=torch.einsum('ijlk,ikn->jln',H[j], temp)
                    temp = torch.einsum('ijk,ljk->il',temp,cap)
                    sketch[:, :, idx] = temp


            sketches_complete = current_sketchdim

            # ===========================================================================
            # 5. QR decomposition
            # ===========================================================================
            if j == n - 1:
                Q, R = torch.linalg.qr(sketch)
            else:
                temp = sketch.reshape(cap_dim * visible_dim, current_sketchdim)
                Q, R = torch.linalg.qr(temp)
                Q = Q.reshape((visible_dim, cap_dim, current_sketchdim))
            
            if not (outputdim is None):
                done = True
            else:
                # ===========================================================================
                # 7. error calc
                # ===========================================================================
                done = False
                if current_sketchdim == current_maxdim:
                    done = True
                else:
                    norm_est = np.linalg.norm(sketch) / np.sqrt(current_sketchdim)
                    G = torch.linalg.inv(R.T)

                    err_est = np.sqrt(sum(np.linalg.norm(G, axis=0) ** (-2)) / current_sketchdim)
                    done = err_est <= cutoff * norm_est and current_sketchdim >= current_mindim
            
            # print("error estimation", err_est)
            # print("norm estimation",norm_est)
            # print("Validity of condition 1:",(err_est <= cutoff * norm_est and current_sketchdim >= current_mindim))
            # print("Validity of condition 2:",(current_sketchdim == current_maxdim))
            
            # ===========================================================================
            # 9. cap construction
            # ===========================================================================
            if done:
                if j == n - 1:
                    psi_out[j] = Q.permute(1, 0)
                else:
                    psi_out[j] = Q.permute(2, 0, 1)
                cap_dim = current_sketchdim

                if j == n - 1:
                    temp=torch.einsum("ij,kil->jkl",np.conj(Q), H[j])
                    cap =torch.einsum("ijk,lk->ijl",temp, psi[j])
                else:
                    #print(torch.conj(Q).shape,cap.shape)
                    temp = torch.einsum('ijk,jlm->iklm',torch.conj(Q),cap)

                    #temp = np.tensordot(np.conj(Q), cap, axes=(1, 0))
                    temp = torch.einsum("ijkl,mikn->jlmn",temp, H[j])
                    cap = torch.einsum("ijkl,mlj->ikm",temp, psi[j])
                if accuracychecks:
                    check_randomized_apply(H, psi, cap, psi_out, j+1, verbose=False, cutoff=((100*cutoff) if (maxdim is None) else np.Inf))
                break

            # ===========================================================================
            # 10. sketch correction
            # ===========================================================================
            current_sketchdim = min(current_maxdim, current_sketchdim + sketchincrement)
            old_sketch = sketch

            if j == n - 1:
                sketch = torch.zeros(
                    (
                        old_sketch.shape[0],
                        current_sketchdim,
                    ), dtype=torch.complex128  # Use torch.complex64 if less precision is required
                )
                sketch[:, : old_sketch.shape[1]] = torch.from_numpy(old_sketch)
            else:
                sketch = torch.zeros(
                    (
                        old_sketch.shape[0],
                        old_sketch.shape[1],
                        current_sketchdim,
                    ), dtype=torch.complex128  # Use torch.complex64 if less precision is required
                )
                sketch[:, :, : old_sketch.shape[2]] = old_sketch

    temp = torch.einsum('ijk,lk',cap, psi[0])
    #print(H[0].shape,temp.shape)
    psi_out[0] = torch.einsum('ijk,ljk->il',H[0],temp)
    #psi_out[0] = np.tensordot(H[0], temp, axes=([1,2], [1,2]))
    #print(psi_out[0].shape)
    # mps_round_left(psi_out,cutoff,maxdim)
    psi_out= MPS(psi_out, canform="Left")
    if not (finalround is None):
        psi_out.round(stop = finalround)
    return psi_out

