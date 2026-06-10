import numpy as np
import numpy.linalg as la
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None
try:
    import seaborn as sns
except Exception:
    sns = None

from .linalg import truncated_svd,truncated_eigendecomposition

from .MPS import MPS
from .stopping import Cutoff,no_truncation
import copy
import math 


#============ Basic MPS Rounding============  
def round_left(mps, **kwargs):
    if mps.rounded:
        if mps.canform == "Left":
            return
        else:
            mps.canonize_left()
            return
    mps.canonize_right()
    
    U, s, Vt = truncated_svd(mps[-1], **kwargs)
    mps[-1] = Vt

    for i in range(mps.N - 2, 0, -1):
        A = mps[i] @ (U * s)
        U, s, Vt = truncated_svd(np.reshape(A, (A.shape[0], A.shape[1] * A.shape[2])), **kwargs)
        mps[i] = np.reshape(Vt, (len(s), A.shape[1], A.shape[2]))

    mps[0] = mps[0] @ (U * s)

    mps.canform = "Left"
    mps.rounded = True

def round_right(mps, **kwargs):
    if mps.rounded:
        if mps.canform == "Right":
            return
        else:
            mps.canonize_right()
            return
        
    mps.canonize_left()
    
    U, s, Vt = truncated_svd(mps[0], **kwargs)
    mps[0] = U

    for i in range(1, mps.N - 1):
        A = np.tensordot(s[:,np.newaxis] * Vt, mps[i], (1,0))
        U, s, Vt = truncated_svd(np.reshape(A, (A.shape[0] * A.shape[1], A.shape[2])), **kwargs)
        mps[i] = np.reshape(U, (A.shape[0], A.shape[1], len(s)))

    mps[-1] = np.tensordot(s[:,np.newaxis] * Vt, mps[-1], (1,0))

    mps.canform = "Right"
    mps.rounded = True
    
#============ Rounding algoithm from https://tensornetwork.org/mps/index.html#Perez-Garcia:2007_6 ===========  

def density_matrix_rounding(mps,stop=Cutoff(1e-14)):
    mps_c = mps.dagger()
    N = mps.N
    mps_out = [None] * N
    E= [None] * (N-1)

    # ==========================================
    # 0. [Envs]
    # ==========================================
    # E[0] = np.einsum("dX,dA->XA",mps[0],mps_c[0])
    E[0] =  mps[0].T @ mps_c[0] # Resulting shape (XA)
    for i in range(1,N-1):
        # E[i] = np.einsum("XA,XdY->AdY",E[i-1],mps[i])
        E[i] = E[i-1].T @ mps[i].reshape(mps[i].shape[0],-1) # Resulting shape A,dY
        E[i] = E[i].reshape(E[i].shape[0],mps[i].shape[1],mps[i].shape[2]) #Reshape to (A,d,Y)
        
        # E[i] = np.einsum("AdY,AdB->YB",E[i],mps_c[i])
        E_transposed = E[i].transpose(2,0,1)
        E[i] = E_transposed.reshape(E_transposed.shape[0],-1) @ mps_c[i].reshape(-1,mps_c[i].shape[-1])
    
    # ==========================================
    # 3. [First DM]
    # ==========================================
    
    # rho = np.einsum("XA,Xd->dA",E[-1],mps[-1])  
    rho = (E[-1].T @ mps[-1]).T
    # rho = np.einsum("dA,Ak->dk",rho,mps_c[-1])  
    rho = rho @ mps_c[-1]
    _, U, Udag = truncated_eigendecomposition(rho,stop=stop)
    mps_out[-1] = U
    
    # ==========================================
    # 3. [First Cap]
    # ==========================================
    # M_top = np.einsum("dk,Xd->Xk", Udag, mps[-1])
    M_top =  mps[-1] @ Udag

    for j in reversed(range(1, N-1)):
        # top = np.einsum("Yk,XdY->Xdk",M_top,mps[j])
        top = mps[j].reshape(-1,mps[j].shape[-1]) @ M_top #Resuting shape (Xd,k)
        top = top.reshape(mps[j].shape[0],mps[j].shape[1],M_top.shape[1])
        
        bottom = np.conj(top)
        
        # rho = np.einsum("XA,Xdk->Adk",E[j-1],top)
        rho = E[j-1].T @ top.reshape(top.shape[0],-1) # Resulting shape (A,dk)
        rho = rho.reshape(rho.shape[0],top.shape[1],top.shape[2])
        
        # rho = np.einsum("Adk,Alj->dklj",rho,bottom)
        rho_transposed = rho.transpose(1,2,0)
        rho = rho_transposed.reshape(-1,rho_transposed.shape[2]) @ bottom.reshape(bottom.shape[0],-1) # Resulting shape dk,lj
        
        _,U,Udag = truncated_eigendecomposition(rho,stop=stop) #U is (dk,x)

        U = U.reshape(top.shape[1],top.shape[2],U.shape[-1]) #(d,k,x)
        U = U.transpose(2,0,1) #(x,d,k)
        mps_out[j] = U
        
        # M_top = np.einsum("xdk,Xdk->Xx",U,top)
        top_transposed = top.transpose(1,2,0)
        M_top = U.reshape(U.shape[0],-1) @ top_transposed.reshape(-1,top_transposed.shape[-1])
        M_top = M_top.T

    # mps_out[0]= np.einsum("dX,Xx->dx",mps[0],M_top)
    mps_out[0] = mps[0] @ M_top
    return MPS(mps_out)

#============ Rounding Techniques from arxiv.org/abs/2110.04393 ============  
def dass_round(mps,stop=Cutoff(1e-14)):
    if mps.rounded==False:
        mps.canonize_left()

    if stop.cutoff!= None:
        normY = mps.norm()
        tau = stop.cutoff * normY / math.sqrt(mps.N - 1)
        stop = Cutoff(tau)

    Q,R = la.qr(mps[0], mode='reduced')
    U, s, Vt = truncated_svd(R, stop=stop)
    mps[0] = Q @ U
    for i in range(1, mps.N - 1):
        A = np.tensordot(s[:,np.newaxis] * Vt, mps[i], (1,0))
        
        Q,R = la.qr(np.reshape(A, (A.shape[0] * A.shape[1], A.shape[2])), mode='reduced')
        U, s, Vt = truncated_svd(R, stop=stop)
        mps[i] = np.reshape(Q @ U, (A.shape[0], A.shape[1], len(s)))

    mps[-1] = np.tensordot(s[:,np.newaxis] * Vt, mps[-1], (1,0))
    mps.canform = "left"
    mps.rounded = True
    
def orth_then_rand(mps,r):
    mps.round_left()

    Z = mps[0]
    sketch = Z @ np.random.randn(Z.shape[1],r)
    
    Q,_ = la.qr(sketch, mode='reduced')
    mps[0] = Q
    
    M = Q.T @ Z
    mps[1] = np.tensordot(M,mps[1],axes=(1,0))
    
    for i in range(1, mps.N-1):
        Z = mps[i]
        sketch = np.tensordot(Z ,np.random.randn(Z.shape[2],r),axes=(2,0))
        
        Q,_ = la.qr(np.reshape(sketch,(sketch.shape[0]*sketch.shape[1],sketch.shape[2])),mode='reduced')
        mps[i] = np.reshape(Q,(mps[i-1].shape[-1],mps[i].shape[1],Q.shape[-1]))
        
        M = Q.T @ np.reshape(Z,(Z.shape[0]*Z.shape[1],Z.shape[2]))
        mps[i+1] = np.tensordot(M,mps[i+1],axes=(1,0))
                 
def rand_then_orth(mps,r,finalround=False,stop=Cutoff(1e-14)):
    if r is None:
        r = mps[0].shape[1]
    #==================Environment tensors ==================
    N = mps.N
    R = MPS.random_mps(N,r,mps[1].shape[1])
    W = TTpartialContractionsRL(mps,R)
    #==================Random Section==================
    Q, _ = la.qr(np.tensordot(mps[0], W[0], axes=(1, 0)), mode='reduced')
    M = np.tensordot(Q.T, mps[0], axes=(1, 0))
    mps[0] = Q

    for j in range(1, N - 1):
        mps[j] = np.tensordot(M,mps[j],axes=(1,0))
        Z = np.reshape(mps[j],(mps[j].shape[0]*mps[j].shape[1],mps[j].shape[2]))
        sketch = np.tensordot(Z,W[j],axes=(1,0))

        Q,_ = la.qr(sketch,mode='reduced')
        mps[j]= np.reshape(Q,(mps[j-1].shape[-1],mps[j].shape[1],Q.shape[1]))
        M = np.tensordot(Q.T,Z,axes=(1,0))
        
    mps[-1]= np.tensordot(M,mps[-1],axes=(1,0))
    mps.rounded = True
    #==================Final rounding==================
    if finalround:
        if stop.cutoff != None:
            normY = np.linalg.norm(mps[-1],'fro')
            tau = stop.cutoff * normY / math.sqrt(mps.N - 1)
            stop= Cutoff(tau)

        Q,R = la.qr(mps[0], mode='reduced')
        U, s, Vt = truncated_svd(R, stop=stop)
        mps[0] = Q @ U

        for i in range(1, mps.N - 1):
            A = np.tensordot(s[:,np.newaxis] * Vt, mps[i], (1,0))
            Q,R = la.qr(np.reshape(A, (A.shape[0] * A.shape[1], A.shape[2])), mode='reduced')
            U, s, Vt = truncated_svd(R, stop=stop)
            mps[i] = np.reshape(Q @ U, (A.shape[0], A.shape[1], len(s)))

        mps[-1] = np.tensordot(s[:,np.newaxis] * Vt, mps[-1], (1,0))

def nystrom_round(mps, l, stop=no_truncation()):
    if l is None:
        l = mps[0].shape[1]
    #================== Environment tensors ==================
    L = MPS.random_mps(mps.N, l, mps[1].shape[1])
    rho = int(np.ceil(1.5* l))
    # Heuristic choice of oversampling ranks based on Yuji- Fast and stable randomized low rank matrix approximation 2020
    R = MPS.random_mps(mps.N, rho, mps[1].shape[1])
    
    WL = TTpartialContractionsLR(L,mps)
    WR = TTpartialContractionsRL(mps,R)

    # ================== Nystrom Sketch ==================
    U, s, Vt = truncated_svd(WL[0] @ WR[0], stop=stop)
    S_inv_sqrt = np.conj(np.diag(1 / np.sqrt(s)))
    left_factor = WR[0] @ Vt.T @ S_inv_sqrt
    right_factor = S_inv_sqrt @ U.T @ WL[0]

    mps[0] = mps[0] @ left_factor

    for j in range(1, mps.N - 1):
        U, s, Vt = truncated_svd(WL[j] @ WR[j], stop=stop)
        S_inv_sqrt = np.conj(np.diag(1 / np.sqrt(s)))
        left_factor = WR[j] @ Vt.T @ S_inv_sqrt

        mps[j] = np.tensordot(mps[j], left_factor, axes=(2, 0))
        mps[j] = np.tensordot(right_factor, mps[j], axes=(1,0))
        mps[j] = np.reshape(mps[j], (right_factor.shape[0], mps[j].shape[1], left_factor.shape[1]))

        #X[j] = np.reshape(Y[j],(Y[j].shape[0]*Y[j].shape[1],Y[j].shape[2])) @ left_factor
        #X[j] = np.reshape(X[j],(int(X[j].shape[0]/Y[j].shape[1]),X[j].shape[1]*Y[j].shape[1]))
        #X[j]= right_factor @ X[j]
        
        right_factor = S_inv_sqrt @ U.T @ WL[j]

    mps[-1] = right_factor @ mps[-1]  
    
def TTpartialContractionsLR(X,Y):
        WL = [None] * (Y.N-1)
        WL[0] = X[0].T @ Y[0]
        for j in range(1, Y.N - 1):
            WL[j] = np.tensordot(Y[j], WL[j - 1], axes=(0, 1))
            WL[j] = np.tensordot(X[j], WL[j], axes=((1, 0), (0, 2)))
        return WL 
    
def TTpartialContractionsRL(X,Y):
        N = len(X)
        WR = [None] * (N - 1)
        WR[N-2] = X[-1] @ Y[-1].T

        for j in range(N-2, 0, -1):
            WR[j-1]  = np.tensordot(X[j], WR[j], axes=(2, 0))
            WR[j-1] = np.tensordot(WR[j-1], Y[j], axes=((1, 2), (1, 2)))
        return WR

#============ Rounding Techniques from arxiv.org/abs/2110.04393 (BLAS version) ============  

def dass_round_blas(mps,stop=Cutoff(1e-14)):
    if mps.rounded==False:
        mps.canonize_left_blas()
    # if stop.cutoff != None:
    #     normY = mps.norm()
    #     tau = stop.cutoff * normY / math.sqrt(mps.N - 1)
    #     stop = Cutoff(tau)

    Q,R = la.qr(mps[0], mode='reduced')
    U, s, Vt = truncated_svd(R, stop=stop)
    mps[0] = Q @ U
    for i in range(1, mps.N - 1):
        temp = s[:,np.newaxis] * Vt
        # A = np.einsum("ik,klY->ilY",temp,mps[i])
        mps_reshaped = mps[i].reshape(mps[i].shape[0],-1) #Reshape to (DX,l(EY))
        A = temp @ mps_reshaped #resulting shape(p,l(EY))
        A = A.reshape(A.shape[0],mps[i].shape[1],-1) # reshape to (p,l,EY)
        
        Q,R = la.qr(np.reshape(A, (A.shape[0] * A.shape[1], A.shape[2])), mode='reduced')
        U, s, Vt = truncated_svd(R, stop=stop)
        mps[i] = np.reshape(Q @ U, (A.shape[0], A.shape[1], len(s)))

    mps[-1] = (s[:,np.newaxis] * Vt) @ mps[-1]
    mps.canform = "Right"
    mps.rounded = True

def orth_then_rand_blas(mps,r):
    mps.round_left()

    Z = mps[0]
    sketch = Z @ np.random.randn(Z.shape[1],r)
    
    Q,_ = la.qr(sketch, mode='reduced')
    mps[0] = Q
    
    M = Q.T @ Z
    # mps[1] = np.tensordot(M,mps[1],axes=(1,0))
    mps_reshaped = mps[1].reshape(mps[1].shape[0],-1)
    temp = M @ mps_reshaped
    mps[1] = temp.reshape(M.shape[0],mps[1].shape[1],mps[1].shape[2])
    
    for i in range(1, mps.N-1):
        Z = mps[i]
        # sketch = np.tensordot(Z ,np.random.randn(Z.shape[2],r),axes=(2,0))
        Z_reshaped = Z.reshape(Z.shape[0]*Z.shape[1],Z.shape[2]) 
        sketch = Z_reshaped @ np.random.randn(Z.shape[2],r)
        
        # print(sketch.shape)
        Q,_ = la.qr(sketch, mode='reduced')
        mps[i] = np.reshape(Q,(mps[i-1].shape[-1],mps[i].shape[1],Q.shape[-1]))
        
        M = Q.T @ np.reshape(Z,(Z.shape[0]*Z.shape[1],Z.shape[2]))
        mps[i+1] = np.tensordot(M,mps[i+1],axes=(1,0))

def rand_then_orth_blas(mps,r,finalround=False,stop=Cutoff(1e-14)):
    if r is None:
        if stop.cutoff is None and stop.outputdim is not None:
            r = stop.outputdim
        else:
            r = mps[0].shape[1]
    #==================Environment tensors ==================
    N = mps.N
    R = MPS.random_mps(N,r,mps[1].shape[1])
    W = TTpartialContractionsRL_blas(mps,R)
    #==================Random Section==================
    Q, _ = la.qr(np.tensordot(mps[0], W[0], axes=(1, 0)), mode='reduced')
    # M = np.tensordot(Q.T, mps[0], axes=(1, 0))
    M = Q @ mps[0]
    mps[0] = Q

    for j in range(1, N - 1):
        # mps[j] = np.tensordot(M,mps[j],axes=(1,0)) 
        mps_reshaped = mps[j].reshape(mps[j].shape[0],-1)
        temp = M @ mps_reshaped
        mps[j] = temp.reshape(M.shape[0],mps[j].shape[1],mps[j].shape[2])
        
        Z = np.reshape(mps[j],(mps[j].shape[0]*mps[j].shape[1],mps[j].shape[2]))
        # sketch = np.tensordot(Z,W[j],axes=(1,0))
        sketch = Z @ W[j]
        Q,_ = la.qr(sketch,mode='reduced')
        mps[j]= np.reshape(Q,(mps[j-1].shape[-1],mps[j].shape[1],Q.shape[1]))
        M = Q.T @ Z 
        
    mps[-1] = M @ mps[-1]
    mps.rounded = True
    # #==================Final rounding==================
    # if finalround:
    #     if stop.cutoff != None:
    #         normY = np.linalg.norm(mps[-1],'fro')
    #         tau = stop.cutoff * normY / math.sqrt(mps.N - 1)
    #         stop= Cutoff(tau)

    #     Q,R = la.qr(mps[0], mode='reduced')
    #     U, s, Vt = truncated_svd(R, stop=stop)
    #     mps[0] = Q @ U
    #     #Update to use dass round blas 
    #     for i in range(1, mps.N - 1):
    #         A = np.tensordot(s[:,np.newaxis] * Vt, mps[i], (1,0))
    #         Q,R = la.qr(np.reshape(A, (A.shape[0] * A.shape[1], A.shape[2])), mode='reduced')
    #         U, s, Vt = truncated_svd(R, stop=stop)
    #         mps[i] = np.reshape(Q @ U, (A.shape[0], A.shape[1], len(s)))

    #     mps[-1] = np.tensordot(s[:,np.newaxis] * Vt, mps[-1], (1,0))

def nystrom_round_blas(mps, l, stop=no_truncation()):
    if l is None:
        l = mps[0].shape[1]
    #================== Environment tensors ==================
    L = MPS.random_mps(mps.N, l, mps[1].shape[1])
    rho = int(np.ceil(1.5* l))
    # Heuristic choice of oversampling ranks based on Yuji- Fast and stable randomized low rank matrix approximation 2020
    R = MPS.random_mps(mps.N, rho, mps[1].shape[1])
    
    WL = TTpartialContractionsLR_blas(L,mps)
    WR = TTpartialContractionsRL_blas(mps,R)

    # ================== Nystrom Sketch ==================
    U, s, Vt = truncated_svd(WL[0] @ WR[0], stop=stop)
    S_inv_sqrt = np.conj(np.diag(1 / np.sqrt(s)))
    left_factor = WR[0] @ Vt.T @ S_inv_sqrt
    right_factor = S_inv_sqrt @ U.T @ WL[0]

    mps[0] = mps[0] @ left_factor

    for j in range(1, mps.N - 1):
        U, s, Vt = truncated_svd(WL[j] @ WR[j], stop=stop)
        S_inv_sqrt = np.conj(np.diag(1 / np.sqrt(s)))
        left_factor = WR[j] @ Vt.T @ S_inv_sqrt
        
        # mps[j] = np.tensordot(mps[j], left_factor, axes=(2, 0))
        mps_reshaped = mps[j].reshape(-1,mps[j].shape[2])
        temp = mps_reshaped @ left_factor
        mps[j] = temp.reshape(mps[j].shape[0],mps[j].shape[1],left_factor.shape[1])
        
        # mps[j] = np.tensordot(right_factor, mps[j], axes=(1,0))
        mps_reshaped = mps[j].reshape(mps[j].shape[0],-1)
        temp = right_factor @ mps_reshaped
        mps[j] = temp.reshape(right_factor.shape[0],mps[j].shape[1],mps[j].shape[2])

        mps[j] = np.reshape(mps[j], (right_factor.shape[0], mps[j].shape[1], left_factor.shape[1]))
        
        right_factor = S_inv_sqrt @ U.T @ WL[j]

    mps[-1] = right_factor @ mps[-1]

def TTpartialContractionsLR_blas(X,Y):
        WL = [None] * (Y.N-1)
        WL[0] = X[0].T @ Y[0]
        for j in range(1, Y.N - 1):
            # WL[j] = np.tensordot(Y[j], WL[j - 1], axes=(0, 1))
            Y_reshaped = Y[j].reshape(Y[j].shape[0],-1)
            temp = WL[j-1] @ Y_reshaped
            temp = temp.reshape(WL[j-1].shape[0],Y[j].shape[1],Y[j].shape[2])
            WL[j] = temp.transpose(1,2,0)
            
            # WL[j] = np.tenspordot(X[j], WL[j], axes=((1, 0), (0, 2)))
            X_reshaped = X[j].reshape(-1,X[j].shape[2])
            WL_transposed = WL[j].transpose(1,2,0)
            WL_transposed = np.ascontiguousarray(WL_transposed) #Resolves extra view 
            WL_reshaped = WL_transposed.reshape(WL_transposed.shape[0],-1)
            temp = WL_reshaped @ X_reshaped
            WL[j]= temp.T
        return WL 
    
def TTpartialContractionsRL_blas(X,Y):
        N = len(X)
        WR = [None] * (N - 1)
        WR[N-2] = X[-1] @ Y[-1].T

        for j in range(N-2, 0, -1):
            # WR[j-1]  = np.tensordot(X[j], WR[j], axes=(2, 0))
            X_reshaped = X[j].reshape(-1,X[j].shape[2]) 
            temp = X_reshaped @ WR[j]
            WR[j-1] = temp.reshape(X[j].shape[0],X[j].shape[1],WR[j].shape[1])
            
            # WR[j-1] = np.tensordot(WR[j-1], Y[j], axes=((1, 2), (1, 2)))
            WR_reshaped = WR[j-1].reshape(WR[j-1].shape[0],-1)
            Y_transposed = Y[j].transpose(1,2,0)
            Y_reshaped = Y_transposed.reshape(-1,Y_transposed.shape[2])
            WR[j-1] = WR_reshaped @ Y_reshaped
        return WR
 