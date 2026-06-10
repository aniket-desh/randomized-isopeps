import numpy as np
import numpy.linalg as la 
import scipy
from .MPO import MPO
from .MPS import MPS
from .stopping import * 
from .linalg import * 
import time 

def TTSVD(tensor, epsilon):
    d = len(tensor.shape)
    norm_A = np.linalg.norm(tensor)
    delta = epsilon / np.sqrt(d-1) * norm_A
    
    cores = []
    tensor_shape = tensor.shape
    C = tensor.copy()
    r_prev = 1
    
    for k in range(d - 1):
        n_k = tensor_shape[k]
        C = C.reshape(r_prev * n_k, -1)
        U, S, Vh = truncated_svd(C, stop=Cutoff(delta))
        r_k = U.shape[1]
        
        G_k = U.reshape(r_prev, n_k, r_k)
        if k == 0: 
            cores.append(G_k.reshape(G_k.shape[1],G_k.shape[2]))
        else:
            cores.append(G_k)
        C = np.dot(np.diag(S), Vh)
        r_prev = r_k
    
    cores.append(C)  # The last core
    
    return MPS(cores)

#===============================================
# Utility functions
#===============================================

def print_multi_line(*lines, max_width=None):
    """Print multiple lines, with a maximum width.
    """
    if max_width is None:
        import shutil
        max_width, _ = shutil.get_terminal_size()

    max_line_lenth = max(len(ln) for ln in lines)

    if max_line_lenth <= max_width:
        for ln in lines:
            print(ln)

    else:  # pragma: no cover
        max_width -= 10  # for ellipses and pad
        n_lines = len(lines)
        n_blocks = (max_line_lenth - 1) // max_width + 1

        for i in range(n_blocks):
            if i == 0:
                for j, l in enumerate(lines):
                    print(
                        "..." if j == n_lines // 2 else "   ",
                        l[i * max_width:(i + 1) * max_width],
                        "..." if j == n_lines // 2 else "   "
                    )
                print(("{:^" + str(max_width) + "}").format("..."))
            elif i == n_blocks - 1:
                for ln in lines:
                    print("   ", ln[i * max_width:(i + 1) * max_width])
            else:
                for j, ln in enumerate(lines):
                    print(
                        "..." if j == n_lines // 2 else "   ",
                        ln[i * max_width:(i + 1) * max_width],
                        "..." if j == n_lines // 2 else "   ",
                    )
                print(("{:^" + str(max_width) + "}").format("..."))

def check_randomized_apply(H, psi, cap, output, j, verbose=False, cutoff=1e-10):
    sites = sites_for_mpo(H)
 
    # sites * H * psi
    Hsite = np.tensordot(H[0], sites[0], (0,0))
    env = np.tensordot(Hsite, psi[0], (1,0))
    if 0 == j-2:
        mpsmpoenv = env
    for i in range(1,len(H)-1):
        Hsite = np.tensordot(H[i], sites[i], (1,0))
        Hsitepsi = np.tensordot(Hsite, psi[i], (2,1))
        env = np.tensordot(env, Hsitepsi, ([0,1],[0,2]))
        if i == j-2:
            mpsmpoenv = env
    Hsite = np.tensordot(H[-1], sites[-1], (1,0))
    Hsitepsi = np.tensordot(Hsite, psi[-1], (1,1))
    result = float(np.tensordot(env, Hsitepsi, ([0,1],[0,1])))

    # Randomized algorithm result
    env = np.tensordot(mpsmpoenv, cap, ([0,1],[1,2]))
    for i in range(j-1,len(H)):
        env = np.tensordot(env, output[i], (0,0))
        env = np.tensordot(env, sites[i], (0,0))
    result2 = float(env)

    if verbose:
        print("Randomized error check {:6.2e} ({:6.2e}, {:6.2e})".format(np.abs(result - result2)/np.abs(result), result, result2))

    if abs(result - result2) > cutoff * abs(result):
        raise UserWarning("Random check indicates descrepancy between answer and algorithm output ({}, {})".format(result, result2))

def maxlinkdim(M):
    maxbond = 0
    N=len(M)
    for i, t in enumerate(M):
        if i == 0:
            maxbond = t.shape[1]
        elif i == N - 1:
            maxbond = max(maxbond, t.shape[0])
        else:
            b1, b2 = t.shape[0], t.shape[2]
            maxbond = max(maxbond, b1, b2)
    return maxbond  
