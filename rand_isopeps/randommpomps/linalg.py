import numpy as np
import numpy.linalg as la
from .stopping import Cutoff,FixedDimension
from scipy.sparse.linalg import svds

def truncated_svd(A, stop = Cutoff(1e-14), abstol = 0.0):
    '''
    Output U, s, Vt of the minimal sizes
    such that norm(A - U * diag(s) * Vt, 'fro')
    <= cutoff * norm(A, 'fro')
    '''
    
    if stop.maxdim is None:
        stop.maxdim = np.Inf

    U, s, Vt = la.svd(A, full_matrices=False)
    if (stop.cutoff is None) and (stop.outputdim is None) and abstol == 0.0:
        return U, s, Vt
    idx = len(s)
    if stop.outputdim is None:
        snorm = la.norm(s)
        cutoff = (snorm * stop.cutoff + abstol) ** 2
        tail_norm_sq = 0.0
        cutoff_met = False
        
        for i in range(len(s)-1,-1,-1):
            tail_norm_sq += s[i] ** 2
            if tail_norm_sq > cutoff:
                idx = i+1
                cutoff_met = True
                break
        if cutoff_met:
            idx = min(idx, stop.maxdim)
        else:
            idx = 0
        
    else:
        idx = min(idx, stop.outputdim)
        
    return U[:,0:idx], s[0:idx], Vt[0:idx, :]

def truncated_eigendecomposition(A, stop = Cutoff(1e-14)):
    '''
    Perform a truncated eigendecomposition on matrix A such that the sum of 
    the eigenvalues beyond the truncation point is less than the specified cutoff.
    '''
 
    eigenvalues, eigenvectors = la.eigh(A)
    idx = np.abs(eigenvalues).argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    if (stop.cutoff is None) and (stop.outputdim is None):    
        return eigenvalues, eigenvectors
    
    truncation_idx = len(eigenvalues)  # Default to no truncation.
    if stop.outputdim is None:
        total_frobenius_norm = np.sqrt(np.sum(eigenvalues**2))
        frobenius_norm_cutoff = total_frobenius_norm * stop.cutoff**2
        tail_frobenius_norm = 0.0
        
        for i in range(len(eigenvalues) - 1, -1, -1):
            tail_frobenius_norm += eigenvalues[i]**2
            if np.sqrt(tail_frobenius_norm) > frobenius_norm_cutoff:
                truncation_idx = i + 1
                break
        truncation_idx = min(truncation_idx, stop.maxdim)
    else:
        truncation_idx = min(truncation_idx, stop.outputdim)

    return eigenvalues[:truncation_idx], eigenvectors[:, :truncation_idx], np.conj(eigenvectors[:, :truncation_idx]).T

def lq(A):
    '''
    A simple wrapper for QR 
    '''
    Qt, R = la.qr(A.T)
    return R.T, Qt.T

def lanczos_fun(v, matvec, fun=np.exp, k=5):
    if isinstance(matvec, np.ndarray):
        A = matvec
        matvec = lambda x: A @ x
    Q = np.zeros((len(v), k+1), dtype=np.complex128)
    gamma = np.linalg.norm(v)
    Q[:,0] = v / gamma
    T = np.zeros((k+1,k+1), dtype=np.complex128)
    qold = np.zeros(Q[:,0].shape)
    beta = 0.0
    for i in range(k):
        Q[:,i+1] = matvec(Q[:,i]) - beta * qold
        T[i,i] = np.real(Q[:,i+1].conj().T @ Q[:,i])
        Q[:,i+1] -= T[i,i] * Q[:,i]
        beta = np.linalg.norm(Q[:,i+1])
        if beta < 1e-12 * gamma:
            k = i+1
            break
        T[i,i+1] = beta
        T[i+1,i] = beta
        Q[:,i+1] = Q[:,i+1] / beta
        qold = Q[:,i]
    T = T[:k,:k]
    Q = Q[:,:k]
    Tevals, Tevecs = np.linalg.eigh(T)
    e1 = np.zeros(k, dtype=v.dtype)
    e1[0] = 1
    output = gamma * (Q @ (Tevecs @ (fun(Tevals) * (Tevecs.conj().T @ e1))))
    return output

def lan_exp(v, A, t=1, k=5):
    return lanczos_fun(v, A, fun=lambda x: np.exp(t*x), k = k)

def krylov_matrix_exp(v, A, h=1, k=5):
    nrm = np.linalg.norm(v)
    v = v / nrm
    V, H = arnoldi(A, v, k)

    exp_hH = np.linalg.expm(h * H[:k, :k])
    result = nrm * np.dot(V[:, :k], exp_hH[:, 0])
    return result

def arnoldi(A, v0, k):
    """
    Arnoldi's method with partial reorthogonalization
    """
    n = A.shape[0]
    V = np.zeros((n, k + 1), dtype=A.dtype)
    H = np.zeros((k + 1, k), dtype=A.dtype)
    
    V[:, 0] = v0 / np.linalg.norm(v0)
    for m in range(k):
        vt = A @ V[:, m]
        
        # orthogonalize vt against all previous vectors in V
        for j in range(m + 1):
            H[j, m] = np.vdot(V[:, j], vt)  #  np.vdot invokes complex conjugate transpose here
            vt -= H[j, m] * V[:, j]
        
        H[m + 1, m] = np.linalg.norm(vt)
        
        # reorthogonalize 
        for j in range(m + 1):
            correction = np.vdot(V[:, j], vt)
            vt -= correction * V[:, j]
            H[j, m] += correction
        
        H[m + 1, m] = np.linalg.norm(vt)
        
        # update basis 
        if H[m + 1, m] > 1e-10 and m != k - 1:
            V[:, m + 1] = vt / H[m + 1, m]
    
    return V, H
