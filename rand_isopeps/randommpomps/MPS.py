import numpy as np
import numpy.linalg as la
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# from .rounding import *
from .MPO import MPO
from .linalg import truncated_svd,lq
from .stopping import Cutoff,no_truncation
import copy
import math 

class MPS:
    def __init__(self, mps, canform="None", rounded=False):
        self.tensors = mps
        self.N = len(mps)
        self.canform = canform
        self.rounded = rounded
    
    def copy(self):
        new_mps = copy.deepcopy(self)
        return new_mps
    
    def __len__(self):
        return self.N
        
    def __getitem__(self, index):
        return self.tensors[index]

    def __setitem__(self, index, value):
        self.tensors[index] = value 
    
    #============Orthogonalization============    
    def canonize(self):
        if "None" != self.canform:
            return
        self.canonize_right()
    
    def canonize_left(self):
        if "Left" == self.canform:
            return

        # Qt, R = la.qr(self[-1].T)
        L,Q = lq(self[-1])
        self[-1] = Q
        # L = R.T

        for i in range(self.N - 2, 0, -1):
            A = np.tensordot(self[i],L, (2,0))
            Qt, R = la.qr((np.reshape(A, (A.shape[0],A.shape[1]*A.shape[2]))).T)
            Q = Qt.T
            self[i] = np.reshape(Q, (min(A.shape[0],A.shape[1]*A.shape[2]),A.shape[1],A.shape[2]))
            L= R.T 
        self[0] =  self[0] @ L
        self.canform = "Left"
        
    def canonize_left_blas(self) : 

        if "Left" == self.canform:
            return

        Qt, R = la.qr(self[-1].T)
        self[-1] = Qt.T
        L = R.T

        for i in range(self.N - 2, 0, -1):
            # A = np.einsum("DdY,Yk->Ddk")
            mps_reshaped =self[i].reshape(-1,self[i].shape[2])
            A = mps_reshaped @ L
            A = A.reshape(self[i].shape[0],self[i].shape[1],L.shape[1])
            
            Qt, R = la.qr((np.reshape(A, (A.shape[0],A.shape[1]*A.shape[2]))).T)
            Q = Qt.T
            self[i] = np.reshape(Q, (min(A.shape[0],A.shape[1]*A.shape[2]),A.shape[1],A.shape[2]))
            L= R.T 
        self[0] =  self[0] @ L
        self.canform = "Left"
    
    def canonize_right(self):
        if "Right" == self.canform:
            return
        Q, R = la.qr(self[0])
        self[0] = Q

        for i in range(1, self.N - 1):
            A = np.tensordot(R, self[i], (1, 0))
            Q, R = la.qr(np.reshape(A, (A.shape[0] * A.shape[1], A.shape[2])))
            self[i] = np.reshape(Q, (A.shape[0],A.shape[1],min(A.shape[2],A.shape[0]*A.shape[1])))

        self[-1] = R @ self[-1]
        self.canform = "Right"

    #============Operations============      
    def eval(self, sites):
        A = sites[0] @ self[0]
        for i in range(1, self.N):
            A = sites[i] @ (np.tensordot(A, self[i], axes=(0, 0)))
        return float(A)
    
    def scale(self, scalar):
        self.canonize()
        if "Right" == self.canform:
            if np.iscomplexobj(scalar) and not np.iscomplexobj(self[-1]):
                self[-1] = self[-1].astype(np.complex128)
            self[-1] *= scalar
        else:
            if np.iscomplexobj(scalar) and not np.iscomplexobj(self[0]):
                self[0] = self[0].astype(np.complex128)
            self[0] *= scalar
        
    def add(self, mps2, subtract=False, compress=False, stop = Cutoff(1e-14)):
        mps = []
        
        assert(self[0].shape[0] == mps2[0].shape[0])
        m1 = self[0].shape[1]
        m2 = mps2[0].shape[1]
        new_site = np.zeros((self[0].shape[0], m1+m2),dtype="complex")
        new_site[:,0:m1] = self[0]
        new_site[:,m1:] = mps2[0]
        mps.append(new_site)
        
        for i in range(1,self.N-1):
            assert(self[i].shape[1] == mps2[i].shape[1])
            m1 = self[i].shape[0]            
            n1 = self[i].shape[2]
            m2 = mps2[i].shape[0]
            n2 = mps2[i].shape[2]
            new_site = np.zeros((m1+m2,self[i].shape[1],n1+n2),dtype="complex")
            new_site[0:m1,:,0:n1] = self[i]
            new_site[m1:,:,n1:] = mps2[i]
            mps.append(new_site)
            
        assert(self[-1].shape[1] == mps2[-1].shape[1])
        m1 = self[-1].shape[0]
        m2 = mps2[-1].shape[0]
        new_site = np.zeros((m1+m2,self[-1].shape[1]),dtype="complex")
        new_site[0:m1,:] = self[-1]
        new_site[m1:,:] = (-1 if subtract else 1) * mps2[-1]
        mps.append(new_site)

        mps = MPS(mps)
        if compress:
            mps.round(stop=stop)
        
        return mps

    def sub(self, mps2, **kwargs):
        return self.add(mps2, subtract=True, **kwargs)
    
    def dagger(self):
        """
        Computes the Hermitian conjugate (dagger) of the entire MPS.
        """
        daggered_tensors = [np.conj(tensor) for tensor in self.tensors]
        return MPS(daggered_tensors, canform=self.canform, rounded=self.rounded)
    
    def norm(self):
        self.canonize()
        if "Right" == self.canform:
            return np.linalg.norm(self[-1],'fro')
        else:
            return np.linalg.norm(self[0],'fro')

    def normalize(self):
        self.canonize()
        if "Right" == self.canform:
            self[-1] /= np.linalg.norm(self[-1],'fro')
        else:
            self[0] /= np.linalg.norm(self[0],'fro')       
 
    def round(self, **kwargs):  #TODO add support for other rounding in randomcontraction.rounding
        from .rounding import round_left,round_right #TODO:possibly slow way of handling circular import exception
        if self.rounded:
            return
        elif "Left" == self.canform:
            round_right(self, **kwargs)
        else:
            round_left(self,**kwargs)
    
    def self_inner_product(self):   #TODO: Blas me
        mps_c =self.dagger()
        site = np.einsum("dA,dX->AX",mps_c[0],self[0]) 
        for i in range(1,self.N-1):
            temp = np.einsum("AX,XdY->AdY",site,self[i])
            site = np.einsum("AdY,AdB->BY",temp,mps_c[i])
        site = np.einsum("AY,Yd->Ad",site,self[-1])
        inner_product = np.einsum("Ad,Ad",site,mps_c[-1]) 
        return inner_product
    
    def inner_product(self,other): #TODO: merge fast inner product 
        mps_c = other.dagger()
        site = np.einsum("dA,dX->AX",mps_c[0],self[0]) 
        for i in range(1,self.N-1):
        
            temp = np.einsum("AX,XdY->AdY",site,self[i])
            site = np.einsum("AdY,AdB->BY",temp,mps_c[i])
        site = np.einsum("AY,Yd->Ad",site,self[-1])
        inner_product = np.einsum("Ad,Ad",site,mps_c[-1]) 
        return inner_product

    #============Operator Overloading===========    
    def __mul__(self, other):
        new_tensors = self.tensors.copy()
        
        if np.iscomplexobj(other):
            if other.real == 0:
                other_dtype = np.float64
            else:
                other_dtype = other.dtype
        else:
            other_dtype = other.dtype

        for i, tensor in enumerate(new_tensors):
            if self.canform == "Left" and i == 0:
                new_tensors[i] = tensor.astype(other_dtype) * other
            elif self.canform != "Left" and i == len(new_tensors) - 1:
                new_tensors[i] = tensor.astype(other_dtype) * other

        return MPS(new_tensors, canform=self.canform, rounded=self.rounded)

    def __div__(self, other):
        return (1.0/other) * self
    
    def __add__(self, other):
        return self.add(other, compress = True)

    def __sub__(self, other):
        return self.sub(other, compress=True)
    
    #============Misc============ 
    def display_tensors(self):
        for t in self.tensors:
            print(t.shape)
            
    def bond_size(self,i):
        if i ==0: 
            return self[0].shape[1]
        else:
            return self[i].shape[2]
   
    def max_bond_dim(self):
        maxbond = 0
        for i, t in enumerate(self.tensors):
            if i == 0:
                maxbond = t.shape[1]
            elif i == self.N - 1:
                maxbond = max(maxbond, t.shape[0])
            else:
                b1, b2 = t.shape[0], t.shape[2]
                maxbond = max(maxbond, b1, b2)
        return maxbond  
         
    def size(self):
        return self.N
    
    def show(self, max_width=None):
        from .misc import print_multi_line

        def check_orthogonality(tensor, index):
            if index == 0 or index == self.N - 1:
                reshaped_tensor = tensor
            else:
                reshaped_tensor = tensor.reshape(tensor.shape[0] * tensor.shape[1], tensor.shape[2])

            left_product = np.dot(reshaped_tensor.conj().T, reshaped_tensor)
            right_identity = np.eye(left_product.shape[0])

            left_orthogonal = np.allclose(left_product, right_identity)

            if index == 0 or index == self.N - 1:
                reshaped_tensor = tensor
            else:
                reshaped_tensor = tensor.reshape(tensor.shape[0], tensor.shape[1] * tensor.shape[2])

            right_product = np.dot(reshaped_tensor, reshaped_tensor.conj().T)
            left_identity = np.eye(right_product.shape[0])

            right_orthogonal = np.allclose(right_product, left_identity)

            if left_orthogonal and right_orthogonal:
                return 'both'
            elif left_orthogonal:
                return 'left'
            elif right_orthogonal:
                return 'right'
            else:
                return None

        l1 = ""
        l2 = ""

        for i in range(0, self.N - 1):
            bdim = self.bond_size(i)
            strl = len(str(bdim))
            l1 += f"│{bdim}"

            orthogonality = check_orthogonality(self[i], i)
            if orthogonality == 'left':
                l2 += ">" + ("-" if bdim < 100 else "━") * max(strl , 1)
            elif orthogonality == 'right':
                l2 += "<" + ("-" if bdim < 100 else "━") * max(strl , 1)
            elif orthogonality == 'both':
                l2 += "■" + ("-" if bdim < 100 else "━") * max(strl , 1)
            else:
                l2 += "●" + ("-" if bdim < 100 else "━") * max(strl , 1)
        
        orthogonality = check_orthogonality(self[-1], self.N - 1)
        l1 += "│"

        if orthogonality == 'left':
            l2 += ">"
        elif orthogonality == 'right':
            l2 += "<"
        elif orthogonality == 'both':
            l2 += "■"
        else:
            l2 += "●"

        print_multi_line(l1, l2, max_width=max_width)
        
    def contract_all(self):
        # start with the first tensor
        result = self[0]
        
        # contract the tensors one by one
        for i in range(1,self.N):
            print(i)
            
            if i == 1: 
                result = np.tensordot(result, self[i], axes=([1], [0]))  # contract over the last dimension of result and the first dimension of mps[i]
            else:
                result = np.tensordot(result, self[i], axes=([-1], [0])) 

        return result
    
    def is_left_orth(self):
        # Left to right sweep
        for i in range(0,self.N-1):
            tensor = self[i]
            if i == 0 :
                reshaped_tensor = tensor
            else:
                reshaped_tensor = tensor.reshape(tensor.shape[0]*tensor.shape[1],tensor.shape[2])
            product = np.dot(reshaped_tensor.conj().T, reshaped_tensor)
            identity = np.eye(product.shape[0])
            if not np.allclose(product, identity):
                print(f"Tensor at index {i} is not right-orthogonal.")
                plt.imshow(np.abs(product), cmap='mako', interpolation='nearest')
                plt.title(f"Heatmap of Product Matrix at tensor {i}")
                plt.colorbar()
                plt.show()
                assert False, "Orthogonality test failed"
        print("All tensors are left-orthogonal.")
          
    def is_right_orth(self):
        # Right to left sweep
        for i in reversed(range(2, self.N)):
            tensor = self[i]
            if i == self.N-1 :
                reshaped_tensor = tensor
            else:
                reshaped_tensor = tensor.reshape(tensor.shape[0],tensor.shape[1]*tensor.shape[2])
            product = np.dot(reshaped_tensor, reshaped_tensor.conj().T)
            identity = np.eye(product.shape[0])
            
            if not np.allclose(product, identity):
                print(f"Tensor at index {i} is not left-orthogonal.")
                # Visualize the product matrix as a heatmap
                plt.imshow(np.abs(product), cmap='mako', interpolation='nearest')
                plt.title(f"Heatmap of Product Matrix at tensor {i}")
                plt.colorbar()
                plt.show()
                assert False, "Orthogonality test failed" 
        print("All tensors are right-orthogonal.")

    @staticmethod 
    def mps_to_mpo(mps):
        """Converts an MPS to an MPO by adding an auxillary bond dimension of 1 """
        mpo = []
        # First tensor: shape (d, X, 1)
        first_tensor = mps.tensors[0]
        d, X = first_tensor.shape
        new_tensor = np.zeros((d, X, 1), dtype=first_tensor.dtype)
        new_tensor[:, :, 0] = first_tensor
        mpo.append(new_tensor)

        # Middle tensors: shape (X1, d, X2, 1)
        for tensor in mps.tensors[1:-1]:
            X1, d, X2 = tensor.shape
            new_tensor = np.zeros((X1, d, X2, 1), dtype=tensor.dtype)
            new_tensor[:, :, :, 0] = tensor
            mpo.append(new_tensor)
        
        # Last tensor: shape (X, d, 1)
        last_tensor = mps.tensors[-1]
        X, d = last_tensor.shape
        new_tensor = np.zeros((X, d, 1), dtype=tensor.dtype)
        new_tensor[:, :, 0] = last_tensor
        mpo.append(new_tensor)

        return MPO(mpo)
    
    @staticmethod
    def random_mps(n, m, d=2, random_tensor=np.random.randn, dtype=float):
        return MPS([random_tensor(d, m).astype(dtype)] + [random_tensor(m, d, m).astype(dtype) for _ in range(n - 2)] + [random_tensor(m, d).astype(dtype)])
    
    @staticmethod
    def random_incremental_mps(n, d, seed,dtype=np.complex128):
        mps = [np.random.randn(d, seed).astype(dtype)]  # Initialize with the first tensor, bond dimension 1 for the first tensor
        for i in range(0, n-2):
            mps.append(np.random.randn(seed+i,d,seed+i+1))
        mps.append(np.random.randn(seed+i+1,d))
        return MPS(mps)


 
def norm(thing):
    return thing.norm()
