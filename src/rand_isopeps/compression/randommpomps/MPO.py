import numpy as np
import numpy.linalg as la 
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None
import copy
from .stopping import Cutoff,no_truncation
from .linalg import lq,truncated_svd

class MPO:
    def __init__(self, mpo, canform="None", rounded=False):
        self.tensors = mpo
        self.N = len(mpo)
        self.canform = canform
        self.rounded = rounded
    
    def __len__(self):
        return self.N
    
    def copy(self):
        new_mpo = copy.deepcopy(self)
        return new_mpo
    
    def __getitem__(self, index):
        return self.tensors[index]
    
    def __setitem__(self, index, value):
        self.tensors[index] = value 
    
    def canonize(self):
        if "None" != self.canform:
            return
        self.canonize_right()
        
    def canonize_right(self):
        if "Right" == self.canform:
            return
        
        T = self[0]
        T = np.transpose(T, (0,2,1))
        T = np.reshape(T, (T.shape[0]*T.shape[1], T.shape[2]))
        Q, R = la.qr(T)
        Q = np.reshape(Q, (self[0].shape[0],self[0].shape[2],Q.shape[1]))
        Q = np.transpose(Q, (0,2,1))
        self[0] = Q

        for i in range(1, self.N - 1):
            A = np.tensordot(R, self[i], (1, 0))
            A = np.transpose(A, (0,1,3,2))
            Q, R = la.qr(np.reshape(A, (A.shape[0]*A.shape[1]*A.shape[2], A.shape[3])))
            Q = np.reshape(Q, (A.shape[0], A.shape[1], A.shape[2], Q.shape[1]))
            self[i] =  np.transpose(Q, (0,1,3,2))  
    
        self[-1] = np.tensordot(R , self[-1],axes=(1,0))
        self.canform = "Right"

    right_canonize = canonize_right

    def canonize_left(self):
        if "Left" == self.canform:
            return

        T = self[-1]
        reshaped_tensor = np.reshape(T,(T.shape[0],T.shape[1]*T.shape[2]))
        L, Q = lq(reshaped_tensor)
        self[-1] = np.reshape(Q,(Q.shape[0],T.shape[1],T.shape[2]))
        
        for i in range(self.N - 2, 0, -1):
            A = np.tensordot(self[i], L, (2,0))
            A = np.transpose(A, (0,1,3,2))
            L, Q = lq(np.reshape(A, (A.shape[0],A.shape[1]*A.shape[2]*A.shape[3])))
            self[i] = np.reshape(Q, (Q.shape[0],A.shape[1],A.shape[2],A.shape[3]))

        last_tensor = np.tensordot(self[0],L,axes=(1,0))
        self[0] = np.transpose(last_tensor,(0,2,1))
        self.canform = "Left"

    left_canonize = canonize_left

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

    def dagger(self):
        """
        Computes the Hermitian conjugate (dagger) of the entire MPO.
        just iterate though and conjugate instead. 
        """
        daggered_tensors = [np.conj(tensor) for tensor in self.tensors]
        return MPO(daggered_tensors, canform=self.canform, rounded=self.rounded)
    
    def norm(self):
        self.canonize()
        if "Right" == self.canform:
            reshaped_tensor = np.reshape(self[-1],(self[-1].shape[0],self[-1].shape[1]*self[-1].shape[2]))
            return np.linalg.norm(reshaped_tensor,'fro')
        else:
            reshaped_tensor = np.reshape(self[0],(self[0].shape[0]*self[0].shape[2],self[0].shape[1]))
            return np.linalg.norm(reshaped_tensor,'fro')
        
    def normalize(self):
        self.canonize()
        if "Right" == self.canform:
            self[-1] /= np.linalg.norm(self[-1].reshape(self[-1].shape[0],-1),'fro')
        else:
            self[0] /= np.linalg.norm(self[0].reshape(self[0].shape[1],-1),'fro')   
    
    def __mul__(self, down):
        mpo = []
        site = np.tensordot(self[0], down[0], (2,0)) # site(self0,self1,down1,down2)
        site = np.reshape(site, (site.shape[0], site.shape[1]*site.shape[2], site.shape[3]))
        mpo.append(site)

        assert(len(self) == len(down))
        for i in range(1, len(self)-1):
            site = np.tensordot(self[i], down[i], (3,1)) # site(self0,self1,self2,down0,down2,down3)
            site = np.transpose(site, (0,3,1,2,4,5)) # site(self0,down0,self1,self2,down2,down3)
            site = np.reshape(site, (site.shape[0]*site.shape[1], site.shape[2], site.shape[3]*site.shape[4], site.shape[5]))
            mpo.append(site)

        site = np.tensordot(self[-1], down[-1], (2,1)) # site(self0,self1,down0,down2)
        site = np.transpose(site, (0,2,1,3)) # site(self0,down0,self1,down2)
        site = np.reshape(site, (site.shape[0]*site.shape[1], site.shape[2], site.shape[3]))
        mpo.append(site)

        return mpo
    
    def sub(self, mps2, **kwargs):
        return self.add(mps2, subtract=True, **kwargs)
    
    def __sub__(self, other):
        return self.sub(other, compress=True) #change to True when roudning is added 
    
    def add(self, mpo2, subtract=False, compress=False, stop=Cutoff(1e-14)):
        mpo = []

        # Left boundary
        assert(self[0].shape[0] == mpo2[0].shape[0])
        assert(self[0].shape[2] == mpo2[0].shape[2])

        m1 = self[0].shape[1]
        m2 = mpo2[0].shape[1]
        new_left_boundary = np.zeros((self[0].shape[0], m1 + m2, self[0].shape[2]), dtype="complex")
        new_left_boundary[:, 0:m1, :] = self[0]
        new_left_boundary[:, m1:, :] = mpo2[0]
        mpo.append(new_left_boundary)

        # Core tensors
        
        for i in range(1, self.N - 1):
            assert(self[i].shape[1] == mpo2[i].shape[1])
            assert(self[i].shape[3] == mpo2[i].shape[3])

            m1 = self[i].shape[0]
            n1 = self[i].shape[2]
            m2 = mpo2[i].shape[0]
            n2 = mpo2[i].shape[2]
            new_core = np.zeros((m1 + m2, self[i].shape[1], n1 + n2, self[i].shape[3]), dtype="complex")
            new_core[0:m1, :, 0:n1, :] = self[i]
            new_core[m1:, :, n1:, :] = mpo2[i]
            mpo.append(new_core)

        assert(self[-1].shape[1] == mpo2[-1].shape[1])
        assert(self[-1].shape[2] == mpo2[-1].shape[2])

        m1 = self[-1].shape[0]
        m2 = mpo2[-1].shape[0]
        new_right_boundary = np.zeros((m1 + m2, self[-1].shape[1], self[-1].shape[2]), dtype="complex")
        new_right_boundary[0:m1, :, :] = self[-1]
        new_right_boundary[m1:, :, :] = (-1 if subtract else 1) * mpo2[-1]
        mpo.append(new_right_boundary)

        mpo = MPO(mpo)
        if compress:
            mpo.round(stop=stop)

        return mpo
    
    def round(self, stop=Cutoff(1e-14)):
        if self.rounded:
            if self.canform == "Right":
                return
            else:
                self.canonize_right()
                return
        
        self.canonize_left()
        
        U, S, Vt = truncated_svd(self[0].reshape(self[0].shape[0] * self[0].shape[2], -1), stop=stop)
        self[0] = U.reshape(self[0].shape[0], U.shape[-1], self[0].shape[2])
        
        for i in range(1, self.N - 1):
            K = np.diag(S) @ Vt
            A = np.einsum("kD,DdEl->kdEl", K, self[i])
            U, S, Vt = truncated_svd(np.reshape(A, (A.shape[0] * A.shape[1] * A.shape[3], A.shape[2])), stop=stop)
            self[i] = np.reshape(U, (A.shape[0], A.shape[1], U.shape[-1], A.shape[3]))
            
        K = np.diag(S) @ Vt
        self[-1] = np.einsum("kD,Ddl->kdl", K, self[-1])
        
        self.canform = "Right"
        self.rounded = True
        
    def __add__(self, other):
        return self.add(other, compress = True)
    
    def is_right_orth(self):
        # Right to left sweep
        for i in reversed(range(1, self.N)):
            tensor = self[i]
            if i == self.N-1 :
                reshaped_tensor = tensor.reshape(tensor.shape[0],tensor.shape[1]*tensor.shape[2])
            else:
                reshaped_tensor = tensor.reshape(tensor.shape[0],tensor.shape[1]*tensor.shape[3]*tensor.shape[2])
            product = np.dot(reshaped_tensor, reshaped_tensor.conj().T)
            identity = np.eye(product.shape[0])
            
            if not np.allclose(product, identity):
                print(f"Tensor at index {i} is not right-orthogonal.")
                # Visualize the product matrix as a heatmap
                plt.imshow(np.abs(product), cmap='mako', interpolation='nearest')
                plt.title(f"Heatmap of Product Matrix at tensor {i}")
                plt.colorbar()
                plt.show()
                assert False, "Orthogonality test failed" 
        print("All tensors are right-orthogonal.")
        
    def is_left_orth(self):
        # Left to right sweep
        for i in range(0,self.N-1):
            tensor = self[i]
            if i == 0 :
                reshaped_tensor = tensor.reshape(tensor.shape[0]*tensor.shape[2],tensor.shape[1])
            else:
                reshaped_tensor = tensor.reshape(tensor.shape[0]*tensor.shape[1]*tensor.shape[3],tensor.shape[2])
            product = np.dot(reshaped_tensor.conj().T, reshaped_tensor)
            identity = np.eye(product.shape[0])
            if not np.allclose(product, identity):
                print(f"Tensor at index {i} is not left-orthogonal.")
                plt.imshow(np.abs(product), cmap='mako', interpolation='nearest')
                plt.title(f"Heatmap of Product Matrix at tensor {i}")
                plt.colorbar()
                plt.show()
                assert False, "Orthogonality test failed"
        print("All tensors are left-orthogonal.")  
        
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
    
    def show(self, max_width=None):
        from .misc import print_multi_line
        l1 = ""
        l2 = ""
        l3 = ""
        for i in range(0,self.N - 1):
            bdim = self.bond_size(i)
            strl = len(str(bdim))
            l1 += f"│{bdim}"
            l2 += "●" + ("─" if bdim < 100 else "━") * strl
            l3 += "│" + " " * strl

        l1 += "│"
        l2 += "●"
        l3 += "│"
    
        print_multi_line(l1, l2,l3, max_width=max_width) 

    __matmul__ = __mul__
    
    @staticmethod
    def random_mpo(n, m, d=2, d2=2, dtype=float, random_tensor=np.random.randn):
        def random_tensor_with_dtype(*args, dtype=dtype):
            output = random_tensor(*args).astype(dtype)
            return output

        return MPO([random_tensor_with_dtype(d, m, d2)] + 
                   [random_tensor_with_dtype(m, d, m, d2) for _ in range(n - 2)] + 
                   [random_tensor_with_dtype(m, d, d2)])

    @staticmethod
    def fully_random_mpo(n=10, random_tensor=np.random.randn):
        randint = lambda: np.random.randint(1, 7)
        bond_dim = randint()
        sites = [random_tensor(randint(), bond_dim, randint())]
        for i in range(n-2):
            new_bond_dim = randint()
            sites.append(random_tensor(bond_dim, randint(), new_bond_dim, randint()))
            bond_dim = new_bond_dim
        sites.append(random_tensor(bond_dim, randint(), randint()))
        return MPO(sites)
    
    @staticmethod
    def identity(d, N, m=None):
            """Constructs an identity MPO for a system with local dimension d and N sites."""
            if m is None:
                m = d  
            tensors = []

            # Leftmost tensor (d x m x d)
            left_tensor = np.zeros((d, m, d))
            for i in range(d):
                left_tensor[i, 0, i] = 1.0  
            tensors.append(left_tensor)
            
            # Middle tensors (m x d x m x d)
            for _ in range(1, N - 1):
                middle_tensor = np.zeros((m, d, m, d))
                for i in range(d):
                    for j in range(m):
                        middle_tensor[j, i, j, i] = 1.0
                tensors.append(middle_tensor)
            
            # Rightmost tensor (m x d x d)
            right_tensor = np.zeros((m, d, d))
            for i in range(d):
                right_tensor[0, i, i] = 1.0  
            tensors.append(right_tensor)

            return MPO(tensors)
    
    @staticmethod
    def random_incremental_mpo(N, d, seed, d2=None):
        if d2 is None:
            d2 = d
        mpo = [np.random.randn(d, seed, d2)]  # Initialize with the first tensor, bond dimension 1 for the first tensor
        for i in range(0, N-2):
            mpo.append(np.random.randn(seed+i,d,seed+i+1,d2))
        mpo.append(np.random.randn(seed+i+1, d,d2))

        return MPO(mpo)
