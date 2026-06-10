#/usr/bin/env python
import numpy as np
import scipy as sp
import sys
import os 

#Adjust as needed
# (vendored) do not override BLAS threads globally


#Adjust locally 
# (vendored) C++ build path removed; Python fallback is used

try:
    from libincrementalqr import setup, add_cols, extract_q, get_error_estimate
    libincrementalqr_available = True
except Exception as e:

    libincrementalqr_available = False

def complex_to_real(A):
    rows, cols = A.shape
    
    B = np.zeros((2*rows, 2*cols), dtype=float)
    
    real_A = np.real(A)
    imag_A = np.imag(A)
    
    B[0::2, 0::2] = real_A
    B[0::2, 1::2] = -imag_A
    B[1::2, 0::2] = imag_A
    B[1::2, 1::2] = real_A
    
    return B

def real_to_complex(B):
    rows, cols = B.shape
    assert rows % 2 == 0 and cols % 2 == 0, "Input matrix dimensions must be even."
    
    new_rows, new_cols = rows // 2, cols // 2
    
    real_part = B[0::2, 0::2]
    imag_part = B[1::2, 0::2]
    
    return real_part + 1j * imag_part

class IncrementalQR(object):

    def _libincrementalqr_available(self):
        return libincrementalqr_available and self.use_cpp_if_available

    def _complex_to_real(self):
        return self._libincrementalqr_available() and self.dtype == complex

    def __init__(self, data, size = None, use_cpp_if_available=True):
        self.use_cpp_if_available = use_cpp_if_available
        self.dtype = data.dtype

        self.m = data.shape[0]
        self.n = data.shape[1]
        assert(self.m >= self.n)
        if self._complex_to_real():
            self.m *= 2
            self.n *= 2

                    
        self.size = size
        if self.size is None:
            self.size = 2 * self.n
        assert(self.size >= self.n)

        if self._complex_to_real():
            self.data = np.zeros((self.m, self.size), order='F', dtype=float)
        else:
            self.data = np.zeros((self.m, self.size), order='F', dtype=self.dtype)
        if self._complex_to_real():
            self.data[:,:self.n] = complex_to_real(data)
        else:
            self.data[:,:self.n] = data
        self.tau = np.zeros(self.size, dtype=self.data.dtype)
        if self._libincrementalqr_available():
            setup(self.data, self.tau, self.m, self.n)
        else:
            self._setup()
        self.open = True

    def _setup(self):
        # QR factorization
        geqrf = sp.linalg.get_lapack_funcs('geqrf', dtype=self.data.dtype)
        self.data[:,:self.n], self.tau[:self.n], _, _ = geqrf(self.data[:,:self.n])

        # Invert upper triangular part
        trtri = sp.linalg.get_lapack_funcs('trtri', dtype=self.data.dtype)
        self.data[:self.n,:self.n] = trtri(self.data[:self.n,:self.n])[0]
        
    def _resize(self, new_size = None):
        if new_size == None:
            new_size = 2 * self.size
        assert(new_size > self.size)
        
        tmp = self.data
        self.data = np.zeros((self.m, new_size), order='F', dtype=self.data.dtype)
        self.data[:,:self.size] = tmp

        tmp = self.tau
        self.tau = np.zeros(new_size, dtype=self.data.dtype)
        self.tau[:self.size] = tmp

        self.size = new_size

    def append(self, new_data):
        assert(self.open)
        if self._complex_to_real():
            assert(self.m == 2*new_data.shape[0])
            k = 2*new_data.shape[1]
        else:
            assert(self.m == new_data.shape[0])
            k = new_data.shape[1]
        if self.n + k > self.size:
            assert(self.n + k <= self.m)
            self._resize(new_size=self.n + k)  # Ensure enough space

        if self._libincrementalqr_available():
            if self._complex_to_real():
                # print(self.data.shape, self.n, new_data.shape)
                self.data[:, self.n:self.n + k] = complex_to_real(new_data)
            else:
                self.data[:, self.n:self.n + k] = new_data    
            add_cols(self.data, self.tau, self.m, self.n, k)
        else:
            # Multiply new data by Q'
            ormqr = sp.linalg.get_lapack_funcs('ormqr', dtype=self.data.dtype)
            transpose = 'T' if np.isrealobj(self.data) else 'C'
            lwork = int(np.real(ormqr('L', transpose, self.data[:,:self.n], self.tau[:self.n], new_data, -1)[1][0]))
            self.data[:,self.n:self.n+k] = ormqr('L', transpose, self.data[:,:self.n], self.tau[:self.n], new_data, lwork)[0]

            # QR factorize bottom part of new data
            geqrf = sp.linalg.get_lapack_funcs('geqrf', dtype=self.data.dtype)
            self.data[self.n:,self.n:self.n+k], self.tau[self.n:self.n+k], _, _ = geqrf(self.data[self.n:,self.n:self.n+k])

            # Invert triangular part of new QR factor
            trtri = sp.linalg.get_lapack_funcs('trtri', dtype=self.data.dtype)
            self.data[self.n:self.n+k,self.n:self.n+k] = trtri(self.data[self.n:self.n+k,self.n:self.n+k])[0]

            # Set top left part of new data
            self.data[:self.n,self.n:self.n+k] = -np.triu(self.data[:self.n,:self.n]) @ self.data[:self.n,self.n:self.n+k] @ np.triu(self.data[self.n:self.n+k,self.n:self.n+k])

        self.n += k

    def error_estimate(self):
        assert(self.open)
        if self._libincrementalqr_available():
            return get_error_estimate(self.data, self.m, self.n)
        else:
            return np.sqrt(sum(np.linalg.norm(np.triu(self.data[:self.n,:self.n]), axis=1) ** -2) / self.n)
    
    def get_q(self):
        if self.open:
            if self._libincrementalqr_available():
                import time
                start = time.time()
                extract_q(self.data, self.tau, self.m, self.n)
                self.open = False
                if self._complex_to_real():
                    start = time.time()
                    output = real_to_complex(self.data[:, :self.n])
                    return output
                else:
                    return self.data[:, :self.n]
            else:
                orgqr = sp.linalg.get_lapack_funcs('orgqr', dtype=self.data.dtype)
                return orgqr(self.data[:,:self.n], self.tau[:self.n])[0]

if __name__ == "__main__":
    from time import time
    import numpy as np

    A = np.random.randn(10000, 200)
    B = np.random.randn(10000, 200)
    
    # C++ implementation
    print('C++ implementation')
    total_start = time()
    
    start = time()
    iqr = IncrementalQR(A)
    total = time() - start
    print("Initialize\t{}".format(total))

    start = time()
    iqr.append(B)
    total = time() - start
    print("Append\t{}".format(total))
    
    start = time()
    err_cpp = iqr.error_estimate()
    total = time() - start
    print("Error estimation\t{}".format(total))
    
    start = time()
    Q_cpp = iqr.get_q()
    total = time() - start
    print("Get Q\t{}".format(total))
    
    total_time_cpp = time() - total_start
    print("Total time for C++ implementation\t{}".format(total_time_cpp))
    print("")

    # Scipy implementation
    print('Scipy implementation')
    total_start = time()

    start = time()
    iqr = IncrementalQR(A, use_cpp_if_available=False)
    total = time() - start
    print("Initialize\t{}".format(total))

    start = time()
    iqr.append(B)
    total = time() - start
    print("Append\t{}".format(total))
    
    start = time()
    err_sp = iqr.error_estimate()
    total = time() - start
    print("Error estimation\t{}".format(total))
    
    start = time()
    Q_sp = iqr.get_q()
    total = time() - start
    print("Get Q\t{}".format(total))
    
    total_time_sp = time() - total_start
    print("Total time for Scipy implementation\t{}".format(total_time_sp))

    assert(np.allclose(Q_cpp, Q_sp))
    assert(np.allclose(err_cpp, err_sp))
    print("Accuracy checks passed!")
    print("")

    # C++ implementation (complex)
    print('C++ implementation (complex)')
    total_start = time()

    start = time()
    iqr = IncrementalQR(A.astype(np.cdouble))
    total = time() - start
    print("Initialize\t{}".format(total))

    start = time()
    iqr.append(B.astype(np.cdouble))
    total = time() - start
    print("Append\t{}".format(total))
    
    start = time()
    err_cpp_C = iqr.error_estimate()
    total = time() - start
    print("Error estimation\t{}".format(total))
    
    start = time()
    Q_cpp_C = iqr.get_q()
    total = time() - start
    print("Get Q\t{}".format(total))
    
    total_time_cpp_complex = time() - total_start
    print("Total time for C++ implementation (complex)\t{}".format(total_time_cpp_complex))
    
    assert(np.allclose(Q_cpp, Q_cpp_C))
    assert(np.allclose(err_cpp, err_cpp_C))
    print("Accuracy checks passed!")
    print("")

    # Scipy implementation (complex)
    print('Scipy implementation (complex)')
    total_start = time()

    start = time()
    iqr = IncrementalQR(A.astype(np.cdouble), use_cpp_if_available=False)
    total = time() - start
    print("Initialize\t{}".format(total))

    start = time()
    iqr.append(B.astype(np.cdouble))
    total = time() - start
    print("Append\t{}".format(total))
    
    start = time()
    err_sp = iqr.error_estimate()
    total = time() - start
    print("Error estimation\t{}".format(total))
    
    start = time()
    Q_sp = iqr.get_q()
    total = time() - start
    print("Get Q\t{}".format(total))
    
    total_time_sp_complex = time() - total_start
    print("Total time for Scipy implementation (complex)\t{}".format(total_time_sp_complex))

    assert(np.allclose(Q_cpp, Q_sp))
    assert(np.allclose(err_cpp, err_sp))
    print("Accuracy checks passed!")