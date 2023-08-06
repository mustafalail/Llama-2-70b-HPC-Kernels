
# numpy

# Prompt: "Create 3 code suggestions using the following parameters: Python numpy general matrix-vector multiply def."

# SUGGESTION 1 #

import numpy as np

def matrix_vector_multiply(matrix, vector):
    """
    Returns the dot product of a matrix and a vector.
    
    Parameters:
        matrix (numpy array): The matrix for the multiplication.
        vector (numpy array): The vector for the multiplication.
    
    Returns:
        numpy array: The result of the matrix-vector multiplication.
    """
    return np.dot(matrix, vector)


# SUGGESTION 2 #

import numpy as np

def dynamic_matrix_vector_multiply(matrix, vector, num_iterations=1000):
    """
    Returns the result of repeatedly multiplying a matrix and a vector.
    
    Parameters:
        matrix (numpy array): The matrix for the multiplication.
        vector (numpy array): The vector for the multiplication.
        num_iterations (int, optional): The number of times to repeat the multiplication. Defaults to 1000.
    
    Returns:
        numpy array: The result of the repeated matrix-vector multiplication.
    """
    result = np.zeros((num_iterations,))
    for i in range(num_iterations):
        result[i] = np.dot(matrix, vector)
    return result


# SUGGESTION 3 #

import numpy as np
from numba import jit

@jit
def matrix_vector_multiply(matrix, vector):
    """
    A faster version of the matrix-vector multiplication using NumBA JIT compiler.
    
    Parameters:
        matrix (numpy array): The matrix for the multiplication.
        vector (numpy array): The vector for the multiplication.
    
    Returns:
        numpy array: The result of the matrix-vector multiplication.
    """
    return np.dot(matrix, vector)

# ------------------------------------------------------------------------------------------------------------------------------------------------------

# cuPy

# Prompt: "Create 3 code suggestions using the following parameters: Python cuPy general matrix-vector multiply def."

# SUGGESTION 1 #

import cupy

def cupy_matrix_vector_multiply(A, b):
    """
    Perform matrix-vector multiplication using CuPy.
    
    Parameters:
        A (cupy.ndarray): The matrix for the multiplication.
        b (cupy.ndarray): The vector for the multiplication.
    
    Returns:
        cupy.ndarray: The result of the matrix-vector multiplication.
    """
    return cupy.matmul(A, b)


# SUGGESTION 2 #

import cupy

def batched_cuppy_matrix_vector_multiply(A, b, batch_size=16):
    """
    Perform batched matrix-vector multiplication using CuPy.
    
    Parameters:
        A (cupy.ndarray): The matrix for the multiplication.
        b (cupy.ndarray): The vector for the multiplication.
        batch_size (int, optional): The size of the batch. Defaults to 16.
    
    Returns:
        cupy.ndarray: The result of the batched matrix-vector multiplication.
    """
    # Split the inputs into batches
    A_batches = cupy.split(A, batch_size, axis=0)
    b_batches = cupy.split(b, batch_size, axis=0)
    
    # Perform batched matrix-vector multiplication
    results = []
    for batch_a, batch_b in zip(A_batches, b_batches):
        results.append(cupy.matmul(batch_a, batch_b))
    
    # Concatenate the results
    return cupy.concatenate(results)


# SUGGESTION 3 #

import cupy

def strided_cuppy_matrix_vector_multiply(A, b, stride=1):
    """
    Perform strided matrix-vector multiplication using CuPy.
    
    Parameters:
        A (cupy.ndarray): The matrix for the multiplication.
        b (cupy.ndarray): The vector for the multiplication.
        stride (int, optional): The stride of the matrix. Defaults to 1.
    
    Returns:
        cupy.ndarray: The result of the strided matrix-vector multiplication.
    """
    # Get the shape of the matrix and vector
    m, n = A.shape
    k = b.shape[1]
    
    # Create a strided version of the matrix
    A_strided = cupy.lib.stride_tricks.as_strided(A, shape=(m, n), strides=(stride, 1))
    
    # Perform strided matrix-vector multiplication
    return cupy.matmul(A_strided, b)

# ------------------------------------------------------------------------------------------------------------------------------------------------------

# pyCUDA

# Prompt: "Create 3 code suggestions using the following parameters: Python pyCUDA general matrix-vector multiply def."

# SUGGESTION 1 #

import pycuda.driver as cudadrv
from pycuda.compiler import SourceModule

# Define the kernel
mod = SourceModule("""
__global__ void matrix_vector_mult(float *A, float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = dot(A, B, i);
    }
}

__shared__ float dot(float *A, float *B, int i) {
    float sum = 0;
    for (int j = 0; j < N; ++j) {
        sum += A[i * N + j] * B[j];
    }
    return sum;
}
 """)

# Load the kernel
k = mod.get_function("matrix_vector_mult")

# Initialize data
N = 8
A = np.random.rand(N, N).astype(np.float32)
B = np.random.rand(N, 1).astype(np.float32)
C = np.zeros((N, 1), dtype=np.float32)
dA = cudadrv.to_device(A)
dB = cudadrv.to_device(B)
dC = cudadrv.to_device(C)

# Launch kernel
block_size = 16
grid_size = (N + block_size - 1) // block_size
k(dA, dB, dC, N, block_size, grid_size)

# Synchronize
cudadrv.synchronize()

# Copy result back to host
C = dC.copy_to_host()


# SUGGESTION 2 #

import pycuda.driver as cudadrv
from pycuda.compiler import SourceModule

# Define the kernel
mod = SourceModule("""
__global__ void matrix_vector_mult(float *A, float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = dot(A, B, i);
    }
}

__shared__ float dot(float *A, float *B, int i) {
    float sum = 0;
    for (int j = 0; j < N; ++j) {
        sum += A[i * N + j] * B[j];
    }
    return sum;
}
 """)

# Load the kernel
k = mod.get_function("matrix_vector_mult")

# Initialize data
N = 8
A = np.random.rand(N, N).astype(np.float32)
B = np.random.rand(N, 1).astype(np.float32)
C = np.zeros((N, 1), dtype=np.float32)
dA = cudadrv.to_device(A)
dB = cudadrv.to_device(B)
dC = cudadrv.to_device(C)

# Launch kernel
block_size = 16
grid_size = (N + block_size - 1) // block_size
k(dA, dB, dC, N, block_size, grid_size)

# Synchronize
cudadrv.synchronize()

# Copy result back to host
C = dC.copy_to_host()


# SUGGESTION 3 #

import pycuda.driver as cudadrv
from pycuda.compiler import SourceModule

# Define the kernel
mod = SourceModule("""
__global__ void matrix_vector_mult(float *A, float *B, float *C, int N) {
    __shared__ float smem[N];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        smem[i] = dot(A, B, i);
        __syncthreads();
        C[i] = smem[i];
    }
}

__shared__ float dot(float *A, float *B, int i) {
    float sum = 0;
    for (int j = 0; j < N; ++j) {
        sum += A[i * N + j] * B[j];
    }
    return sum;
}
 """)

# Load the kernel
k = mod.get_function("matrix_vector_mult")

# Initialize data
N = 8
A = np.random.rand(N, N).astype(np.float32)
B = np.random.rand(N, 1).astype(np.float32)
C = np.zeros((N, 1), dtype=np.float32)
dA = cudadrv.to_device(A)
dB = cudadrv.to_device(B)
dC = cudadrv.to_device(C)

# Launch kernel
block_size = 16
grid_size = (N + block_size - 1) // block_size
k(dA, dB, dC, N, block_size, grid_size)

# Synchronize
cudadrv.synchronize()

# Copy result back to host
C = dC.copy_to_host()

# ------------------------------------------------------------------------------------------------------------------------------------------------------

# Numba

# Prompt: "Create 3 code suggestions using the following parameters: Python Numba general matrix-vector multiply def."

# SUGGESTION 1 #

import numpy as np
from numba import jit

@jit(nopython=True)
def matvec_mul(A, b):
    """
    Matrix-vector multiplication.
    
    Parameters
    ----------
    A : ndarray
        Input matrix.
    b : ndarray
        Input vector.
        
    Returns
    -------
    out : ndarray
        Output vector.
    """
    m, n = A.shape
    out = np.zeros((m,))
    for i in range(m):
        for j in range(n):
            out[i] += A[i, j] * b[j]
    return out


# SUGGESTION 2 #

import numpy as np
from numba import jit

@jit(nopython=True)
def matvec_mul(As, bs):
    """
    Element-wise matrix-vector multiplication.
    
    Parameters
    ----------
    As : ndarray
        Array of input matrices.
    bs : ndarray
        Array of input vectors.
        
    Returns
    -------
    out : ndarray
        Array of output vectors.
    """
    m, n = As.shape
    out = np.zeros((len(As), n))
    for i in range(len(As)):
        for j in range(n):
            out[i, j] = np.sum(As[i] * bs[j])
    return out


# SUGGESTION 3 #

import numpy as np
from numba import jit, prange, parallel

@jit(nopython=True)
@parallel
def matvec_mul(As, bs):
    """
    Parallel element-wise matrix-vector multiplication.
    
    Parameters
    ----------
    As : ndarray
        Array of input matrices.
    bs : ndarray
        Array of input vectors.
        
    Returns
    -------
    out : ndarray
        Array of output vectors.
    """
    m, n = As.shape
    out = np.zeros((m, bs.size))

    # Split the computation into smaller chunks
    chunksize = 100
    nchunks = (bs.size + chunksize - 1) // chunksize

    @jit(nopython=True)
    def chunk_matvec_mul(A, b, out):
        """
        Computes the matrix-vector product for a single chunk.
        
        Parameters
        ----------
        A : ndarray
            Input matrix.
        b : ndarray
            Input vector.
        out : ndarray
            Output vector.
        """
        for i in prange(chunksize):
            for j in range(n):
                out[i, j] = np.dot(A[i], b[j])

    # Perform the computation in parallel
    with parallel():
        for chunk in range(nchunks):
            chunk_matvec_mul(As[:, chunk*chunksize:(chunk+1)*chunksize], bs[chunk*chunksize:(chunk+1)*chunksize], out[:, chunk*chunksize:(chunk+1)*chunksize])

    return out

# ------------------------------------------------------------------------------------------------------------------------------------------------------
