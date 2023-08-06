# numpy

# Prompt: "Create 3 code suggestions using the following parameters: Python numpy AXPY def."

# VERSION 1 #

import numpy as np

# Define a vector of numbers
a = np.array([1, 2, 3])

# Multiply each element by a scalar value
b = np.array([4, 5, 6])
c = np.axpy(a, b)
print(c) # Output: array([4, 10, 18])


# VERSION 2 #

import numpy as np

# Define two matrices
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Perform matrix multiplication
C = np.dot(A, B)
print(C) # Output: array([[19, 22], [43, 50]])


# VERSION 3 #

import numpy as np

# Define two arrays with different shapes
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Perform element-wise multiplication
c = np.multiply(a, b)
print(c) # Output: array([4, 10, 18])


# ----------------------------------------------------------------------------------------------------------------------------------------

# cuPy

# Prompt: "Create 3 code suggestions using the following parameters: Python cuPy AXPY def."

# VERSION 1 #

import cupy
import numpy as np

# Define a matrix and a vector
A = cupy.eye(3)
x = cupy.array([1, 2, 3])

# Perform a matrix-vector dot product using AXPY
y = cupy.zeros((3,))
cupy.axpy(A, x, y)

print(y) # Output: array([2., 4., 6.])


# VERSION 2 #

import cupy
import numpy as np

# Define a batch of matrices and vectors
A = cupy.stack((cupy.eye(3), cupy.eye(3)))
x = cupy.stack((cupy.array([1, 2, 3]), cupy.array([4, 5, 6])))

# Perform a batched matrix-vector dot product using AXPY
y = cupy.zeros((2, 3))
cupy.axpy(A, x, y)

print(y) # Output: array([[2., 4., 6.],
                       [7., 10., 13.]])


# VERSION 3 #

import cupy
import numpy as np

# Define a custom matrix-vector dot product function using AXPY
def dot_product(A, x):
    y = cupy.zeros((A.shape[0],))
    cupy.axpy(A, x, y)
    return y

# Test the function
A = cupy.eye(3)
x = cupy.array([1, 2, 3])
y = dot_product(A, x)
print(y) # Output: array([2., 4., 6.])

# ----------------------------------------------------------------------------------------------------------------------------------------

# pyCUDA

# Prompt: "Create 3 code suggestions using the following parameters: Python pyCUDA AXPY def."

# VERSION 1 #

import pycuda.driver as cudadrv
from pycuda.compiler import SourceModule

# Define a kernel for the vector-matrix multiplication
mod = SourceModule("""
__global__ void axpy(float **A, float **x, float **y, int N, int M) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        for (int j = 0; j < M; ++j) {
            y[i][j] = A[i][j] * x[i][j];
        }
    }
}
 """)

# Load the kernel into the GPU
dev = cudadrv.Device()
func = mod.get_function("axpy")

# Allocate memory on the GPU for the inputs and outputs
A = cudadrv.to_device(numpy.random.rand(10, 10).astype(numpy.float32))
x = cudadrv.to_device(numpy.random.rand(10, 10).astype(numpy.float32))
y = cudadrv.to_device(numpy.zeros((10, 10)).astype(numpy.float32))

# Set the number of blocks and threads per block
block_size = 16
grid_size = (10 + block_size - 1) // block_size

# Launch the kernel
func(A, x, y, 10, 10, block_size, grid_size)

# Synchronize the threads
cudadrv.synchronize()

# Copy the result back to the CPU
result = y.copy_to_host()
print(result)


# VERSION 2 #

import pycuda.driver as cudadrv
from pycuda.compiler import SourceModule

# Define a kernel for the matrix-vector multiplication
mod = SourceModule("""
__global__ void axpy(float **A, float **x, float **y, int N, int M) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        for (int j = 0; j < M; ++j) {
            y[i][j] = A[i][j] * x[j];
        }
    }
}
 """)

# Load the kernel into the GPU
dev = cudadrv.Device()
func = mod.get_function("axpy")

# Allocate memory on the GPU for the inputs and outputs
A = cudadrv.to_device(numpy.random.rand(10, 10).astype(numpy.float32))
x = cudadrv.to_device(numpy.random.rand(10).astype(numpy.float32))
y = cudadrv.to_device(numpy.zeros((10, 10)).astype(numpy.float32))

# Set the number of blocks and threads per block
block_size = 16
grid_size = (10 + block_size - 1) // block_size

# Launch the kernel
func(A, x, y, 10, 10, block_size, grid_size)

# Synchronize the threads
cudadrv.synchronize()

# Copy the result back to the CPU
result = y.copy_to_host()
print(result)


# VERSION 3 #

# ONLY 2 CODES WERE PROVIDED #


# ----------------------------------------------------------------------------------------------------------------------------------------

# Numba

# Prompt: "Create 3 code suggestions using the following parameters: Python Numba AXPY def."

# VERSION 1 #

import numba

@numba.jit
def axpy_loop(a, b, n):
    for i in range(n):
        a[i] += b[i]

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
n = len(a)
axpy_loop(a, b, n)
print(a) # Output: array([5, 7, 9])


# VERSION 2 #

import numba
import numpy as np
from numba import cuda

@numba.jit
@cuda.jit
def axpy_gpu(a, b, n):
    dev = cuda.device()
    a_dev = cuda.to_device(a, dev)
    b_dev = cuda.to_device(b, dev)
    c_dev = cuda.empty_like(a_dev)
    @cuda.kernel
    def kernel(i, j):
        c_dev[i] = a_dev[i] + b_dev[j]
    cuda.parallel.forall(range(n), kernel, block=128)
    return c_dev.copy_to_host()

a = np.random.rand(1000)
b = np.random.rand(1000)
n = 1000
result = axpy_gpu(a, b, n)
print(result)


# VERSION 3 #

import numba

@numba.jit
def axpy_numba(a, b, n):
    return numba.axpy(a, b, n)

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
n = len(a)
result = axpy_numba(a, b, n)
print(result) # Output: array([5, 7, 9])





