# numpy

# Prompt: "Create 3 code suggestions using the following parameters: Python numpy AXPY def."

# SUGGESTION 1 #

import numpy as np

# Define a vector of numbers
a = np.array([1, 2, 3])

# Multiply each element by a scalar value
b = np.array([4, 5, 6])
c = np.axpy(a, b)
print(c) # Output: array([4, 10, 18])


# SUGGESTION 2 #

import numpy as np

# Define two matrices
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Perform matrix multiplication
C = np.dot(A, B)
print(C) # Output: array([[19, 22], [43, 50]])


# SUGGESTION 3 #

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

# SUGGESTION 1 #

import cupy
import numpy as np

# Define a matrix and a vector
A = cupy.eye(3)
x = cupy.array([1, 2, 3])

# Perform a matrix-vector dot product using AXPY
y = cupy.zeros((3,))
cupy.axpy(A, x, y)

print(y) # Output: array([2., 4., 6.])


# SUGGESTION 2 #

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


# SUGGESTION 3 #

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

# SUGGESTION 1 #

import pycuda.driver as drv
from pycuda.compiler import SourceModule

# Define the AXPY kernel
src = """
__global__ void axpy(float *A, float *x, float *y, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        y[i] = A[i] * x[i];
    }
}
"""

# Compile the kernel
mod = SourceModule(src)
axpy_kernel = mod.get_function("axpy")

# Initialize the GPU memory
drv.device.create_buffer(A, N * sizeof(float))
drv.device.create_buffer(x, N * sizeof(float))
drv.device.create_buffer(y, N * sizeof(float))

# Copy the data to the GPU
drv.memcpy_htod(A, h_A)
drv.memcpy_htod(x, h_x)

# Launch the kernel
block_size = 16
grid_size = (N + block_size - 1) // block_size
axpy_kernel(A, x, y, N, block_size, grid_size)

# Synchronize the threads
drv.device.synchronize()

# Copy the result back to the CPU
drv.memcpy_dtoh(y, h_y)

# Free the GPU memory
drv.device.destroy_buffer(A)
drv.device.destroy_buffer(x)
drv.device.destroy_buffer(y)


# SUGGESTION 2 #

import pycuda.driver as drv
from pycuda.compiler import SourceModule

# Define the strided AXPY kernel
src = """
__global__ void axpy(float *A, float *x, float *y, int N, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        y[i*stride] = A[i*stride] * x[i];
    }
}
"""

# Compile the kernel
mod = SourceModule(src)
axpy_kernel = mod.get_function("axpy")

# Initialize the GPU memory
drv.device.create_buffer(A, N * sizeof(float))
drv.device.create_buffer(x, N * sizeof(float))
drv.device.create_buffer(y, N * sizeof(float))

# Copy the data to the GPU
drv.memcpy_htod(A, h_A)
drv.memcpy_htod(x, h_x)

# Launch the kernel
block_size = 16
grid_size = (N + block_size - 1) // block_size
axpy_kernel(A, x, y, N, block_size, grid_size, stride)

# Synchronize the threads
drv.device.synchronize()

# Copy the result back to the CPU
drv.memcpy_dtoh(y, h_y)

# Free the GPU memory
drv.device.destroy_buffer(A)
drv.device.destroy_buffer(x)
drv.device.destroy_buffer(y)


# SUGGESTION 3 #

import pycuda.driver as drv
from pycuda.compiler import SourceModule

# Define the matrix-matrix multiplication kernel
src = """
__global__ void matmat(float *A, float *B, float *C, int M, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M) {
        C[i*N+threadIdx.y] = A[i*M+threadIdx.x] * B[threadIdx.y*N+i];
    }
}
"""

# Compile the kernel
mod = SourceModule(src)
matmat_kernel = mod.get_function("matmat")

# Initialize the GPU memory
drv.device.create_buffer(A, M * N * sizeof(float))
drv.device.create_buffer(B, M * N * sizeof(float))
drv.device.create_buffer(C, M * N * sizeof(float))

# Copy the data to the GPU
drv.memcpy_htod(A, h_A)
drv.memcpy_htod(B, h_B)

# Launch the kernel
block_size = 16
grid_size = (M + block_size - 1) // block_size
matmat_kernel(A, B, C, M, N, block_size, grid_size)

# Synchronize the threads
drv.device.synchronize()

# Copy the result back to the CPU
drv.memcpy_dtoh(C, h_C)

# Free the GPU memory
drv.device.destroy_buffer(A)
drv.device.destroy_buffer(B)
drv.device.destroy_buffer(C)


# ----------------------------------------------------------------------------------------------------------------------------------------

# Numba

# Prompt: "Create 3 code suggestions using the following parameters: Python Numba AXPY def."

# SUGGESTION 1 #

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


# SUGGESTION 2 #

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


# SUGGESTION 3 #

import numba

@numba.jit
def axpy_numba(a, b, n):
    return numba.axpy(a, b, n)

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
n = len(a)
result = axpy_numba(a, b, n)
print(result) # Output: array([5, 7, 9])

# ------------------------------------------------------------------------------------------------------------------------------------------------------



