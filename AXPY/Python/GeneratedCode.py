# numpy

# Prompt: "Give me code for the following:
# Python, numpy, AXPY, def"

# VERSION 1 #

import numpy as np

# Define a function for the matrix multiplication
def matrix_mult(A, B):
    # Perform the matrix multiplication using AXPY
    C = np.zeros((A.shape[0], B.shape[1]))
    np.axpy(C, A @ B)
    return C

# Test the function with some sample matrices
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(matrix_mult(A, B))

# VERSION 2 #

# VERSION 3 #


# ----------------------------------------------------------------------------------------------------------------------------------------

# cuPy

# Prompt: "Give me code for the following:
# Python, cuPy, AXPY, def"

# VERSION 1 #

import cublas as cuda
from cupy import zeros

# Define a function for the matrix multiplication
def matrix_mult(A, B):
    # Move the inputs to the GPU
    dA = cuda.to_device(A)
    dB = cuda.to_device(B)
    
    # Allocate memory on the GPU for the result
    dC = zeros((A.shape[0], B.shape[1]), dtype=float32)
    
    # Perform the matrix multiplication using AXPY
    cuda.axpy(dC, dA, dB, alpha=1.0)
    
    # Move the result back to the CPU
    C = dC.get()
    
    return C

# Test the function with some sample matrices
A = np.random.rand(2, 2)
B = np.random.rand(2, 2)
C = matrix_mult(A, B)
print(C)

# VERSION 2 #

# VERSION 3 #


# ----------------------------------------------------------------------------------------------------------------------------------------

# pyCUDA

# Prompt: "Give me code for the following:
# Python, pyCUDA, AXPY, def"

# VERSION 1 #

import pycuda.driver as cuda
from pycuda.compiler import SourceModule

# Define a function for the matrix multiplication
def matrix_mult(A, B):
    # Create a module for the kernel
    mod = SourceModule("""
        __global__ void axpy(float *A, float *B, float *C, int N) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < N) {
                C[i] = A[i] * B[i];
            }
        }
    """)

    # Load the module onto the device
    func = mod.get_function("axpy")

    # Move the inputs to the device
    dA = cuda.to_device(A)
    dB = cuda.to_device(B)

    # Allocate memory on the device for the result
    dC = cuda.malloc(A.size * sizeof(float))

    # Launch the kernel
    blocks = (A.size // 32) + 1
    threads = 32
    func(dA, dB, dC, A.size, blocks, threads)

    # Move the result back to the host
    C = dC.get()

    return C

# Test the function with some sample matrices
A = np.random.rand(2, 2)
B = np.random.rand(2, 2)
C = matrix_mult(A, B)
print(C)

# VERSION 2 #

# VERSION 3 #


# ----------------------------------------------------------------------------------------------------------------------------------------

# Numba

# Prompt: "Give me code for the following:
# Python, Numba, AXPY, def"

# VERSION 1 #

import numpy as np
from numba import jit, cuda

@jit(cuda)
def matrix_mult(A, B):
    # Perform axis-wise multiplication
    C = cuda. Axpy(A, B, axis=1)
    return C

# Test the function with some sample matrices
A = np.random.rand(2, 2)
B = np.random.rand(2, 2)
C = matrix_mult(A, B)
print(C)

# VERSION 2 #

# VERSION 3 #





