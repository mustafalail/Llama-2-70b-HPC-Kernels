# ===================================================================================================================
#               numpy Programming Model
# ===================================================================================================================

# Prompt: "Create 3 code suggestions using the following parameters: Python numpy general matrix-matrix multiply def."

# SUGGESTION 1 # ----------------------------------

import numpy as np

def matrix_multiply(A, B):
    return np.dot(A, B)


# SUGGESTION 2 # ----------------------------------

import numpy as np

def matrix_multiply(A, B):
    return np.lib.arraysetops.matmat(A, B)


# SUGGESTION 3 # ----------------------------------

import numpy as np

def matrix_multiply(A, B):
    result = np.zeros((A.shape[0], B.shape[1]))
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for k in range(A.shape[1]):
                result[i, j] += A[i, k] * B[k, j]
    return result

# ===================================================================================================================
#               cuPy Programming Model
# ===================================================================================================================

# Prompt: "Create 3 code suggestions using the following parameters: Python cuPy general matrix-matrix multiply def."

# SUGGESTION 1 # ----------------------------------

import cupy

def matrix_multiply(A, B):
    return cupy.mm(A, B)


# SUGGESTION 2 # ----------------------------------

import cupy

def matrix_multiply(A, B):
    return cupy.mm(A, B, stride=1)


# SUGGESTION 3 # ----------------------------------

import cupy

def matrix_multiply(A, B):
    block_size = 16
    num_blocks = (A.shape[0] + block_size - 1) // block_size
    result = cupy.zeros((A.shape[0], B.shape[1]), dtype=A.dtype)
    for i in range(num_blocks):
        for j in range(num_blocks):
            block_a = A[i*block_size:(i+1)*block_size, :]
            block_b = B[:, j*block_size:(j+1)*block_size]
            result[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = \
                cupy.mm(block_a, block_b)
    return result

# ===================================================================================================================
#               pyCUDA Programming Model
# ===================================================================================================================

# Prompt: "Create 3 code suggestions using the following parameters: Python pyCUDA general matrix-matrix multiply def."

# SUGGESTION 1 # ----------------------------------

import pycuda.autoinit
from pycuda import driver

def matrix_multiply(A, B):
    # Create GPU buffers for the matrices
    A_buf = driver.create_buffer(A.shape, A.dtype)
    B_buf = driver.create_buffer(B.shape, B.dtype)

    # Copy the data from the CPU arrays to the GPU buffers
    driver.copy_to_device(A, A_buf)
    driver.copy_to_device(B, B_buf)

    # Define the kernel function
    @driver.register_kernel
    def matmul(A, B):
        # Perform matrix multiplication
        C = A @ B
        return C

    # Execute the kernel on the GPU
    C_buf = driver.empty_like(A_buf)
    matmul[A_buf.shape[0], A_buf.shape[1], B_buf.shape[1]](A_buf, B_buf, C_buf)

    # Copy the result from the GPU buffer to a CPU array
    C = driver.copy_to_host(C_buf)

    return C


# SUGGESTION 2 # ----------------------------------

import pycuda.autoinit
from pycuda import driver

def matrix_multiply(A, B):
    # Create a shared memory pool
    smem = driver.shared_memory_pool(A.shape[0], B.shape[1])

    # Allocate space for the matrices in the shared memory pool
    A_smem = smem.allocate(A.shape[0], A.shape[1])
    B_smem = smem.allocate(B.shape[0], B.shape[1])

    # Copy the data from the CPU arrays to the shared memory
    driver.copy_to_device(A, A_smem)
    driver.copy_to_device(B, B_smem)

    # Define the kernel function
    @driver.register_kernel
    def matmul(A, B):
        # Perform matrix multiplication using shared memory
        C = A @ B
        return C

    # Execute the kernel on the GPU
    C_smem = smem.allocate(A.shape[0], B.shape[1])
    matmul[A_smem.shape[0], A_smem.shape[1], B_smem.shape[1]](A_smem, B_smem, C_smem)

    # Copy the result from the shared memory to a CPU array
    C = driver.copy_to_host(C_smem)

    return C


# SUGGESTION 3 # ----------------------------------

import pycuda.autoinit
from pycuda import driver

def matrix_multiply(A, B):
    # Create a shared memory pool
    smem = driver.shared_memory_pool()

    # Allocate space for the matrices in the shared memory pool dynamically
    A_smem = smem.dynamic_allocate(A.shape[0], A.shape[1])
    B_smem = smem.dynamic_allocate(B.shape[0], B.shape[1])

    # Copy the data from the CPU arrays to the shared memory
    driver.copy_to_device(A, A_smem)
    driver.copy_to_device(B, B_smem)

    # Define the kernel function
    @driver.register_kernel
    def matmul(A, B):
        # Perform matrix multiplication using shared memory
        C = A @ B
        return C

    # Execute the kernel on the GPU
    C_smem = smem.dynamic_allocate(A.shape[0], B.shape[1])
    matmul[A_smem.shape[0], A_smem.shape[1], B_smem.shape[1]](A_smem, B_smem, C_smem)

    # Copy the result from the shared memory to a CPU array
    C = driver.copy_to_host(C_smem)

    return C

# ===================================================================================================================
#               Numba Programming Model
# ===================================================================================================================

# Prompt: "Create 3 code suggestions using the following parameters: Python Numba general matrix-matrix multiply def."

# SUGGESTION 1 # ----------------------------------

import numba

@numba.jit
def matrix_multiply(A, B):
    """
    Multiplies two matrices A and B and returns the result.
    """
    n = A.shape[0]
    m = B.shape[1]
    C = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            for k in range(A.shape[1]):
                C[i, j] += A[i, k] * B[k, j]
    return C


# SUGGESTION 2 # ----------------------------------

import numba
import numpy as np

@numba.jit
def strided_matrix_multiply(A, B, stride_a, stride_b):
    """
    Multiplies two matrices A and B with strides stride_a and stride_b.
    """
    n = A.shape[0]
    m = B.shape[1]
    C = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            for k in range(A.shape[1]):
                C[i, j] += A[i*stride_a, k] * B[k*stride_b, j]
    return C


# SUGGESTION 3 # ----------------------------------

import numba
import numpy as np

@numba.jit
def matrix_multiply_broadcast(A, B, broadcast_dim=None):
    """
    Multiplies two matrices A and B and returns the result.
    If broadcast_dim is None, it will automatically determine the broadcast dimension.
    """
    if broadcast_dim is None:
        broadcast_dim = max(A.ndim, B.ndim) - 1
    
    C = np.zeros((A.shape[:broadcast_dim] + B.shape[broadcast_dim:]))
    for i in range(A.shape[0]):
        for j in range(B.shape[0]):
            for k in range(A.shape[1]):
                C[i, j, k] = A[i, k] * B[k, j]
    return C

# ------------------------------------------------------------------------------------------------------------------------------------------------------
