# ===================================================================================================================
#               Threads Programming Model
# ===================================================================================================================

# Prompt: "Create 3 code suggestions using the following parameters: Julia Threads general matrix-vector multiply."

# SUGGESTION 1 # ----------------------------------

using Threads

# Create a matrix and vector
A = Matrix(rand(10, 10))
b = Vector(rand(10))

# Perform matrix-vector multiplication using threads
@threads for i in 1:10
    A[i, :] * b
end


# SUGGESTION 2 # ----------------------------------

using Parallel

# Create a matrix and vector
A = Matrix(rand(10, 10))
b = Vector(rand(10))

# Perform matrix-vector multiplication using parallel blocks
@parallel for i in 1:10
    A[i, :] * b
end


# SUGGESTION 3 # ----------------------------------

using Distributed

# Create a matrix and vector
A = Matrix(rand(10, 10))
b = Vector(rand(10))

# Split the matrix and vector into smaller chunks
A_chunks = split(A, (5, 5))
b_chunks = split(b, (5,))

# Perform matrix-vector multiplication in parallel on multiple processes
@distributed for chunk in A_chunks
    @distributed for j in 1:length(chunk)
        chunk[j, :] * b_chunks[j]
    end
end



# ===================================================================================================================
#               CUDA Programming Model
# ===================================================================================================================

# Prompt: "Create 3 code suggestions using the following parameters: Julia CUDA general matrix-vector multiply."

# SUGGESTION 1 # ----------------------------------

using CUDA

# Define the matrix and vector
A = CuMatrix(rand(Float32, 10, 10))
b = CuVector(rand(Float32, 10))

# Perform matrix-vector multiplication
C = A * b


# SUGGESTION 2 # ----------------------------------

using CUDA

# Define the matrix and vector with strides
A = CuMatrix(rand(Float32, 10, 10), stride=2)
b = CuVector(rand(Float32, 10), stride=2)

# Perform matrix-vector multiplication with strides
C = A[:,:] * b[::2]


# SUGGESTION 3 # ----------------------------------

using CUDA

# Define the matrix and vector
A = CuMatrix(rand(Float32, 10, 10))
b = CuVector(rand(Float32, 10))

# Allocate shared memory for the matrix-vector product
smem = CuSharedMemory( Float32, 10 )

# Perform matrix-vector multiplication using shared memory
for i in 1:10
    # Load elements from global memory to shared memory
    smem[i] = A[i,:] * b[i]
end

# Store the result in a new CuMatrix object
C = smem[:]



# ===================================================================================================================
#               AMDGPU Programming Model
# ===================================================================================================================

# Prompt: "Create 3 code suggestions using the following parameters: Julia AMDGPU general matrix-vector multiply."

# SUGGESTION 1 # ----------------------------------

using AMDGPU

# Define the matrix and vector
A = AMDGPU.zeros((10, 10), Float32)
b = AMDGPU.zeros((10,), Float32)

# Perform matrix-vector multiplication
C = A * b


# SUGGESTION 2 # ----------------------------------

using AMDGPU

# Define the matrix and vector with strides
A = AMDGPU.zeros((10, 10), Float32, stride=2)
b = AMDGPU.zeros((10,), Float32, stride=2)

# Perform matrix-vector multiplication with strides
C = A[:,:] * b[::2]


# SUGGESTION 3 # ----------------------------------

using AMDGPU

# Define the matrix and vector
A = AMDGPU.zeros((10, 10), Float32)
b = AMDGPU.zeros((10,), Float32)

# Allocate shared memory for the matrix-vector product
smem = AMDGPU.shared_memory( Float32, 10 )

# Perform matrix-vector multiplication using shared memory
for i in 1:10
    # Load elements from global memory to shared memory
    smem[i] = A[i,:] * b[i]
end

# Store the result in a new AMDGPU array
C = smem[:]



# ===================================================================================================================
#               KernelAbstractions Programming Model
# ===================================================================================================================

# Prompt: "Create 3 code suggestions using the following parameters: Julia KernelAbstractions general matrix-vector multiply."

# SUGGESTION 1 # ----------------------------------

using KernelAbstractions

# Define the matrix and vector
A = KernelAbstractions.Matrix(undef, (10, 10), Float32)
b = KernelAbstractions.Vector(undef, (10,), Float32)

# Perform matrix-vector multiplication
C = A * b


# SUGGESTION 2 # ----------------------------------

using KernelAbstractions

# Define the matrix and vector with strides
A = KernelAbstractions.Matrix(undef, (10, 10), Float32, stride=2)
b = KernelAbstractions.Vector(undef, (10,), Float32, stride=2)

# Perform matrix-vector multiplication with strides
C = A[:,:] * b[::2]


# SUGGESTION 3 # ----------------------------------

using KernelAbstractions

# Define the matrix and vector
A = KernelAbstractions.Matrix(undef, (10, 10), Float32)
b = KernelAbstractions.Vector(undef, (10,), Float32)

# Allocate shared memory for the matrix-vector product
smem = KernelAbstractions.shared_memory( Float32, 10 )

# Perform matrix-vector multiplication using shared memory
for i in 1:10
    # Load elements from global memory to shared memory
    smem[i] = A[i,:] * b[i]
end

# Store the result in a new kernel abstraction object
C = smem[:]



# ---------------------------------------------------------------------------------------------------------------------------------------------------------


