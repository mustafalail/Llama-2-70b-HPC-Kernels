# ===================================================================================================================
#               Threads Programming Model
# ===================================================================================================================

# Prompt: "Create 3 code suggestions using the following parameters: Julia Threads general matrix-matrix multiply."

# SUGGESTION 1 # ----------------------------------

using Threads

# Create two matrices
A = Matrix(rand(Float64, 10, 10))
B = Matrix(rand(Float64, 10, 10))

# Define a function for matrix multiplication
function matmult(A, B)
    return A * B
end

# Split the matrices into smaller chunks
n = 2
chunksize = (size(A, 1) + n - 1) / n
A_chunks = collect(nothing, i -> view(A, :, i .* chunksize .+ (i == div(n, 2) ? 1 : 0)))
B_chunks = collect(nothing, i -> view(B, :, i .* chunksize .+ (i == div(n, 2) ? 1 : 0)))

# Perform matrix multiplication in parallel
@threads for i in 1:n for j in 1:n
    local chunk_a = A_chunks[i]
    local chunk_b = B_chunks[j]
    local result = matmult(chunk_a, chunk_b)
    # Write the result back to the main array
    view(A, :, (i - 1) * chunksize .+ 1 : min(i * chunksize, size(A, 1))) .= result
end


# SUGGESTION 2 # ----------------------------------

using Distributed

# Create two matrices
A = Matrix(rand(Float64, 10, 10))
B = Matrix(rand(Float64, 10, 10))

# Define a function for matrix multiplication
function matmult(A, B)
    return A * B
end

# Split the matrices into smaller chunks
n = 2
chunksize = (size(A, 1) + n - 1) / n
A_chunks = collect(nothing, i -> view(A, :, i .* chunksize .+ (i == div(n, 2) ? 1 : 0)))
B_chunks = collect(nothing, i -> view(B, :, i .* chunksize .+ (i == div(n, 2) ? 1 : 0)))

# Create a distributed array to store the results
result = Dict()

# Perform matrix multiplication in parallel
for i in 1:n for j in 1:n
    local chunk_a = A_chunks[i]
    local chunk_b = B_chunks[j]
    local result = matmult(chunk_a, chunk_b)
    # Store the result in the distributed array
    result[(i - 1) * chunksize .+ 1 : min(i * chunksize, size(A, 1))] = result
end

# Collect the results from the distributed array
final_result = collect(result)


# SUGGESTION 3 # ----------------------------------

using Parallel

# Create two matrices
A = Matrix(rand(Float64, 10, 10))
B = Matrix(rand(Float64, 10, 10))

# Define a function for matrix multiplication
function matmult(A, B)
    return A * B
end

# Split the matrices into smaller chunks
n = 2
chunksize = (size(A, 1) + n - 1) / n
A_chunks = collect(nothing, i -> view(A, :, i .* chunksize .+ (i == div(n, 2) ? 1 : 0)))
B_chunks = collect(nothing, i -> view(B, :, i .* chunksize .+ (i == div(n, 2) ? 1 : 0)))

# Perform matrix multiplication in parallel
pmap(matmult, (A_chunks, B_chunks), axis=1) do (chunk_a, chunk_b)
    return chunk_a * chunk_b
end

# Convert the result back to a Matrix
final_result = convert(Matrix, final_result)



# ===================================================================================================================
#               CUDA Programming Model
# ===================================================================================================================

# Prompt: "Create 3 code suggestions using the following parameters: Julia CUDA general matrix-matrix multiply."

# SUGGESTION 1 # ----------------------------------

using CUDA

# Define the matrices
A = CuMatrix(rand(Float32, 10, 10))
B = CuMatrix(rand(Float32, 10, 10))

# Perform matrix-matrix multiplication
C = A * B

# Print the result
println(C)


# SUGGESTION 2 # ----------------------------------

using CUDA

# Define the matrices
A = CuMatrix(rand(Float32, 10, 10))
B = CuMatrix(rand(Float32, 10, 10))

# Define strides for A and B
stride_a = (2, 3)
stride_b = (3, 2)

# Perform strided matrix-matrix multiplication
C = A[:, stride_a] * B[stride_b, :]

# Print the result
println(C)


# SUGGESTION 3 # ----------------------------------

using CUDA

# Define the matrices
A = CuMatrix(rand(Float32, 10, 10))
B = CuMatrix(rand(Float32, 10, 10))

# Define shared memory blocks
block_size = 32
shared_memory = CuArray{Float32}(undef, block_size, block_size)

# Perform shared memory matrix-matrix multiplication
@cuda threads per block = block_size
syncthreads()

for i = 1:div(size(A, 1), block_size)
    for j = 1:div(size(A, 2), block_size)
        @shared begin
            local A_local = A[i:min(i + block_size - 1, size(A, 1)), j:min(j + block_size - 1, size(A, 2))]
            local B_local = B[i:min(i + block_size - 1, size(B, 1)), j:min(j + block_size - 1, size(B, 2))]
            local C_local = zeros(Float32, block_size, block_size)
            for k = 1:min(block_size, size(A, 1))
                for l = 1:min(block_size, size(A, 2))
                    C_local[k, l] = dot(A_local[k, l], B_local[l, k])
                end
            end
            syncthreads()
            if ThreadsPerBlock() > 1
                shared_memory[threadIdx().y * block_size + threadIdx().x] = C_local
            end
        end
    end
end

# Copy the result from shared memory to global memory
C = Array( Float32, size(A, 1), size(A, 2) )
copy!(C, shared_memory)

# Print the result
println(C)



# ===================================================================================================================
#               AMDGPU Programming Model
# ===================================================================================================================

# Prompt: "Create 3 code suggestions using the following parameters: Julia AMDGPU general matrix-matrix multiply."

# SUGGESTION 1 # ----------------------------------

using AMDGPU

# Define the matrices
A = AMDGPU.Matrix{Float32, 2}(rand(10, 10), rand(10, 10))
B = AMDGPU.Matrix{Float32, 2}(rand(10, 10), rand(10, 10))

# Perform matrix-matrix multiplication
C = A * B

# Print the result
println(C)


# SUGGESTION 2 # ----------------------------------

using AMDGPU

# Define the matrices
A = AMDGPU.Matrix{Float32, 2}(rand(10, 10), rand(10, 10))
B = AMDGPU.Matrix{Float32, 2}(rand(10, 10), rand(10, 10))

# Define the strides
strideA = (2, 3)
strideB = (3, 2)

# Perform strided matrix-matrix multiplication
C = A * B'::StridedMatrixMultiplication(strideA, strideB)

# Print the result
println(C)


# SUGGESTION 3 # ----------------------------------

using AMDGPU

# Define the matrices
A = AMDGPU.Matrix{Float32, 2}(rand(10, 10), rand(10, 10))
B = AMDGPU.Matrix{Float32, 2}(rand(10, 10), rand(10, 10))

# Define the broadcasting shape
broadcast_shape = (10, 10, 1, 1)

# Perform matrix-matrix multiplication with broadcasting
C = A * B[:, :, 1, 1]::BroadcastedMatrixMultiplication(broadcast_shape)

# Print the result
println(C)



# ===================================================================================================================
#               KernelAbstractions Programming Model
# ===================================================================================================================

# Prompt: "Create 3 code suggestions using the following parameters: Julia KernelAbstractions general matrix-matrix multiply."

# SUGGESTION 1 # ----------------------------------

using KernelAbstractions

# Define the matrices
A = KernelAbstractions.Matrix{Float64, 2}(rand(10, 10), rand(10, 10))
B = KernelAbstractions.Matrix{Float64, 2}(rand(10, 10), rand(10, 10))

# Perform matrix-matrix multiplication
C = A * B

# Print the result
println(C)


# SUGGESTION 2 # ----------------------------------

using KernelAbstractions

# Define the matrices
A = KernelAbstractions.Matrix{Float64, 2}(rand(10, 10), rand(10, 10))
B = KernelAbstractions.Matrix{Float64, 2}(rand(10, 10), rand(10, 10))

# Define the strides
strideA = (2, 3)
strideB = (3, 2)

# Perform strided matrix-matrix multiplication
C = A * B'::StridedMatrixMultiplication(strideA, strideB)

# Print the result
println(C)


# SUGGESTION 3 # ----------------------------------

using KernelAbstractions

# Define the matrices
A = KernelAbstractions.Matrix{Float64, 2}(rand(10, 10), rand(10, 10))
B = KernelAbstractions.Matrix{Float64, 2}(rand(10, 10), rand(10, 10))

# Define the broadcasting shape
broadcast_shape = (10, 10, 1, 1)

# Perform matrix-matrix multiplication with broadcasting
C = A * B[:, :, 1, 1]::BroadcastedMatrixMultiplication(broadcast_shape)

# Print the result
println(C)



# ---------------------------------------------------------------------------------------------------------------------------------------------------------

