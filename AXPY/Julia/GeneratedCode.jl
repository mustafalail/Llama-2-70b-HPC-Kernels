
# ===================================================================================================================
#               Threads Programming Model
# ===================================================================================================================

# Prompt: "Create 3 code suggestions using the following parameters: Julia Threads AXPY."

# SUGGESTION 1 # ----------------------------------

using LinearAlgebra
using ThreadPools

# Define matrix dimensions
n = 1000
m = 1000
k = 1000

# Create matrices
A = rand(Float64, n, k)
B = rand(Float64, k, m)
C = similar(A, n, m)

# Define thread pool
pool = ThreadPool(4) # 4 threads

# Perform matrix multiplication using AXPY
@sync for i in 1:n
    for j in 1:m
        C[i, j] = dot(A[:, i], B[j, :])
end

# Print result
println("Matrix product: $(C)")


# SUGGESTION 2 # ----------------------------------
  
using LinearAlgebra
using ThreadPools

# Define matrix dimensions
n = 1000
p = 100

# Create matrices
A = rand(Float64, n, p)
b = rand(Float64, n)

# Define thread pool
pool = ThreadPool(4) # 4 threads

# Perform linear least squares using AXPY
x = zeros( Float64, p )
@sync for i in 1:n
    for j in 1:p
        x[j] += dot(A[:, i], b[i])
end

# Print solution
println("Least squares solution: $x")


# SUGGESTION 3 # ----------------------------------

using LinearAlgebra
using ThreadPools

# Define matrix dimension
n = 1000

# Create matrix
A =rand(Float64, n, n)

# Define thread pool
pool = ThreadPool(4) # 4 threads

# Perform eigenvalue decomposition using AXPY
λ, V = eigs(A, nev=10)

# Print eigenvalues and corresponding vectors
println("Eigenvalues: $(λ)")
println("Eigenvectors: $(V[:, 1:10])")



# ===================================================================================================================
#               CUDA Programming Model
# ===================================================================================================================

# Prompt: "Create 3 code suggestions using the following parameters: Julia CUDA AXPY."

# SUGGESTION 1 # ----------------------------------

using CUDA
using LinearAlgebra

# Define matrix dimensions
n = 1000
m = 1000
k = 1000

# Create matrices
A = cu(rand(Float64, n, k))
B = cu(rand(Float64, k, m))
C = similar(A, n, m)

# Define thread block size
blockSize = (32, 32)

# Shared memory size
sharedMem = 16 * 1024

# Number of blocks
numBlocks = div(n, blockSize[1])

# Perform matrix multiplication using AXPY
@cuda begin
    @kernel void axpy!(Int(n), Int(m), Int(k), Float64(A), Float64(B), Float64(C), sharedMem, blockSize)
        @inbounds begin
            # Compute row and column indices within block
            i = blockIdx().x * blockSize().x + threadIdx().x
            j = blockIdx().y * blockSize().y + threadIdx().y

            # Compute partial sum for current element
            var partialSum = zero(Float64)
            for l in 1:k
                partialSum += A[i, l] * B[l, j]

            # Store partial sum in shared memory
            shared Mem[threadIdx().x + threadIdx().y * sharedMem] = partialSum

            # Synchronize threads within block
            syncthreads()

            # Update final result
            if threadIdx().x == 0 && threadIdx().y == 0
                C[i, j] = sharedMem[threadIdx().x + threadIdx().y * sharedMem]
            end
        end
    end
end

# Print result
println("Matrix product: $(C)")


# SUGGESTION 2 # ----------------------------------

using CUDA
using LinearAlgebra

# Define matrix dimensions
n = 1000
p = 100

# Create matrices
A = cu(rand(Float64, n, p))
b = cu(rand(Float64, n))

# Define thread block size
blockSize = (32, 32)

# Shared memory size
sharedMem = 16 * 1024

# Number of blocks
numBlocks = div(n, blockSize[1])

# Perform linear least squares using AXPY
@cuda begin
    @kernel void axpy!(Int(n), Int(p), Float64(A), Float64(b), Float64(x), sharedMem, blockSize)
        @inbounds begin
            # Compute row and column indices within block
            i = blockIdx().x * blockSize().x + threadIdx().x
            j = blockIdx().y * blockSize().y + threadIdx().y

            # Compute partial sum for current element
            var partialSum = zero(Float64)
            for l in 1:p
                partialSum += A[i, l] * b[l]

            # Store partial sum in shared memory
            shared Mem[threadIdx().x + threadIdx().y * sharedMem] = partialSum

            # Synchronize threads within block
            syncthreads()

            # Update final result
            if threadIdx().x == 0 && threadIdx().y == 0
                x[j] = sharedMem[threadIdx().x + threadIdx().y * sharedMem]
            end
        end
    end
end

# Print solution
println("Least squares solution: $x")


# SUGGESTION 3 # ----------------------------------

using CUDA
using LinearAlgebra

# Define matrix dimensions
n = 1000
m = 1000
k = 1000

# Create matrices
A = cu(rand(Float64, n, k))
B = cu(rand(Float64, k, m))
C = similar(A, n, m)

# Define thread block size
blockSize = (32, 32)

# Shared memory size
sharedMem = 16 * 1024

# Number of blocks
numBlocks = div(n, blockSize[1])

# Perform matrix-matrix multiplication using AXPY
@cuda begin
    @kernel void axpy!(Int(n), Int(m), Int(k), Float64(A), Float64(B), Float64(C), sharedMem, blockSize)
        @inbounds begin
            # Compute row and column indices within block
            i = blockIdx().x * blockSize().x + threadIdx().x
            j = blockIdx().y * blockSize().y + threadIdx().y

            # Compute partial sum for current element
            var partialSum = zero(Float64)
            for l in 1:k
                partialSum += A[i, l] * B[l, j]

            # Store partial sum in shared memory
            shared Mem[threadIdx().x + threadIdx().y * sharedMem] = partialSum

            # Synchronize threads within block
            syncthreads()

            # Update final result
            if threadIdx().x == 0 && threadIdx().y == 0
                C[i, j] = sharedMem[threadIdx().x + threadIdx().y * sharedMem]
            end
        end
    end
end

# Print result
println("Matrix-matrix product: $(C)")



# ===================================================================================================================
#               AMDGPU Programming Model
# ===================================================================================================================

# Prompt: "Create 3 code suggestions using the following parameters: Julia AMDGPU AXPY."

# SUGGESTION 1 # ----------------------------------

using AMDGPU
using LinearAlgebra

# Define matrix dimensions
n = 1000
m = 1000
k = 1000

# Create matrices
A = rand(Float64, n, k)
B = rand(Float64, k, m)
C = similar(A, n, m)

# Define thread block size
blockSize = (32, 32)

# Shared memory size
sharedMem = 16 * 1024

# Number of blocks
numBlocks = div(n, blockSize[1])

# Perform matrix-matrix multiplication using AXPY
@amdgpu begin
    @kernel void axpy!(Int(n), Int(m), Int(k), Float64(A), Float64(B), Float64(C), sharedMem, blockSize)
        @inbounds begin
            # Compute row and column indices within block
            i = blockIdx().x * blockSize().x + threadIdx().x
            j = blockIdx().y * blockSize().y + threadIdx().y

            # Compute partial sum for current element
            var partialSum = zero(Float64)
            for l in 1:k
                partialSum += A[i, l] * B[l, j]

            # Store partial sum in shared memory
            shared Mem[threadIdx().x + threadIdx().y * sharedMem] = partialSum

            # Synchronize threads within block
            syncthreads()

            # Update final result
            if threadIdx().x == 0 && threadIdx().y == 0
                C[i, j] = sharedMem[threadIdx().x + threadIdx().y * sharedMem]
            end
        end
    end
end

# Print result
println("Matrix-matrix product: $(C)")


# SUGGESTION 2 # ----------------------------------

using AMDGPU
using LinearAlgebra

# Define vector dimensions
n = 1000

# Create vectors
a = rand(Float64, n)
b = rand(Float64, n)

# Define thread block size
blockSize = (32, )

# Shared memory size
sharedMem = 16 * 1024

# Number of blocks
numBlocks = div(n, blockSize[1])

# Perform vector-vector dot product using AXPY
@amdgpu begin
    @kernel void axpy!(Int(n), Float64(a), Float64(b), Float64(dotProduct), sharedMem, blockSize)
        @inbounds begin
            # Compute index within block
            i = blockIdx().x * blockSize().x + threadIdx().x

            # Compute partial sum for current element
            var partialSum = zero(Float64)
            for j in 1:n
                partialSum += a[i] * b[j]

            # Store partial sum in shared memory
            shared Mem[threadIdx().x + threadIdx().y * sharedMem] = partialSum

            # Synchronize threads within block
            syncthreads()

            # Update final result
            if threadIdx().x == 0 && threadIdx().y == 0
                dotProduct[i] = sharedMem[threadIdx().x + threadIdx().y * sharedMem]
            end
        end
    end
end

# Print result
println("Dot product: $(dotProduct)")


# SUGGESTION 3 # ----------------------------------

using AMDGPU

# Define the arrays
a = zeros(Float64, 1000)
b = zeros(Float64, 1000)

# Set the beta parameter
beta = 2.0

# Perform an all-to-all Axpy operation
AXPY!(a, b, beta)

# Print the results
println(a)



# ===================================================================================================================
#               KernelAbstractions Programming Model
# ===================================================================================================================

# Prompt: "Create 3 code suggestions using the following parameters: Julia KernelAbstractions AXPY."

# SUGGESTION 1 # ----------------------------------

using KernelAbstractions

# Define the kernel
kernel = @kernel begin
    # Define the input arrays
    a, b = @in(Array{Float64}, 2)
    
    # Define the output array
    c = similar(a)
    
    # Perform the Axpy operation
    for i in 1:length(a)
        for j in 1:length(b)
            c[i] += a[i] * b[j]
        end
    end
    
    return c
end

# Create some test data
a = rand(Float64, 1000)
b = rand(Float64, 1000)

# Run the kernel
c = run(kernel, (a, b))

# Print the result
println(c)


# SUGGESTION 2 # ----------------------------------

using KernelAbstractions
using LinearAlgebra

# Define the Axpy function
function axpy(a, b, beta)
    # Perform the Axpy operation
    for i in 1:length(a)
        for j in 1:length(b)
            a[i] += beta * b[j]
        end
    end
end

# Wrap the Axpy function with a kernel
kernel = @kernel axpy(a, b, beta)

# Create some test data
a =rand(Float64, 1000)
b = rand(Float64, 1000)
beta = 2.0

# Run the kernel
result = run(kernel, (a, b, beta))

# Print the result
println(result)


# SUGGESTION 3 # ----------------------------------

using KernelAbstractions

# Define the serial Axpy function
function axpy_serial(a, b, beta)
    # Perform the Axpy operation
    for i in 1:length(a)
        for j in 1:length(b)
            a[i] += beta * b[j]
        end
    end
end

# Wrap the serial Axpy function with a kernel
kernel = @kernel axpy_serial(a, b, beta)

# Create some test data
a = rand(Float64, 1000)
b = rand(Float64, 1000)
beta = 2.0

# Run the kernel in parallel
using Threads
nthreads = 4
@sync begin
    for i in 1:nthreads
        run(kernel, (a, b, beta), threads=i)
    end
end

# Print the result
println(a)



# ---------------------------------------------------------------------------------------------------------------------------------------------------------
