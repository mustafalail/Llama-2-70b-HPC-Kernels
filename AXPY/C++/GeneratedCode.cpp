// OpenMP

// Prompt: "Create 3 code suggestions using the following parameters: C++ OpenMP AXPY function."

// SUGGESTION 1 // ----------------------------------

#include <iostream>
#include <omp.h>

void axpy(float *x, float *y, int n) {
    #pragma omp parallel num_threads(4) default(none) shared(x, y, n)
    {
        int i = omp_get_thread_num();
        int chunk_size = n / 4;
        int start_idx = i * chunk_size;
        int end_idx = (i == 3) ? n : (i + 1) * chunk_size - 1;
        float sum = 0.0f;
        for (int j = start_idx; j <= end_idx; j++) {
            sum += x[j] + y[j];
        }
        x[i] = sum;
    }
}

int main() {
    int n = 1000;
    float *x = new float[n];
    float *y = new float[n];
    // initialize arrays
    for (int i = 0; i < n; i++) {
        x[i] = i % 2 == 0 ? 1.0f : 0.5f;
        y[i] = i % 3 == 0 ? 1.0f : 0.75f;
    }
    axpy(x, y, n);
    // print results
    for (int i = 0; i < n; i++) {
        std::cout << x[i] << " ";
    }
    return 0;
}


// SUGGESTION 2 // ----------------------------------

#include <iostream>
#include <omp.h>

void axpy(float *x, float *y, int n) {
    #pragma omp parallel num_threads(4) default(none) shared(x, y, n)
    {
        int i = omp_get_thread_num();
        int chunk_size = n / 4;
        int start_idx = i * chunk_size;
        int end_idx = (i == 3) ? n : (i + 1) * chunk_size - 1;
        float sum = 0.0f;
        #pragma omp parallel num_threads(2) default(none) shared(sum, start_idx, end_idx)
        {
            int j = start_idx;
            while (j <= end_idx) {
                sum += x[j] + y[j];
                j += 2;
            }
        }
        x[i] = sum;
    }
}

int main() {
    int n = 1000;
    float *x = new float[n];
    float *y = new float[n];
    // initialize arrays
    for (int i = 0; i < n; i++) {
        x[i] = i % 2 == 0 ? 1.0f : 0.5f;
        y[i] = i % 3 == 0 ? 1.0f : 0.75f;
    }
    axpy(x, y, n);
    // print results
    for (int i = 0; i < n; i++) {
        std::cout << x[i] << " ";
    }
    return 0;
}


// SUGGESTION 3 // ----------------------------------

#include <iostream>
#include <omp.h>

double dotProduct(float *x, float *y, int n) {
    double sum = 0.0;
    #pragma omp parallel num_threads(4) default(none) shared(sum, x, y, n)
    {
        int i = omp_get_thread_num();
        int chunk_size = n / 4;
        int start_idx = i * chunk_size;
        int end_idx = (i == 3) ? n : (i + 1) * chunk_size - 1;
        double local_sum = 0.0;
        #pragma omp for schedule(static) nowait
        for (int j = start_idx; j <= end_idx; j++) {
            local_sum += x[j] * y[j];
        }
        #pragma omp critical
        sum += local_sum;
    }
    return sum;
}

int main() {
    int n = 1000;
    float *x = new float[n];
    float *y = new float[n];
    // initialize arrays
    for (int i = 0; i < n; i++) {
        x[i] = i % 2 == 0 ? 1.0f : 0.5f;
        y[i] = i % 3 == 0 ? 1.0f : 0.75f;
    }
    double result = dotProduct(x, y, n);
    // print result
    std::cout << "Dot product: " << result << std::endl;
    return 0;
}



// ---------------------------------------------------------------------------------------------------------------------------------------------------------

// OpenMP(offload)

// Prompt: "Create 3 code suggestions using the following parameters: C++ OpenMP(offload) AXPY function."

// SUGGESTION 1 // ----------------------------------

#include <iostream>
#include <cuda_runtime.h>
#include <omp.h>

__global__ void axpy(float *x, float *y, float *z, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        z[idx] = x[idx] + y[idx];
    }
}

void axpyCPU(float *x, float *y, float *z, int n) {
    #pragma omp parallel num_threads(4) default(none) shared(x, y, z, n)
    {
        int i = omp_get_thread_num();
        int chunk_size = n / 4;
        int start_idx = i * chunk_size;
        int end_idx = (i == 3) ? n : (i + 1) * chunk_size - 1;
        #pragma omp for schedule(static) nowait
        for (int j = start_idx; j <= end_idx; j++) {
            z[j] = x[j] + y[j];
        }
    }
}

int main() {
    int n = 1000;
    float *x = new float[n];
    float *y = new float[n];
    float *z = new float[n];
    // Initialize arrays
    for (int i = 0; i < n; i++) {
        x[i] = i % 2 == 0 ? 1.0f : 0.5f;
        y[i] = i % 3 == 0 ? 1.0f : 0.75f;
    }
    // Offload AXPY function to GPU
    axpy<<<1, 256>>>(x, y, z, n);
    // Synchronize GPU and CPU
    cudaDeviceSynchronize();
    // Print result
    std::cout << "AXPY result: " << z[0] << std::endl;
    return 0;
}


// SUGGESTION 2 // ----------------------------------

#include <iostream>
#include <cuda_runtime.h>
#include <omp.h>

__global__ void axpy(float *x, float *y, float *z, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        z[idx] = x[idx] + y[idx];
    }
}

void axpyCPU(float *x, float *y, float *z, int n) {
    #pragma omp parallel num_threads(4) default(none) shared(x, y, z, n)
    {
        int i = omp_get_thread_num();
        int chunk_size = n / 4;
        int start_idx = i * chunk_size;
        int end_idx = (i == 3) ? n : (i + 1) * chunk_size - 1;
        #pragma omp for schedule(static) nowait
        for (int j = start_idx; j <= end_idx; j++) {
            z[j] = x[j] + y[j];
        }
    }
}

int main() {
    int n = 1000;
    float *x = new float[n];
    float *y = new float[n];
    float *z = new float[n];
    // Initialize arrays
    for (int i = 0; i < n; i++) {
        x[i] = i % 2 == 0 ? 1.0f : 0.5f;
        y[i] = i % 3 == 0 ? 1.0f : 0.75f;
    }
    // Offload AXPY function to GPU
    axpy<<<1, 256>>>(x, y, z, n);
    // Synchronize GPU and CPU
    cudaDeviceSynchronize();
    // Print result
    std::cout << "AXPY result: " << z[0] << std::endl;
    return 0;
}


// SUGGESTION 3 // ----------------------------------

#include <iostream>
#include <cuda_runtime.h>
#include <omp.h>

__global__ void axpy(float *x, float *y, float *z, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        z[idx] = x[idx] + y[idx];
    }
}

void axpyCPU(float *x, float *y, float *z, int n) {
    #pragma omp parallel num_threads(4) default(none) shared(x, y, z, n)
    {
        int i = omp_get_thread_num();
        int chunk_size = n / 4;
        int start_idx = i * chunk_size;
        int end_idx = (i == 3) ? n : (i + 1) * chunk_size - 1;
        #pragma omp for schedule(static) nowait
        for (int j = start_idx; j <= end_idx; j += 4) {
            float xval = x[j];
            float yval = y[j];
            z[j] = xval + yval;
            z[j+1] = x[j+1] + y[j+1];
            z[j+2] = x[j+2] + y[j+2];
            z[j+3] = x[j+3] + y[j+3];
        }
    }
}

int main() {
    int n = 1000;
    float *x = new float[n];
    float *y = new float[n];
    float *z = new float[n];
    // Initialize arrays
    for (int i = 0; i < n; i++) {
        x[i] = i % 2 == 0 ? 1.0f : 0.5f;
        y[i] = i % 3 == 0 ? 1.0f : 0.75f;
    }
    // Offload AXPY function to GPU
    axpy<<<1, 256>>>(x, y, z, n);
    // Synchronize GPU and CPU
    cudaDeviceSynchronize();
    // Print result
    std::cout << "AXPY result: " << z[0] << std::endl;
    return 0;
}



// ---------------------------------------------------------------------------------------------------------------------------------------------------------

// OpenACC

// Prompt: "Create 3 code suggestions using the following parameters: C++ OpenACC AXPY function."

// SUGGESTION 1 // ----------------------------------

#include <iostream>
#include <cuda_runtime.h>
#include <openacc.h>

__device__ void axpy(float *x, float *y, float *z, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        z[idx] = x[idx] + y[idx];
    }
}

int main() {
    int n = 1000;
    float *x = new float[n];
    float *y = new float[n];
    float *z = new float[n];
    // Initialize arrays
    for (int i = 0; i < n; i++) {
        x[i] = i % 2 == 0 ? 1.0f : 0.5f;
        y[i] = i % 3 == 0 ? 1.0f : 0.75f;
    }
    // Define OpenACC region
    #pragma acc data present(x, y, z)
    {
        // Launch kernel
        #pragma acc kernels
        for (int i = 0; i < 100; i++) {
            axpy<<<1, 256>>>(x, y, z, n);
        }
    }
    // Synchronize GPU and CPU
    cudaDeviceSynchronize();
    // Print result
    std::cout << "AXPY result: " << z[0] << std::endl;
    return 0;
}


// SUGGESTION 2 // ----------------------------------

#include <iostream>
#include <cuda_runtime.h>
#include <openacc.h>

__device__ void axpy(float *x, float *y, float *z, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        z[idx] = x[idx] + y[idx];
    }
}

int main() {
    int n = 1000;
    float *x = new float[n];
    float *y = new float[n];
    float *z = new float[n];
    // Initialize arrays
    for (int i = 0; i < n; i++) {
        x[i] = i % 2 == 0 ? 1.0f : 0.5f;
        y[i] = i % 3 == 0 ? 1.0f : 0.75f;
    }
    // Define OpenACC region
    #pragma acc data present(x, y, z)
    {
        // Launch kernel
        #pragma acc kernels async(cudaMemcpyAsync)
        for (int i = 0; i < 100; i++) {
            axpy<<<1, 256>>>(x, y, z, n);
        }
    }
    // Synchronize GPU and CPU
    cudaDeviceSynchronize();
    // Print result
    std::cout << "AXPY result: " << z[0] << std::endl;
    return 0;
}


// SUGGESTION 3 // ----------------------------------

#include <iostream>
#include <cuda_runtime.h>
#include <openacc.h>

__device__ void axpy(float *x, float *y, float *z, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        z[idx] = x[idx] + y[idx];
    }
}

int main() {
    int n = 1000;
    float *x = new float[n];
    float *y = new float[n];
    float *z = new float[n];
    // Initialize arrays
    for (int i = 0; i < n; i++) {
        x[i] = i % 2 == 0 ? 1.0f : 0.5f;
        y[i] = i % 3 == 0 ? 1.0f : 0.75f;
    }
    // Define OpenACC region
    #pragma acc data present(x, y, z)
    {
        // Launch kernel with dynamic parallelism
        #pragma acc kernels async(cudaMemcpyAsync) num_gangs(8) dim(128)
        for (int i = 0; i < 100; i++) {
            axpy<<<1, 256>>>(x, y, z, n);
        }
    }
    // Synchronize GPU and CPU
    cudaDeviceSynchronize();
    // Print result
    std::cout << "AXPY result: " << z[0] << std::endl;
    return 0;
}



// ---------------------------------------------------------------------------------------------------------------------------------------------------------

// HIP(Heterogeneous-Compute Interface for Portability)

// Prompt: "Create 3 code suggestions using the following parameters: C++ HIP(Heterogeneous-Compute Interface for Portability) AXPY function."

// SUGGESTION 1 // ----------------------------------

#include <hip/hip.h>
#include <iostream>

__global__ void axpy(float *x, float *y, float *z, int n) {
    int idx = hipBlockIdx_x * hipBlock Dim_x + hipThreadIdx_x;
    if (idx < n) {
        z[idx] = x[idx] + y[idx];
    }
}

int main() {
    int n = 1000;
    float *x = new float[n];
    float *y = new float[n];
    float *z = new float[n];
    // Initialize arrays
    for (int i = 0; i < n; i++) {
        x[i] = i % 2 == 0 ? 1.0f : 0.5f;
        y[i] = i % 3 == 0 ? 1.0f : 0.75f;
    }
    // Copy data to device
    hipMalloc((void **)&x, n * sizeof(float));
    hipMalloc((void **)&y, n * sizeof(float));
    hipMalloc((void **)&z, n * sizeof(float));
    hipMemcpy(x, h_x, n * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(y, h_y, n * sizeof(float), hipMemcpyHostToDevice);
    // Launch kernel
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    axpy<<<gridSize, blockSize>>>(x, y, z, n);
    // Wait for kernel completion
    hipDeviceSynchronize();
    // Copy result back to host
    hipMemcpy(h_z, z, n * sizeof(float), hipMemcpyDeviceToHost);
    // Free device memory
    hipFree(x);
    hipFree(y);
    hipFree(z);
    // Print result
    std::cout << "AXPY result: " << h_z[0] << std::endl;
    return 0;
}


// SUGGESTION 2 // ----------------------------------

#include <hip/hip.h>
#include <iostream>

__shared__ float smem[256];

__global__ void axpy(float *x, float *y, float *z, int n) {
    int idx = hipBlockIdx_x * hipBlock Dim_x + hipThreadIdx_x;
    if (idx < n) {
        smem[threadIdx.x] = x[idx];
        __syncthreads();
        smem[threadIdx.x] += y[idx];
        __syncthreads();
        z[idx] = smem[threadIdx.x];
    }
}

int main() {
    int n = 1000;
    float *x = new float[n];
    float *y = new float[n];
    float *z = new float[n];
    // Initialize arrays
    for (int i = 0; i < n; i++) {
        x[i] = i % 2 == 0 ? 1.0f : 0.5f;
        y[i] = i % 3 == 0 ? 1.0f : 0.75f;
    }
    // Copy data to device
    hipMalloc((void **)&x, n * sizeof(float));
    hipMalloc((void **)&y, n * sizeof(float));
    hipMalloc((void **)&z, n * sizeof(float));
    hipMemcpy(x, h_x, n * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(y, h_y, n * sizeof(float), hipMemcpyHostToDevice);
    // Launch kernel
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    axpy<<<gridSize, blockSize>>>(x, y, z, n);
    // Wait for kernel completion
    hipDeviceSynchronize();
    // Copy result back to host
    hipMemcpy(h_z, z, n * sizeof(float), hipMemcpyDeviceToHost);
    // Free device memory
    hipFree(x);
    hipFree(y);
    hipFree(z);
    // Print result
    std::cout << "AXPY result: " << h_z[0] << std::endl;
    return 0;
}


// SUGGESTION 3 // ----------------------------------

#include <hip/hip.h>
#include <iostream>

__global__ void axpy(float *x, float *y, float *z, int n) {
    int idx = hipBlockIdx_x * hipBlock Dim_x + hipThreadIdx_x;
    if (idx < n) {
        float temp = x[idx];
        for (int i = 0; i < 4; i++) {
            temp += y[idx + i];
        }
        z[idx] = temp;
    }
}

int main() {
    // Define the number of blocks and threads per block
int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
int threadsPerBlock = BLOCK_SIZE;

// Allocate memory on the device for the output array
float *d_z;
cudaMalloc((void **)&d_z, n * sizeof(float));

// Launch the kernel
axpy<<<numBlocks, threadsPerBlock>>>(d_z, x, y, n);

// Wait for the kernel to finish
cudaDeviceSynchronize();

// Copy the result from the device to the host
float *h_z;
cudaMemcpy(h_z, d_z, n * sizeof(float), cudaMemcpyDeviceToHost);

// Free the memory on the device
cudaFree(d_z);

// Print the result
std::cout << "AXPY result: " << h_z[0] << std::endl;

return 0;
}



// ---------------------------------------------------------------------------------------------------------------------------------------------------------

// CUDA

// Prompt: "Create 3 code suggestions using the following parameters: C++ CUDA AXPY function."

// SUGGESTION 1 // ----------------------------------

#include <cuda_runtime.h>
#include <iostream>

__device__ float axpy(float x, float y, float z) {
    return x + y * z;
}

int main() {
    const int N = 1024;
    float *x, *y, *z;
    cudaMalloc((void **)&x, N * sizeof(float));
    cudaMalloc((void **)&y, N * sizeof(float));
    cudaMalloc((void **)&z, N * sizeof(float));

    // Initialize data
    for (int i = 0; i < N; ++i) {
        x[i] = i / 10.0f;
        y[i] = i * 10.0f;
        z[i] = 0.0f;
    }

    // Launch kernel
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    axpy<<<gridSize, blockSize>>>(x, y, z, N);

    // Sync and print
    cudaDeviceSynchronize();
    for (int i = 0; i < N; ++i) {
        std::cout << z[i] << ", ";
    }
    std::cout << std::endl;

    cudaFree(x);
    cudaFree(y);
    cudaFree(z);
    return 0;
}


// SUGGESTION 2 // ----------------------------------

#include <cuda_runtime.h>
#include <iostream>

__device__ float axpy(float x, float y, float z) {
    return x + y * z;
}

int main() {
    const int N = 1024;
    float *x, *y, *z;
    cudaMalloc((void **)&x, N * sizeof(float));
    cudaMalloc((void **)&y, N * sizeof(float));
    cudaMalloc((void **)&z, N * sizeof(float));

    // Initialize data
    for (int i = 0; i < N; ++i) {
        x[i] = i / 10.0f;
        y[i] = i * 10.0f;
        z[i] = 0.0f;
    }

    // Shared memory declaration
    __shared__ float smem[256];

    // Kernel launch configuration
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // Launch kernel
    axpy<<<gridSize, blockSize, 256>>>(x, y, z, N, smem);

    // Sync and print
    cudaDeviceSynchronize();
    for (int i = 0; i < N; ++i) {
        std::cout << z[i] << ", ";
    }
    std::cout << std::endl;

    cudaFree(x);
    cudaFree(y);
    cudaFree(z);
    return 0;
}


// SUGGESTION 3 // ----------------------------------

#include <cuda_runtime.h>
#include <iostream>

__device__ float axpy(float x, float y, float z) {
    return x + y * z;
}

int main() {
    const int N = 1024;
    float *x, *y, *z;
    cudaMalloc((void **)&x, N * sizeof(float));
    cudaMalty(void **)&y, N * sizeof(float));
    cudaMalloc((void **)&z, N * sizeof(float));

    // Initialize data
    for (int i = 0; i < N; ++i) {
        x[i] = i / 10.0f;
        y[i] = i * 10.0f;
        z[i] = 0.0f;
    }

    // Constant memory declaration
    __constant__ float cmem[256];

    // Kernel launch configuration
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // Load constant memory
    cudaMemcpyToSymbol(cmem, &x[0], N * sizeof(float), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(cmem + N, &y[0], N * sizeof(float), 0, cudaMemcpyHostToDevice);

    // Launch kernel
    axpy<<<gridSize, blockSize, 256>>>(x, y, z, N, cmem);

    // Sync and print
    cudaDeviceSynchronize();
    for (int i = 0; i < N; ++i) {
        std::cout << z[i] << ", ";
    }
    std::cout << std::endl;

    cudaFree(x);
    cudaFree(y);
    cudaFree(z);
    return 0;
}



// ---------------------------------------------------------------------------------------------------------------------------------------------------------
