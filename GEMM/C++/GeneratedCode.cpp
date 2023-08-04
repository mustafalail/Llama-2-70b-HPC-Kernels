// OpenMP

// Prompt: "Give me code for the following:
// C++, OpenMP, general matrix-matrix multiply, function"


#include <iostream>
#include <omp.h>

void matrixMultiply(double* A, double* B, double* C, int m, int n, int p) {
    #pragma omp parallel num_threads(4) default(none) shared(A, B, C, m, n, p)
    {
        int i, j, k;
        double sum;
        
        #pragma omp for schedule(static)
        for (i = 0; i < m; ++i) {
            #pragma omp parallel for schedule(static)
            for (j = 0; j < n; ++j) {
                sum = 0;
                #pragma omp parallel for schedule(static)
                for (k = 0; k < p; ++k) {
                    sum += A[i * p + k] * B[k * n + j];
                }
                C[i * n + j] = sum;
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

// OpenMP (offload)

// Prompt: "Give me code for the following:
// C++, OpenMP (offload), general matrix-matrix multiply, function"

#include <iostream>
#include <omp.h>

void matrixMultiply(double* A, double* B, double* C, int m, int n, int p) {
    // Declare device variables
    double* d_A, *d_B, *d_C;

    // Allocate memory on device
    cudaMalloc((void**)&d_A, m * p * sizeof(double));
    cudaMalloc((void**)&d_B, n * p * sizeof(double));
    cudaMalloc((void**)&d_C, m * n * sizeof(double));

    // Copy data from host to device
    cudaMemcpy(d_A, A, m * p * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, n * p * sizeof(double), cudaMemcpyHostToDevice);

    // Define kernel launch configuration
    int blockSize = 128;
    int gridSize = (m + blockSize - 1) / blockSize;

    // Launch kernel to compute matrix multiplication
    #pragma omp parallel num_threads(4) default(none) shared(d_A, d_B, d_C, m, n, p, blockSize, gridSize)
    {
        int i, j, k;
        double sum;

        // Compute thread block size
        int blockIdx = blockIdx.x * blockSize + threadIdx.x;
        int blockDim = blockSize * gridSize;

        // Iterate over blocks of A and B
        for (int b = 0; b < gridSize; ++b) {
            for (int a = 0; a < blockDim; ++a) {
                int i = blockIdx.y * blockDim + a;
                int j = blockIdx.x * blockDim + b;

                // Compute dot product of row i of A and column j of B
                sum = 0;
                for (k = 0; k < p; ++k) {
                    sum += d_A[i * p + k] * d_B[k * n + j];
                }

                // Store result in C
                d_C[i * n + j] = sum;
            }
        }
    }

    // Synchronize threads
    cudaDeviceSynchronize();

    // Copy results from device to host
    cudaMemcpy(C, d_C, m * n * sizeof(double), cudaMemcpyDeviceToHost);

    // Free memory on device
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

// HIP (Heterogeneous-Compute Interface for Portability)

// Prompt: "Give me code for the following:
// C++, Heterogeneous-Compute Interface for Portability, general matrix-matrix multiply, function"

#include <hip/hip_runtime.h>
#include <iostream>

// Matrix dimensions
const int M = 1024;
const int N = 1024;
const int P = 1024;

// Define hip functions
__global__ void matrixMultiplyKernel(float* A, float* B, float* C, int m, int n, int p) {
  // Thread block size
  const int blockSize = 128;

  // Calculate thread ID within block
  int tid = blockIdx.x * blockSize + threadIdx.x;
  int tids = blockIdx.x * blockSize + threadIdx.x + blockSize;

  // Calculate global thread ID
  int gtid = blockIdx.x * gridDim.x * blockSize + threadIdx.x;

  // Shared memory allocation
  __shared__ float smem[blockSize][P];

  // Load data from global memory into shared memory
  if (gtid < m * n) {
    int i = gtid / n;
    int j = gtid % n;
    for (int k = 0; k < P; ++k) {
      smem[threadIdx.x][k] = A[i * P + k] * B[j * P + k];
    }
  }

  // Wait for all threads in block to finish loading data
  __syncthreads();

  // Perform matrix multiplication
  if (gtid < m * n) {
    int i = gtid / n;
    int j = gtid % n;
    float sum = 0;
    for (int k = 0; k < P; ++k) {
      sum += smem[threadIdx.x][k] * B[j * P + k];
    }
    C[i * n + j] = sum;
  }
}

int main() {
  // Initialize hip context and devices
  hip::init();
  hip::device dev = hip::get_default_device();

  // Create arrays on host and device
  float* A = new float[M * P];
  float* B = new float[N * P];
  float* C = new float[M * N];
  hip::array<float> d_A(A, M * P);
  hip::array<float> d_B(B, N * P);
  hip::array<float> d_C(C, M * N);

  // Set up kernel parameters
  int blockSize = 128;
  int gridSize = (M + blockSize - 1) / blockSize;

  // Launch kernel
  hipLaunchKernelGGL(matrixMultiplyKernel, dim3(gridSize), dim3(blockSize), 0, NULL, d_A, d_B, d_C, M, N, P);

  // Wait for kernel to complete
  hipDeviceSynchronize();

  // Delete arrays on device
  delete[] d_A;
  delete[] d_B;
  delete[] d_C;

  return 0;
}

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

// OpenACC

// Prompt: "Give me code for the following:
// C++, OpenACC, general matrix-matrix multiply, function"

#include <iostream>
#include <cmath>
#include <openacc.h>

// Matrix dimensions
const int M = 1024;
const int N = 1024;
const int P = 1024;

// Define OpenACC data types
typedef float acc_data_t;
typedef acc_data_t (*acc_func_t)(acc_data_t, acc_data_t, int, int, int);

// Define OpenACC kernel function
acc_func_t matrixMultiplyKernel = (acc_func_t)NULL;

// Initialize OpenACC environment
void initAcc() {
  // Create OpenACC accelerator
  acc_device_t dev = acc_create_device(ACC_DEVICE_TYPE_GPU, "gpu", 0);

  // Create OpenACC context
  acc_context_t ctx = acc_create_context(dev, ACC_CONTEXT_THREAD_LOCAL, 0);

  // Set OpenACC kernel function
  matrixMultiplyKernel = acc_register_function(ctx, "matrixMultiplyKernel", ACCTYPE_FUNCTION_DOUBLE, sizeof(double));
}

// Cleanup OpenACC resources
void cleanupAcc() {
  // Destroy OpenACC context
  acc_destroy_context(matrixMultiplyKernel->context);

  // Destroy OpenACC accelerator
  acc_destroy_device(matrixMultiplyKernel->device);
}

// General matrix-matrix multiply function
void matrixMultiply(float* A, float* B, float* C, int m, int n, int p) {
  // Initialize thread blocks and grids
  int blockSize = 128;
  int gridSize = (m + blockSize - 1) / blockSize;

  // Allocate shared memory for matrix elements
  acc_data_t* sMem = acc_malloc(sizeof(acc_data_t) * blockSize * P);

  // Copy input matrices to device
  acc_memcpy(A, &sMem[0], sizeof(acc_data_t) * m * P, cudaMemcpyHostToDevice);
  acc_memcpy(B, &sMem[0], sizeof(acc_data_t) * n * P, cudaMemcpyHostToDevice);

  // Launch kernel
  matrixMultiplyKernel(sMem, blockSize, gridSize, m, n, p);

  // Synchronize threads
  acc_barrier(ACC_BARRIER_KERNEL);

  // Copy result matrix from device to host
  acc_memcpy(&C[0], sMem, sizeof(acc_data_t) * m * n, cudaMemcpyDeviceToHost);

  // Free shared memory
  acc_free(sMem);
}

int main() {
  // Initialize OpenACC environment
  initAcc();

  // Allocate memory for input and output matrices
  float* A = new float[M * P];
  float* B = new float[N * P];
  float* C = new float[M * N];

  // Initialize input matrices
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < P; ++j) {
      A[i * P + j] = i * P + j;
    }
  }

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < P; ++j) {
      B[i * P + j] = i * P + j;
    }
  }

  // Call matrix multiplication function
  matrixMultiply(A, B, C, M, N, P);

  // Print result matrix
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      std::cout << C[i * N + j] << " ";
    }
    std::cout << std::endl;
  }

  // Cleanup OpenACC resources
  cleanupAcc();

  return 0;
}

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

// CUDA

// Prompt: "Give me code for the following:
// C++, CUDA, general matrix-matrix multiply, function"

#include <iostream>
#include <cuda_runtime.h>

__global__ void matrixMultiplyKernel(float *A, float *B, float *C, int m, int n, int p) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= m || j >= n) return;

  float sum = 0;
  for (int k = 0; k < p; ++k) {
    sum += A[i * p + k] * B[k * n + j];
  }
  C[i * n + j] = sum;
}

int main() {
  int m = 4;
  int n = 5;
  int p = 3;

  // allocate memory on host
  float *A = new float[m * p];
  float *B = new float[n * p];
  float *C = new float[m * n];

  // initialize data
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < p; ++j) {
      A[i * p + j] = i * p + j;
    }
  }

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < p; ++j) {
      B[i * p + j] = i * p + j;
    }
  }

  // copy data to device
  float *dA, *dB, *dC;
  cudaMalloc((void **)&dA, m * p * sizeof(float));
  cudaMalloc((void **)&dB, n * p * sizeof(float));
  cudaMalloc((void **)&dC, m * n * sizeof(float));
  cudaMemcpy(dA, A, m * p * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B, n * p * sizeof(float), cudaMemcpyHostToDevice);

  // set block and grid sizes
  int blockSize = 16;
  int gridSize = (m + blockSize - 1) / blockSize;

  // launch kernel
  matrixMultiplyKernel<<<gridSize, blockSize>>>(dA, dB, dC, m, n, p);

  // copy result from device to host
  cudaMemcpy(C, dC, m * n * sizeof(float), cudaMemcpyDeviceToHost);

  // free memory on device
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);

  // print result
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      std::cout << C[i * n + j] << " ";
    }
    std::cout << std::endl;
  }

  delete[] A;
  delete[] B;
  delete[] C;

  return 0;
}

