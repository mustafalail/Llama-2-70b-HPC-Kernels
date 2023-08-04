// OpenMP

// Prompt: "Give me code for the following:
// C++, OpenMP, general matrix-vector multiply, function"

#include <iostream>
#include <omp.h>

#include <iostream>
#include <omp.h>

void matrixVectorMultiply(double* A, double* B, double* C, int m, int n, int k) {
    #pragma omp parallel num_threads(4) default(none) shared(A, B, C, m, n, k)
    {
        int i = omp_get_thread_num();
        int chunkSize = (m + k - 1) / 4;
        int startRow = i * chunkSize;
        int endRow = (i == 3) ? m : (startRow + chunkSize);
        int startCol = 0;
        int endCol = n;
        
        // Each thread computes its own portion of the result vector
        double* C_local = new double[endCol - startCol];
        #pragma omp for schedule(static) nowait
        for (int j = startCol; j < endCol; ++j) {
            double sum = 0;
            for (int l = startRow; l < endRow; ++l) {
                sum += A[l * k + j] * B[i * k + l];
            }
            C_local[j - startCol] = sum;
        }
        
        // Combine the local results from all threads
        #pragma omp barrier
        std::cout << "Thread " << i << " done\n";
        if (i == 0) {
            memcpy(C, C_local, (endCol - startCol) * sizeof(double));
        }
        #pragma omp barrier
    }
}


// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

// OpenMP (offload)

// Prompt: "Give me code for the following:
// C++, OpenMP (offload), general matrix-vector multiply, function"

#include <iostream>
#include <omp.h>

// Declare a function that performs a matrix-vector multiplication
void matrixVectorMultiply(double* A, double* B, double* C, int m, int n, int k) {
  // Check if the input dimensions match
  if (m != n || n != k) {
    std::cerr << "Error: Dimensions do not match" << std::endl;
    return;
  }

  // Initialize OpenMP
  omp_set_num_threads(4);

  // Define a data sharing construct for the input arrays
  #pragma omp declare mapper(A, B, C)

  // Offload the computation to a device (e.g., GPU)
  #pragma omp parallel num_threads(4) default(none) shared(A, B, C, m, n, k)
  {
    // Each thread computes its own portion of the result vector
    double* C_local = new double[n];
    #pragma omp for schedule(static) nowait
    for (int j = 0; j < n; ++j) {
      double sum = 0;
      for (int l = 0; l < m; ++l) {
        sum += A[l * k + j] * B[l];
      }
      C_local[j] = sum;
    }

    // Synchronize threads
    #pragma omp barrier

    // Copy the local results from all threads to the host
    #pragma omp single
    {
      std::copy(C_local, C_local + n, C, n);
    }

    // Release resources
    delete[] C_local;
  }
}

int main() {
  int m = 8;
  int n = 8;
  int k = 8;

  double* A = new double[m * k];
  double* B = new double[k];
  double* C = new double[n];

  // Initialize inputs
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < k; ++j) {
      A[i * k + j] = i % 2 == 0 ? 1.0 : 0.5;
    }
  }
  for (int i = 0; i < k; ++i) {
    B[i] = i % 2 == 0 ? 1.0 : 0.5;
  }

  // Call the matrix-vector multiplication function
  matrixVectorMultiply(A, B, C, m, n, k);

  // Print the result
  for (int i = 0; i < n; ++i) {
    std::cout << C[i] << " ";
  }
  std::cout << std::endl;

  // Release resources
  delete[] A;
  delete[] B;
  delete[] C;

  return 0;
}

// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

// HIP (Heterogeneous-Compute Interface for Portability)

// Prompt: "Give me code for the following:
// C++, Heterogeneous-Compute Interface for Portability, general matrix-vector multiply, function"


#include <hip/hip_runtime.h>
#include <iostream>

__global__ void matrixVectorMultiplyKernel(float* A, float* B, float* C, int m, int n, int k) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;

  float sum = 0;
  for (int l = 0; l < m; ++l) {
    sum += A[l * k + idx] * B[l];
  }
  C[idx] = sum;
}

void matrixVectorMultiply(float* A, float* B, float* C, int m, int n, int k) {
  int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  int threadsPerBlock = BLOCK_SIZE;

  hipLaunchKernelGGL(matrixVectorMultiplyKernel, dim3(numBlocks), dim3(threadsPerBlock), 0, 0, A, B, C, m, n, k);
}

int main() {
  int m = 8;
  int n = 8;
  int k = 8;

  float* A = new float[m * k];
  float* B = new float[k];
  float* C = new float[n];

  // Initialize inputs
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < k; ++j) {
      A[i * k + j] = i % 2 == 0 ? 1.0f : 0.5f;
    }
  }
  for (int i = 0; i < k; ++i) {
    B[i] = i % 2 == 0 ? 1.0f : 0.5f;
  }

  // Call the kernel
  matrixVectorMultiply(A, B, C, m, n, k);

  // Print the result
  for (int i = 0; i < n; ++i) {
    std::cout << C[i] << " ";
  }
  std::cout << std::endl;

  // Release resources
  delete[] A;
  delete[] B;
  delete[] C;

  return 0;
}

// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

// OpenACC

// Prompt: "Give me code for the following:
// C++, OpenACC, general matrix-vector multiply, function"

#include <iostream>
#include <openacc.h>

// Define the matrix and vector sizes
const int M = 8;
const int N = 8;
const int K = 8;

// Define the matrix and vector data
float A[M][N] = {{1, 2, 3, 4, 5, 6, 7, 8},
                  {9, 10, 11, 12, 13, 14, 15, 16},
                  {17, 18, 19, 20, 21, 22, 23, 24},
                  {25, 26, 27, 28, 29, 30, 31, 32},
                  {33, 34, 35, 36, 37, 38, 39, 40},
                  {41, 42, 43, 44, 45, 46, 47, 48},
                  {49, 50, 51, 52, 53, 54, 55, 56},
                  {57, 58, 59, 60, 61, 62, 63, 64}};
float B[N] = {1, 2, 3, 4, 5, 6, 7, 8};
float C[M] = {};

// Define the OpenACC region
#pragma acc region( Gang )
{
  // Perform matrix-vector multiplication
  #pragma acc parallel loop gang worker(C)
  for (int i = 0; i < M; ++i) {
    float sum = 0;
    for (int j = 0; j < N; ++j) {
      sum += A[i][j] * B[j];
    }
    C[i] = sum;
  }
}

int main() {
  // Run the OpenACC region
  #pragma acc runtime
  {
    // Launch the OpenACC region
    #pragma acc spawn regions( Gang )
  }

  // Synchronize the threads
  #pragma acc synchronize

  // Print the result
  for (int i = 0; i < M; ++i) {
    std::cout << C[i] << " ";
  }
  std::cout << std::endl;

  return 0;
}

// --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

// CUDA

// Prompt: "Give me code for the following:
// C++, CUDA, general matrix-vector multiply, function"

#include <cuda_runtime.h>

__global__ void matrixVectorMultiplyKernel(float* A, float* B, float* C, int m, int n, int k) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= m) return;
  
  float sum = 0;
  for (int i = 0; i < k; ++i) {
    sum += A[idx + i * m] * B[i];
  }
  C[idx] = sum;
}

void matrixVectorMultiply(float* A, float* B, float* C, int m, int n, int k) {
  int numBlocks = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim3 blocks(numBlocks, 1, 1);
  dim3 threads(BLOCK_SIZE, 1, 1);
  
  matrixVectorMultiplyKernel<<<blocks, threads>>>(A, B, C, m, n, k);
}

int main() {
  const int m = 8;
  const int n = 8;
  const int k = 8;
  
  float* A = new float[m * k];
  float* B = new float[k * n];
  float* C = new float[m * n];
  
  // Initialize matrices
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < k; ++j) {
      A[i * k + j] = i % 2 == 0 ? 1.0f : 0.0f;
    }
  }
  for (int i = 0; i < k; ++i) {
    for (int j = 0; j < n; ++j) {
      B[i * n + j] = i % 2 == 0 ? 1.0f : 0.0f;
    }
  }
  
  matrixVectorMultiply(A, B, C, m, n, k);
  
  // Print result
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


