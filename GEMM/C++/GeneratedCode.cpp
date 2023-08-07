//===================================================================================================================
//               OpenMP Programming Model
//===================================================================================================================

// Prompt: "Create 3 code suggestions using the following parameters: C++ OpenMP General matrix-matrix multiply function."

// SUGGESTION 1 // ----------------------------------

#include <iostream>
#include <omp.h>

void matrixMultiply(double** A, double** B, double** C, int m, int n, int p) {
    #pragma omp parallel num_threads(p) default(none) shared(A, B, C, m, n)
    {
        int i = omp_get_thread_num();
        int chunkSize = (m + p - 1) / p;
        int startRow = i * chunkSize;
        int endRow = (i == p - 1) ? m : (startRow + chunkSize);
        int startCol = 0;
        int endCol = n;
        
        // Multiply rows of A and columns of B and sum into C
        #pragma omp for schedule(static) nowait
        for (int j = startCol; j <= endCol; ++j) {
            for (int k = startRow; k <= endRow; ++k) {
                double sum = 0;
                for (int l = 0; l < p; ++l) {
                    if (l != i) {
                        sum += A[k][l] * B[l][j];
                    }
                }
                C[k][j] = sum;
            }
        }
    }
}


// SUGGESTION 2 // ----------------------------------

#include <iostream>
#include <omp.h>

void matrixMultiply(double** A, double** B, double** C, int m, int n, int p) {
    #pragma omp taskwait
    {
        int i = omp_get_thread_num();
        int chunkSize = (m + p - 1) / p;
        int startRow = i * chunkSize;
        int endRow = (i == p - 1) ? m : (startRow + chunkSize);
        int startCol = 0;
        int endCol = n;
        
        // Multiply rows of A and columns of B and sum into C
        #pragma omp task firstprivate(A, B, C, m, n)
        #pragma omp task shared(A, B, C, m, n)
        {
            for (int j = startCol; j <= endCol; ++j) {
                for (int k = startRow; k <= endRow; ++k) {
                    double sum = 0;
                    for (int l = 0; l < p; ++l) {
                        if (l != i) {
                            sum += A[k][l] * B[l][j];
                        }
                    }
                    C[k][j] = sum;
                }
            }
        }
    }
}


// SUGGESTION 3 // ----------------------------------

#include <iostream>
#include <omp.h>

void matrixMultiply(double** A, double** B, double** C, int m, int n, int p) {
    #pragma omp parallel num_threads(p) default(none)
    {
        int i = omp_get_thread_num();
        int chunkSize = (m + p - 1) / p;
        int startRow = i * chunkSize;
        int endRow = (i == p - 1) ? m : (startRow + chunkSize);
        int startCol = 0;
        int endCol = n;
        
        // Multiply rows of A and columns of B and sum into C
        #pragma omp sections
        {
            #pragma omp section
            {
                for (int j = startCol; j <= endCol; ++j) {
                    for (int k = startRow; k <= endRow; ++k) {
                        double sum = 0;
                        for (int l = 0; l < p; ++l) {
                            if (l != i) {
                                sum += A[k][l] * B[l][j];
                            }
                        }
                        C[k][j] = sum;
                    }
                }
            }
        }
    }
}



//===================================================================================================================
//               OpenMP(offload) Programming Model
//===================================================================================================================

// Prompt: "Create 3 code suggestions using the following parameters: C++ OpenMP(offload) General matrix-matrix multiply function."

// SUGGESTION 1 // ----------------------------------

#include <iostream>
#include <omp.h>

void matrixMultiply(double* A, double* B, double* C, int m, int n, int p) {
    #pragma omp parallel num_threads(p) default(none)
    {
        int i = omp_get_thread_num();
        int chunkSize = (m + p - 1) / p;
        int startRow = i * chunkSize;
        int endRow = (i == p - 1) ? m : (startRow + chunkSize);
        int startCol = 0;
        int endCol = n;
        
        // Offload computation to device
        #pragma omp offload target(device:A,B,C)
        for (int j = startCol; j <= endCol; ++j) {
            for (int k = startRow; k <= endRow; ++k) {
                double sum = 0;
                for (int l = 0; l < p; ++l) {
                    if (l != i) {
                        sum += A[k][l] * B[l][j];
                    }
                }
                C[k][j] = sum;
            }
        }
    }
}


// SUGGESTION 2 // ----------------------------------

#include <iostream>
#include <omp.h>

void matrixMultiply(double* A, double* B, double* C, int m, int n, int p) {
    #pragma omp parallel num_threads(p) default(none)
    {
        int i = omp_get_thread_num();
        int chunkSize = (m + p - 1) / p;
        int startRow = i * chunkSize;
        int endRow = (i == p - 1) ? m : (startRow + chunkSize);
        int startCol = 0;
        int endCol = n;
        
        // Allocate shared memory for intermediate results
        double* sharedMem = new double[p * n];
        
        // Compute local contributions to shared memory
        #pragma omp for schedule(static) nowait
        for (int j = startCol; j <= endCol; ++j) {
            for (int k = startRow; k <= endRow; ++k) {
                double sum = 0;
                for (int l = 0; l < p; ++l) {
                    if (l != i) {
                        sum += A[k][l] * B[l][j];
                    }
                }
                sharedMem[i * n + j] = sum;
            }
        }
        
        // Synchronize threads
        #pragma omp barrier
        
        // Sum up local contributions from shared memory
        #pragma omp for schedule(static) nowait
        for (int j = startCol; j <= endCol; ++j) {
            double sum = 0;
            for (int l = 0; l < p; ++l) {
                sum += sharedMem[l * n + j];
            }
            C[startRow + j] = sum;
        }
        
        delete[] sharedMem;
    }
}


// SUGGESTION 3 // ----------------------------------

#include <iostream>
#include <omp.h>

void matrixMultiply(double* A, double* B, double* C, int m, int n, int p) {
    #pragma omp parallel num_threads(p) default(none)
    {
        int i = omp_get_thread_num();
        int chunkSize = (m + p - 1) / p;
        int startRow = i * chunkSize;
        int endRow = (i == p - 1) ? m : (startRow + chunkSize);
        int startCol = 0;
        int endCol = n;
        
        // Dynamically adjust the number of threads based on the workload
        #pragma omp for schedule(dynamic)
        for (int j = startCol; j <= endCol; ++j) {
            for (int k = startRow; k <= endRow; ++k) {
                double sum = 0;
                for (int l = 0; l < p; ++l) {
                    if (l != i) {
                        sum += A[k][l] * B[l][j];
                    }
                }
                C[k][j] = sum;
            }
        }
    }
}



//===================================================================================================================
//               OpenACC Programming Model
//===================================================================================================================

// Prompt: "Create 3 code suggestions using the following parameters: C++ OpenACC General matrix-matrix multiply function."

// SUGGESTION 1 // ----------------------------------

#include <iostream>
#include <openacc.h>

void matrixMultiply(double* A, double* B, double* C, int m, int n, int p) {
    #pragma acc data present(A, B, C)
    {
        #pragma acc parallel num_gangs(p) gang(worker)
        {
            int i = worker->idx;
            int j = worker->tid;
            
            for (int k = 0; k < m; ++k) {
                C[k][j] = 0;
                for (int l = 0; l < n; ++l) {
                    C[k][j] += A[k][l] * B[l][j];
                }
            }
        }
    }
}


// SUGGESTION 2 // ----------------------------------

#include <iostream>
#include <openacc.h>

void matrixMultiply(double* A, double* B, double* C, int m, int n, int p) {
    #pragma acc data present(A, B, C)
    {
        #pragma acc parallel num_gangs(p) gang(worker)
        {
            int i = worker->idx;
            int j = worker->tid;
            
            // Allocate shared memory for temporary storage
            double* smem = (double*)malloc(n * sizeof(double));
            
            for (int k = 0; k < m; ++k) {
                C[k][j] = 0;
                for (int l = 0; l < n; ++l) {
                    smem[l] = A[k][l] * B[l][j];
                }
                for (int l = 0; l < n; ++l) {
                    C[k][j] += smem[l];
                }
            }
            
            free(smem);
        }
    }
}


// SUGGESTION 3 // ----------------------------------

#include <iostream>
#include <openacc.h>

void matrixMultiply(double* A, double* B, double* C, int m, int n, int p) {
    #pragma acc data present(A, B, C)
    {
        #pragma acc parallel num_gangs(p) gang(worker)
        {
            int i = worker->idx;
            int j = worker->tid;
            
            // Use strided access pattern to reduce memory coalescing issues
            for (int k = 0; k < m; k += p) {
                for (int l = 0; l < n; l += p) {
                    double sum = 0;
                    for (int q = 0; q < p; ++q) {
                        sum += A[k + q][l + q] * B[l + q][j];
                    }
                    C[k][j] = sum;
                }
            }
        }
    }
}



//===================================================================================================================
//               HIP Programming Model
//===================================================================================================================

// Prompt: "Create 3 code suggestions using the following parameters: C++ HIP(Heterogeneous-Compute Interface for Portability) General matrix-matrix multiply function."

// SUGGESTION 1 // ----------------------------------

#include <hip/hip_runtime.h>
#include <iostream>

__global__ void matrixMultiplyKernel(float* A, float* B, float* C, int m, int n, int p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= m || j >= n) return;
    float sum = 0;
    for (int k = 0; k < p; ++k) {
        sum += A[i + k * m] * B[k * n + j];
    }
    C[i * n + j] = sum;
}

int main() {
    const int m = 8;
    const int n = 8;
    const int p = 4;
    float* A = new float[m * p];
    float* B = new float[p * n];
    float* C = new float[m * n];
    hipDeviceptr_t d_A = hipMemcpyToSymbol(A, hipDeviceCurrent, 0, m * p * sizeof(float), hipMemoryTypeDevice);
    hipDeviceptr_t d_B = hipMemcpyToSymbol(B, hipDeviceCurrent, 0, p * n * sizeof(float), hipMemoryTypeDevice);
    hipDeviceptr_t d_C = hipMemcpyToSymbol(C, hipDeviceCurrent, 0, m * n * sizeof(float), hipMemoryTypeDevice);
    matrixMultiplyKernel<<<1, 256>>>(d_A, d_B, d_C, m, n, p);
    hipDeviceSynchronize();
    delete[] A;
    delete[] B;
    delete[] C;
    return 0;
}


// SUGGESTION 2 // ----------------------------------

#include <hip/hip_runtime.h>
#include <iostream>

__shared__ float smem[32][32];

__global__ void matrixMultiplyKernel(float* A, float* B, float* C, int m, int n, int p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= m || j >= n) return;
    float sum = 0;
    for (int k = 0; k < p; ++k) {
        sum += A[i + k * m] * B[k * n + j];
    }
    smem[threadIdx.x][threadIdx.y] = sum;
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        C[i * n + j] = smem[0][0];
    }
}

int main() {
    const int m = 8;
    const int n = 8;
    const int p = 4;
    float* A = new float[m * p];
    float* B = new float[p * n];
    float* C = new float[m * n];
    hipDeviceptr_t d_A = hipMemcpyToSymbol(A, hipDeviceCurrent, 0, m * p * sizeof(float), hipMemoryTypeDevice);
    hipDeviceptr_t d_B = hipMemcpyToSymbol(B, hipDeviceCurrent, 0, p * n * sizeof(float), hipMemoryTypeDevice);
    hipDeviceptr_t d_C = hipMemcpyToSymbol(C, hipDeviceCurrent, 0, m * n * sizeof(float), hipMemoryTypeDevice);
    int threadsPerBlock = 256;
    int numBlocks = (m + threadsPerBlock - 1) / threadsPerBlock;
    matrixMultiplyKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, m, n, p);
    hipDeviceSynchronize();
    delete[] A;
    delete[] B;
    delete[] C;
    return 0;
}


// SUGGESTION 3 // ----------------------------------

#include <hip/hip_runtime.h>

__global__ void matrixMultiplyKernel(float* A, float* B, float* C, int m, int n, int p) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row >= m || col >= n) return;

  float sum = 0;
  for (int k = 0; k < p; ++k) {
    sum += hipCublasSdot(n, A + k * n, 1, B + k * m, 1);
  }
  C[row * n + col] = sum;
}

int main() {
  int m = 8;
  int n = 8;
  int p = 4;

  // Allocate memory on host
  float* A = new float[m * p];
  float* B = new float[p * n];
  float* C = new float[m * n];

  // Initialize data
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < p; ++j) {
      A[i * p + j] = Rand();
    }
  }
  for (int i = 0; i < p; ++i) {
    for (int j = 0; j < n; ++j) {
      B[i * n + j] = Rand();
    }
  }

  // Copy data to device
  hipMemcpy(d_A, A, m * p * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(d_B, B, p * n * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(d_C, C, m * n * sizeof(float), hipMemcpyHostToDevice);

  // Launch kernel
  int blocks = (m + blockSize.x - 1) / blockSize.x;
  int threads = blockSize.x * blockSize.y;
  matrixMultiplyKernel<<<blocks, threads>>>(d_A, d_B, d_C, m, n, p);

  // Copy result back to host
  hipMemcpy(C, d_C, m * n * sizeof(float), hipMemcpyDeviceToHost);

  delete[] A;
  delete[] B;
  delete[] C;

  return 0;
}



//===================================================================================================================
//               CUDA Programming Model
//===================================================================================================================

// Prompt: "Create 3 code suggestions using the following parameters: C++ CUDA General matrix-matrix multiply function."

// NOTE: CUDA uses __global__ or kernel.

// SUGGESTION 1 // ----------------------------------

__global__ void matrixMultiplyKernel(float* A, float* B, float* C, int m, int n, int p) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row >= m || col >= n) return;

  float sum = 0;
  for (int k = 0; k < p; ++k) {
    sum += hipCublasSdot(n, A + k * n, 1, B + k * m, 1);
  }
  C[row * n + col] = sum;
}


// SUGGESTION 2 // ----------------------------------

__global__ void matrixMultiplyKernel(float* A, float* B, float* C, int m, int n, int p) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row >= m || col >= n) return;

  __shared__ float smem[32][32];

  float sum = 0;
  for (int k = 0; k < p; ++k) {
    sum += A[row * p + k] * B[k * n + col];
  }
  smem[threadIdx.x][threadIdx.y] = sum;

  __syncthreads();

  if (threadIdx.x == 0 && threadIdx.y == 0) {
    C[row * n + col] = smem[0][0];
  }
}


// SUGGESTION 3 // ----------------------------------

__global__ void matrixMultiplyKernel(float* A, float* B, float* C, int m, int n, int p) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row >= m || col >= n) return;

  float sum = 0;
  for (int k = 0; k < p; ++k) {
    sum += A[row * p + k * n] * B[k * m + col];
  }
  C[row * n + col] = sum;
}



// ---------------------------------------------------------------------------------------------------------------------------------------------------------
