//===================================================================================================================
//               OpenMP Programming Model
//===================================================================================================================


// Prompt: "Create 3 code suggestions using the following parameters: C++ OpenMP General matrix-vector multiply function."

// SUGGESTION 1 // ----------------------------------

#include <iostream>
#include <omp.h>

void matrixVectorMultiply(double* A, double* B, double* C, int m, int n) {
    #pragma omp parallel num_threads(4) default(none) shared(A, B, C, m, n)
    {
        // Each thread computes its own block of the result vector
        double* C_thread = C + (omp_get_thread_num() * n);
        for (int i = 0; i < m; ++i) {
            #pragma omp simd
            for (int j = 0; j < n; ++j) {
                C_thread[j] += A[i * n + j] * B[i];
            }
        }
    }
}


// SUGGESTION 2 // ----------------------------------

#include <iostream>
#include <omp.h>

void matrixVectorMultiply(double* A, double* B, double* C, int m, int n) {
    #pragma omp taskwait
    {
        // Create tasks for each element of the result vector
        std::vector<std::future<double>> futures;
        for (int i = 0; i < m; ++i) {
            futures.push_back(std::async(std::launch::async, [&] {
                double sum = 0;
                for (int j = 0; j < n; ++j) {
                    sum += A[i * n + j] * B[j];
                }
                return sum;
            }));
        }
        // Wait for all tasks to complete
        for (auto& future : futures) {
            C[future.get()] = future.get();
        }
    }
}


// SUGGESTION 3 // ----------------------------------

#include <iostream>
#include <omp.h>

void matrixVectorMultiply(double* A, double* B, double* C, int m, int n) {
    #pragma omp parallel num_threads(4) default(none) shared(A, B, C, m, n)
    {
        // Compute each element of the result vector in parallel
        #pragma omp for schedule(static) nowait
        for (int i = 0; i < m; ++i) {
            #pragma omp atomic update
            C[i] += A[i * n] * B[0];
            for (int j = 1; j < n; ++j) {
                #pragma omp atomic update
                C[i] += A[i * n + j] * B[j];
            }
        }
    }
}



//===================================================================================================================
//               OpenMP(offload) Programming Model
//===================================================================================================================


// Prompt: "Create 3 code suggestions using the following parameters: C++ OpenMP(offload) General matrix-vector multiply function."

// SUGGESTION 1 // ----------------------------------

#include <iostream>
#include <omp.h>
#include <cusparse.h>

// Matrix-vector multiply function
void matVecMul(const double* A, const double* x, double* y, int m, int n) {
    // Initialize CUDA context and matrices
    cudaContext_t ctx;
    cusparseHandle_t handle;
    cusparseMatDescr_t descr;
    cuMatrix_t A_dev, x_dev, y_dev;
    checkCudaErrors(cusparseCreate(&handle));
    checkCudaErrors(cusparseSetStream(handle, 0));
    checkCudaErrors(cusparseCreateMatDescr(&descr, CUSPARSE_MATRIX_TYPE_GENERAL, m, n, n, n, CUSPARSE_INDEX_BASE_ZERO));
    checkCudaErrors(cuMatrixAlloc(&A_dev, &descr, m, n));
    checkCudaErrors(cuMatrixAlloc(&x_dev, &descr, m, n));
    checkCudaErrors(cuMatrixAlloc(&y_dev, &descr, m, n));
    checkCudaErrors(cusparseSetPointers(handle, A_dev, x_dev, y_dev, descr));

    // Offload data to device
    checkCudaErrors(cuMemcpyHtoD(A_dev, A, m * n * sizeof(double)));
    checkCudaErrors(cuMemcpyHtoD(x_dev, x, m * n * sizeof(double)));

    // Perform matrix-vector multiplication
    #pragma omp parallel num_threads(4) default(none) shared(A_dev, x_dev, y_dev, m, n)
    {
        // Each thread performs a portion of the matrix-vector multiplication
        int tid = omp_get_thread_num();
        for (int i = tid; i < m; i += 4) {
            for (int j = 0; j < n; ++j) {
                double sum = 0;
                for (int k = 0; k < n; ++k) {
                    sum += A[i * n + k] * x[k * n + j];
                }
                y[i * n + j] = sum;
            }
        }
    }

    // Synchronize threads
    #pragma omp barrier

    // Copy result back to host
    checkCudaErrors(cuMemcpyDtoH(y, y_dev, m * n * sizeof(double)));

    // Free resources
    checkCudaErrors(cusparseDestroyMatDescr(descr));
    checkCudaErrors(cuMatrixFree(A_dev));
    checkCudaErrors(cuMatrixFree(x_dev));
    checkCudaErrors(cuMatrixFree(y_dev));
    checkCudaErrors(cusparseDestroy(handle));
}


// SUGGESTION 2 // ----------------------------------

#include <iostream>
#include <omp.h>
#include <cuda_runtime.h>

void matrixVectorMultiply(double* A, double* x, double* y, int m, int n) {
    // Initialize CUDA context and matrices
    cudaContext_t ctx;
    cusparseHandle_t handle;
    cusparseMatDescr_t descr;
    cuMatrix_t A_dev, x_dev, y_dev;
    checkCudaErrors(cusparseCreate(&handle));
    checkCudaErrors(cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_DEVICE));
    checkCudaErrors(cusparseCreateMatrix(&descr, m, n, CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_INDEX_BASE_ZERO, handle));
    checkCudaErrors(cusparseSetArray(&A_dev, descr, A, m * n));
    checkCudaErrors(cusparseSetArray(&x_dev, descr, x, n));
    checkCudaErrors(cusparseSetArray(&y_dev, descr, y, m));

    // Offload computation to device
    #pragma omp parallel num_threads(8) default(none) shared(A_dev, x_dev, y_dev)
    {
        // Each thread computes a subset of the matrix-vector product
        int tid = omp_get_thread_num();
        int beg = tid * (m / 8);
        int end = (tid == 7) ? m : (beg + (m / 8) - 1);
        double sum = 0;
        for (int i = beg; i <= end; ++i) {
            for (int j = 0; j < n; ++j) {
                sum += A_dev[i * n + j] * x_dev[j];
            }
        }
        y_dev[tid] = sum;
    }

    // Wait for all threads to finish
    #pragma omp barrier

    // Copy results back to host
    checkCudaErrors(cusparseGetArray(&y, descr, y_dev, m));

    // Free resources
    checkCudaErrors(cusparseDestroyMatrix(&descr));
    checkCudaErrors(cusparseDestroyHandle(&handle));
}


// SUGGESTION 3 // ----------------------------------

#include <iostream>
#include <omp.h>
#include <cuda_runtime.h>

void matrixVectorMultiply(double* A, double* x, double* y, int m, int n) {
    // Initialize CUDA context and matrices
    cudaContext_t ctx;
    cusparseHandle_t handle;
    cusparseMatDescr_t descr;
    cuMatrix_t A_dev, x_dev, y_dev;
    checkCudaErrors(cusparseCreate(&handle));
    checkCudaErrors(cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_DEVICE));
    checkCudaErrors(cusparseCreateMatrix(&descr, m, n, CUSPARSE_MATRIX_TYPE_GENERAL, CUSPARSE_INDEX_BASE_ZERO, handle));
    checkCudaErrors(cusparseSetArray(&A_dev, descr, A, m * n));
    // Set up shared memory for threads
    __shared__ double smem[32][32];

    // Thread block size
    int BLOCK_SIZE = 32;

    // Number of threads per block
    int THREADS_PER_BLOCK = BLOCK_SIZE * BLOCK_SIZE;

    // Total number of blocks
    int NUM_BLOCKS = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Iterate over blocks
    for (int b = 0; b < NUM_BLOCKS; ++b) {
        // Thread ID within block
        int t = threadIdx.x + threadIdx.y * BLOCK_SIZE;

        // If thread ID is outside block boundaries, do nothing
        if (t >= BLOCK_SIZE || t >= m) break;

        // Load values from global memory into shared memory
        smem[threadIdx.y][threadIdx.x] = A[b * BLOCK_SIZE + threadIdx.x];
        smem[threadIdx.y][threadIdx.x + BLOCK_SIZE] = A[(b + 1) * BLOCK_SIZE + threadIdx.x];

        // Synchronize threads within block
        __syncthreads();

        // Compute dot product for current block
        double dotProduct = 0;
        for (int i = 0; i < n; ++i) {
            dotProduct += smem[threadIdx.y][i] * x[i];
        }

        // Store dot product back to global memory
        y[b * BLOCK_SIZE + threadIdx.x] = dotProduct;

        // Synchronize threads across blocks
        __barrier();
    }

    // Free resources
    checkCudaErrors(cusparseDestroyMatrix(&descr));
    checkCudaErrors(cusparseDestroyHandle(&handle));
}



//===================================================================================================================
//               OpenACC Programming Model
//===================================================================================================================


// Prompt: "Create 3 code suggestions using the following parameters: C++ OpenACC General matrix-vector multiply function."

// SUGGESTION 1 // ----------------------------------

#include <iostream>
#include <acc.h>

void matrixVectorMultiply(float *A, float *x, float *y, int m, int n) {
  #pragma acc parallel num_gangs(1) gang_size(32) default_device(GPU:0)
  {
    int tid = acc_get_processor_id() + acc_get_thread_id() * 32;
    if (tid >= m) return;
    float dotProduct = 0;
    #pragma acc loop independent collapse(2)
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < m; ++j) {
        dotProduct += A[tid + j * 32] * x[i];
      }
    }
    y[tid] = dotProduct;
  }
}


// SUGGESTION 2 // ----------------------------------

#include <iostream>
#include <acc.h>

void matrixVectorMultiply(float *A, float *x, float *y, int m, int n, int sA, int sx) {
  #pragma acc parallel num_gangs(1) gang_size(32) default_device(GPU:0)
  {
    int tid = acc_get_processor_id() + acc_get_thread_id() * 32;
    if (tid >= m) return;
    float dotProduct = 0;
    #pragma acc loop independent collapse(2)
    for (int i = 0; i < n; i++) {
      const float *Ap = &A[tid + i * sA];
      const float *xp = &x[i * sx];
      for (int j = 0; j < m; j++, Ap += sA, xp += sx) {
        dotProduct += Ap[0] * xp[0] + Ap[1] * xp[1] + Ap[2] * xp[2] + Ap[3] * xp[3];
      }
      y[tid + i * sx] = dotProduct;
    }
  }
}


// SUGGESTION 3 // ----------------------------------

#include <iostream>
#include <acc.h>

void matrixVectorMultiply(float *A, float *x, float *y, int m, int n) {
  // Divide the input matrix and vector into tiles
  int tileSize = 16;
  int numTilesX = (m + tileSize - 1) / tileSize;
  int numTilesY = (n + tileSize - 1) / tileSize;
  float *A_tiles[numTilesX][numTilesY];
  float *x_tiles[numTilesX][numTilesY];
  float *y_tiles[numTilesX][numTilesY];

  // Allocate memory for the tiles
  #pragma acc enter data create(A_tiles, x_tiles, y_tiles)
  {
    for (int i = 0; i < numTilesX; ++i) {
      for (int j = 0; j < numTilesY; ++j) {
        A_tiles[i][j] = (float *)malloc(tileSize * tileSize * sizeof(float));
        x_tiles[i][j] = (float *)malloc(tileSize * sizeof(float));
        y_tiles[i][j] = (float *)malloc(tileSize * sizeof(float));
      }
    }
  }

  // Initialize the tiles
  #pragma acc parallel num_gangs(numTilesX) gang_size(numTilesY) default_device(GPU:0)
  {
    int tile_idx = acc_get_processor_id();
    int i = tile_idx / numTilesY;
    int j = tile_idx % numTilesY;
    for (int k = 0; k < tileSize; ++k) {
      for (int l = 0; l < tileSize; ++l) {
        A_tiles[i][j][k * tileSize + l] = A[k * m + l];
        x_tiles[i][j][k * tileSize + l] = x[k * n + l];
      }
    }
  }

  // Perform matrix-vector multiplication
  #pragma acc parallel num_gangs(numTilesX) gang_size(numTilesY) default_device(GPU:0)
  {
    int tile_idx = acc_get_processor_id();
    int i = tile_idx / numTilesY;
    int j = tile_idx % numTilesY;
    float dotProduct = 0;
    for (int k = 0; k < tileSize; ++k) {
      for (int l = 0; l < tileSize; ++l) {
        dotProduct += A_tiles[i][j][k * tileSize + l] * x_tiles[i][j][k * tileSize + l];
      }
    }
    y_tiles[i][j] = dotProduct;
  }

  // Gather the results
  #pragma acc synchronize
  {
    for (int i = 0; i < numTilesX; ++i) {
      for (int j = 0; j < numTilesY; ++j) {
        y[i * tileSize + j * tileSize] = y_tiles[i][j];
      }
    }
  }

  // Free the memory allocated for the tiles
  #pragma acc leave data delete(A_tiles, x_tiles, y_tiles)
  {
    for (int i = 0; i < numTilesX; ++i) {
      for (int j = 0; j < numTilesY; ++j) {
        free(A_tiles[i][j]);
        free(x_tiles[i][j]);
        free(y_tiles[i][j]);
      }
    }
  }
}



//===================================================================================================================
//                HIP Programming Model
//===================================================================================================================

// HIP(Heterogeneous-Compute Interface for Portability)

// Prompt: "Create 3 code suggestions using the following parameters: C++ HIP(Heterogeneous-Compute Interface for Portability) General matrix-vector multiply function."

// SUGGESTION 1 // ----------------------------------

#include <hip/hip_runtime.h>

__global__ void matrixVecMultiply(const float* A, const float* x, float* y, int m, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= m) return;
  float sum = 0;
  for (int i = 0; i < n; ++i) {
    sum += A[idx * n + i] * x[i];
  }
  y[idx] = sum;
}

void launchKernel(int m, int n, const float* A, const float* x, float* y) {
  int threadsPerBlock = 256;
  int blocksPerGrid = (m + threadsPerBlock - 1) / threadsPerBlock;
  hipLaunchKernelGGL(matrixVecMultiply, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, 0, A, x, y, m, n);
}


// SUGGESTION 2 // ----------------------------------

#include <hip/hip_runtime.h>

__global__ void matrixVecMultiplyStrided(const float* A, const float* x, float* y, int m, int n, int strideA, int stridex) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= m) return;
  float sum = 0;
  for (int i = 0; i < n; ++i) {
    sum += A[idx * strideA + i * stridex] * x[i];
  }
  y[idx] = sum;
}

void launchKernelStrided(int m, int n, const float* A, const float* x, float* y, int strideA, int stridex) {
  int threadsPerBlock = 256;
  int blocksPerGrid = (m + threadsPerBlock - 1) / threadsPerBlock;
  hipLaunchKernelGGL(matrixVecMultiplyStrided, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, 0, A, x, y, m, n, strideA, stridex);
}


// SUGGESTION 3 // ----------------------------------

#include <hip/hip_runtime.h>

__shared__ float smem[256];

__global__ void matrixVecMultiplyShared(const float* A, const float* x, float* y, int m, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= m) return;
  float sum = 0;
  for (int i = 0; i < n; ++i) {
    sum += A[idx * n + i] * x[i];
  }
  smem[threadIdx.x] = sum;
  __syncthreads();
  if (threadIdx.x == 0) {
    y[idx] = smem[0];
  }
}

void launchKernelShared(int m, int n, const float* A, const float* x, float* y) {
  int threadsPerBlock = 256;
  int blocksPerGrid = (m + threadsPerBlock - 1) / threadsPerBlock;
  hipLaunchKernelGGL(matrixVecMultiplyShared, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, 0, A, x, y, m, n);
}

//===================================================================================================================
//               CUDA Programming Model
//===================================================================================================================

// Prompt: "Create 3 code suggestions using the following parameters: C++ CUDA General matrix-vector multiply function."

// NOTE: CUDA uses __global__ or kernel.

// SUGGESTION 1 // ----------------------------------

__global__ void matrixVectorMultiply(float* A, float* x, float* y, int m, int n) {
  // Use shared memory to reduce global memory access
  __shared__ float smem[32][32];
  
  // Compute row of A and x in parallel
  int row = blockIdx.x;
  int col = threadIdx.x;
  float sum = 0;
  for (int i = 0; i < n; i++) {
    sum += A[row * n + i] * x[col + i * 32];
  }
  
  // Store result in shared memory
  smem[row][col] = sum;
  
  // Synchronize threads within a block
  __syncthreads();
  
  // Write result from shared memory to global memory
  if (threadIdx.x == 0) {
    y[row * n + col] = smem[row][col];
  }
}


// SUGGESTION 2 // ----------------------------------

__global__ void matrixVectorMultiply(float* A, float* x, float* y, int m, int n) {
  // Use register blocking to improve memory access patterns
  register float rA[32];
  register float rx[32];
  
  // Load A and x into registers
  for (int i = 0; i < 32; i++) {
    rA[i] = A[i];
    rx[i] = x[i];
  }
  
  // Compute dot product in parallel
  int row = blockIdx.x;
  int col = threadIdx.x;
  float sum = 0;
  for (int i = 0; i < n; i++) {
    sum += rA[row * n + i] * rx[col + i * 32];
  }
  
  // Store result in global memory
  y[row * n + col] = sum;
}


// SUGGESTION 3 // ----------------------------------

__global__ void matrixVectorMultiply(float* A, float* x, float* y, int m, int n) {
  // Use strided access to optimize memory access patterns
  float* Ap = A + blockIdx.x * n;
  float* xp = x + threadIdx.x * 32;
  
  // Compute dot product in parallel
  int row = blockIdx.x;
  int col = threadIdx.x;
  float sum = 0;
  for (int i = 0; i < n; i++) {
    sum += Ap[row * n + i] * xp[col + i * 32];
  }
  
  // Store result in global memory
  y[row * n + col] = sum;
}



// ---------------------------------------------------------------------------------------------------------------------------------------------------------
