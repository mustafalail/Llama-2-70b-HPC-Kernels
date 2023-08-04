// OpenMP

// Give me code for the following:
// C++, OpenMP, general matrix-matrix multiply, function


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

// Give me code for the following:
// C++, OpenMP (offload), general matrix-matrix multiply, function

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

// OpenMP (offload)

// Give me code for the following:
// C++, OpenMP (offload), general matrix-matrix multiply, function
