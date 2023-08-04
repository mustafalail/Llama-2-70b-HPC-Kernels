// OpenMP

// Give me code for the following:
// C++, OpenMP, general matrix-vector multiply, function

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


// -----------------------------------------------------------------------------------------------------------

// OpenMP (offload)

// Give me code for the following:
// C++, OpenMP (offload), general matrix-vector multiply, function

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

// -----------------------------------------------------------------------------------------------------------

// OpenMP (offload)

// Give me code for the following:
// C++, OpenMP (offload), general matrix-vector multiply, function

