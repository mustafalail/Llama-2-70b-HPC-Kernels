// OpenMP

// Prompt: "Give me code for the following:
// C++, OpenMP, AXPY, function"

#include <iostream>
#include <omp.h>

void axpy(double *x, double *y, int n) {
  #pragma omp parallel num_threads(4) default(none) shared(x, y, n)
  {
    int i = omp_get_thread_num();
    // Each thread computes its portion of the dot product
    double sum = 0;
    for (int j = 0; j < n; j++) {
      sum += x[j] * y[j];
    }
    // Synchronize threads
    #pragma omp barrier
    // Combine partial results from all threads
    if (i == 0) {
      std::cout << "Sum: " << sum << std::endl;
    }
  }
}

int main() {
  int n = 10;
  double *x = new double[n];
  double *y = new double[n];
  // Initialize arrays with some values
  for (int i = 0; i < n; i++) {
    x[i] = i / 2.0;
    y[i] = i + 1.0;
  }
  axpy(x, y, n);
  delete[] x;
  delete[] y;
  return 0;
}
