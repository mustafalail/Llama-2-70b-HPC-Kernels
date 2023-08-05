// OpenMP

// Prompt: "Give me code for the following:
// C++, OpenMP, AXPY, function"

// VERSION 1 // 

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


// VERSION 2 // 

#include <iostream>
#include <omp.h>

void axpy(float* x, float* y, int n) {
  #pragma omp parallel num_threads(4) default(none) shared(x, y, n)
  {
    int tid = omp_get_thread_num();
    int start = tid * (n / 4);
    int end = (tid == 3) ? n : (start + (n / 4));
    #pragma omp for schedule(static) nowait
    for (int i = start; i < end; ++i) {
      x[i] += y[i];
    }
  }
}

int main() {
  const int N = 1024;
  float* x = new float[N];
  float* y = new float[N];
  for (int i = 0; i < N; i++) {
    x[i] = i / 2.0f;
    y[i] = i + 1.0f;
  }
  axpy(x, y, N);
  for (int i = 0; i < N; i++) {
    std::cout << x[i] << " ";
  }
  std::cout << std::endl;
  delete[] x;
  delete[] y;
  return 0;
}


// VERSION 3 //

#include <iostream>
#include <vector>
#include <thread>
#include <mutex>

void axpy(float* x, float* y, int n) {
  std::vector<float> vx(n);
  std::vector<float> vy(n);
  for (int i = 0; i < n; i++) {
    vx[i] = x[i];
    vy[i] = y[i];
  }
  std::mutex mtx;
  auto t1 = std::thread([&vx, &vy, n, mtx] {
    for (int i = 0; i < n; i++) {
      vx[i] += vy[i];
    }
  });
  auto t2 = std::thread([&vx, &vy, n, mtx] {
    for (int i = n / 2; i < n; i++) {
      vx[i] += vy[i];
    }
  });
  auto t3 = std::thread([&vx, &vy, n, mtx] {
    for (int i = n / 3; i < n / 2; i++) {
      vx[i] += vy[i];
    }
  });
  auto t4 = std::thread([&vx, &vy, n, mtx] {
    for (int i = 0; i < n / 3; i++) {
      vx[i] += vy[i];
    }
  });
  t1.join();
  t2.join();
  t3.join();
  t4.join();
  for (int i = 0; i < n; i++) {
    x[i] = vx[i];
  }
}

int main() {
  const int N = 1024;
  float* x = new float[N];
  float* y = new float[N];
  for (int i = 0; i < N; i++) {
    x[i] = i / 2.0f;
    y[i] = i + 1.0f;
  }
  axpy(x, y, N);
  for (int i = 0; i < N; i++) {
    std::cout << x[i] << " ";
  }
  std::cout << std::endl;
  delete[] x;
  delete[] y;
  return 0;
}


// --------------------------------------------------------------------------------------------------------------------------------------


// OpenMP (offload)

// Prompt: "Give me code for the following:
// C++, OpenMP (offload), AXPY, function"

// VERSION 1 //

#include <iostream>
#include <omp.h>

// Declare the AXPY function
void axpy(double *x, double *y, int n) {
  #pragma omp parallel num_threads(4) default(none) shared(x, y, n)
  {
    int i = omp_get_thread_num();
    // Compute the dot product for each element in parallel
    double sum = 0;
    for (int j = 0; j < n; j++) {
      sum += x[j] * y[j];
    }
    // Offload the computation to the GPU
    #pragma omp task out(sum)
    {
      // Perform the dot product on the GPU
      // ...
      // Return the result
      return sum;
    }
    // Wait for the task to complete
    #pragma omp taskwait
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


// VERSION 2 //

#include <iostream>
#include <omp.h>

void axpy(float* x, float* y, int n) {
  #pragma omp parallel num_threads(4) default(none) shared(x, y, n)
  {
    int tid = omp_get_thread_num();
    int start = tid * (n / 4);
    int end = (tid == 3) ? n : (start + (n / 4));
    #pragma omp for schedule(static) nowait
    for (int i = start; i < end; ++i) {
      x[i] += y[i];
    }
  }
}

int main() {
  const int N = 1024;
  float* x = new float[N];
  float* y = new float[N];
  for (int i = 0; i < N; i++) {
    x[i] = i / 2.0f;
    y[i] = i + 1.0f;
  }
  axpy(x, y, N);
  for (int i = 0; i < N; i++) {
    std::cout << x[i] << " ";
  }
  std::cout << std::endl;
  delete[] x;
  delete[] y;
  return 0;
}


// VERSION 3 //

#include <iostream>
#include <vector>
#include <omp.h>

void axpy(std::vector<float>& x, std::vector<float>& y, int n) {
  #pragma omp parallel num_threads(4) default(none) shared(x, y, n)
  {
    int tid = omp_get_thread_num();
    int start = tid * (n / 4);
    int end = (tid == 3) ? n : (start + (n / 4));
    #pragma omp for schedule(static) nowait
    for (int i = start; i < end; ++i) {
      x[i] += y[i];
    }
  }
}

int main() {
  const int N = 1024;
  std::vector<float> x(N);
  std::vector<float> y(N);
  for (int i = 0; i < N; i++) {
    x[i] = i / 2.0f;
    y[i] = i + 1.0f;
  }
  axpy(x, y, N);
  for (int i = 0; i < N; i++) {
    std::cout << x[i] << " ";
  }
  std::cout << std::endl;
  return 0;
}


// --------------------------------------------------------------------------------------------------------------------------------------


// HIP (Heterogeneous-Compute Interface for Portability)

// Prompt: "Give me code for the following:
// C++, Heterogeneous-Compute Interface for Portability, AXPY, function"


// VERSION 1 //

#include <hip/hip_runtime.h>
#include <iostream>

// Define the kernel function
__global__ void axpyKernel(float* x, float* y, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    x[idx] += y[idx];
  }
}

// Define the host function
void axpy(float* x, float* y, int n) {
  // Create a HIP context and command queue
  hipContext_t context;
  hipCommandQueue_t queue;
  hipInit(&context, &queue);

  // Allocate memory on the device
  float* dev_x;
  float* dev_y;
  hipMalloc((void**)&dev_x, n * sizeof(float));
  hipMalloc((void**)&dev_y, n * sizeof(float));

  // Copy data from host to device
  hipMemcpy(dev_x, x, n * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(dev_y, y, n * sizeof(float), hipMemcpyHostToDevice);

  // Launch the kernel
  int blockSize = 256;
  int gridSize = (n + blockSize - 1) / blockSize;
  axpyKernel<<<gridSize, blockSize>>>(dev_x, dev_y, n);

  // Copy data from device to host
  hipMemcpy(x, dev_x, n * sizeof(float), hipMemcpyDeviceToHost);
  hipMemcpy(y, dev_y, n * sizeof(float), hipMemcpyDeviceToHost);

  // Free memory on the device
  hipFree(dev_x);
  hipFree(dev_y);

  // Destroy the HIP context and command queue
  hipDestroyContext(context);
  hipDestroyCommandQueue(queue);
}

int main() {
  const int N = 1024;
  float* x = new float[N];
  float* y = new float[N];
  for (int i = 0; i < N; i++) {
    x[i] = i / 2.0f;
    y[i] = i + 1.0f;
  }
  axpy(x, y, N);
  for (int i = 0; i < N; i++) {
    std::cout << x[i] << " ";
  }
  std::cout << std::endl;
  delete[] x;
  delete[] y;
  return 0;
}


// VERSION 2 //

#include <hip/hip_runtime.h>
#include <iostream>

__global__ void axpy(float* x, float* y, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    x[idx] += y[idx];
  }
}

int main() {
  const int N = 1024;
  float* x = new float[N];
  float* y = new float[N];
  for (int i = 0; i < N; i++) {
    x[i] = i / 2.0f;
    y[i] = i + 1.0f;
  }
  hipLaunchKernelGGL(axpy, dim3(128), dim3(128), 0, 0, x, y, N);
  for (int i = 0; i < N; i++) {
    std::cout << x[i] << " ";
  }
  std::cout << std::endl;
  delete[] x;
  delete[] y;
  return 0;
}

// VERSION 3 // 

#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

__global__ void axpy(float* x, float* y, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    x[idx] += y[idx];
  }
}

int main() {
  const int N = 1024;
  std::vector<float> x(N);
  std::vector<float> y(N);
  for (int i = 0; i < N; i++) {
    x[i] = i / 2.0f;
    y[i] = i + 1.0f;
  }
  hipLaunchKernelGGL(axpy, dim3(128), dim3(128), 0, 0, x, y, N);
  for (int i = 0; i < N; i++) {
    std::cout << x[i] << " ";
  }
  std::cout << std::endl;
  return 0;
}

// --------------------------------------------------------------------------------------------------------------------------------------


// OpenACC

// Prompt: "Give me code for the following:
// C++, OpenACC, AXPY, function"

// VERSION 1 //

#include <openacc.h>
#include <iostream>

// Define the kernel function
__kernel void axpyKernel(float* x, float* y, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    x[idx] += y[idx];
  }
}

// Define the host function
void axpy(float* x, float* y, int n) {
  // Initialize OpenACC
  acc_init();

  // Create a buffer on the device
  float* dev_x;
  float* dev_y;
  acc_create_buffer(sizeof(float)*n, &dev_x, &dev_y);

  // Copy data from host to device
  acc_copyin(x, dev_x, n);
  acc_copyin(y, dev_y, n);

  // Set up the kernel parameters
  dim3 blockSize(256);
  dim3 gridSize((n + blockSize.x - 1) / blockSize.x);

  // Launch the kernel
  axpyKernel<<<gridSize, blockSize>>>(dev_x, dev_y, n);

  // Copy data from device to host
  acc_copyout(dev_x, x, n);
  acc_copyout(dev_y, y, n);

  // Free the buffer on the device
  acc_free(dev_x);
  acc_free(dev_y);

  // Shut down OpenACC
  acc_shutdown();
}

int main() {
  const int N = 1024;
  float* x = new float[N];
  float* y = new float[N];
  for (int i = 0; i < N; i++) {
    x[i] = i / 2.0f;
    y[i] = i + 1.0f;
  }
  axpy(x, y, N);
  for (int i = 0; i < N; i++) {
    std::cout << x[i] << " ";
  }
  std::cout << std::endl;
  delete[] x;
  delete[] y;
  return 0;
}


// VERSION 2 //

#include <openacc.h>
#include <iostream>

void axpy(float* x, float* y, int n) {
  #pragma acc data present(x, y)
  #pragma acc parallelism num_gangs(1) num_workers(128)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
      x[idx] += y[idx];
    }
  }
}

int main() {
  const int N = 1024;
  float* x = new float[N];
  float* y = new float[N];
  for (int i = 0; i < N; i++) {
    x[i] = i / 2.0f;
    y[i] = i + 1.0f;
  }
  axpy(x, y, N);
  for (int i = 0; i < N; i++) {
    std::cout << x[i] << " ";
  }
  std::cout << std::endl;
  delete[] x;
  delete[] y;
  return 0;
}

// VERSION 3 // 

#include <openacc.h>
#include <iostream>
#include <vector>

void axpy(std::vector<float>& x, std::vector<float>& y, int n) {
  #pragma acc data present(x, y)
  #pragma acc parallelism num_gangs(1) num_workers(128)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
      x[idx] += y[idx];
    }
  }
}

int main() {
  const int N = 1024;
  std::vector<float> x(N);
  std::vector<float> y(N);
  for (int i = 0; i < N; i++) {
    x[i] = i / 2.0f;
    y[i] = i + 1.0f;
  }
  axpy(x, y, N);
  for (int i = 0; i < N; i++) {
    std::cout << x[i] << " ";
  }
  std::cout << std::endl;
  return 0;
}

// --------------------------------------------------------------------------------------------------------------------------------------


// CUDA

// Prompt: "Give me code for the following:
// C++, CUDA, AXPY, function"

// VERSION 1 // 

#include <cuda_runtime.h>

__global__ void axpyKernel(float* x, float* y, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    x[idx] += y[idx];
  }
}

void axpy(float* x, float* y, int n) {
  int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  int threadsPerBlock = BLOCK_SIZE;

  // Allocate memory on device
  float* dev_x;
  float* dev_y;
  cudaMalloc((void**)&dev_x, n * sizeof(float));
  cudaMalloc((void**)&dev_y, n * sizeof(float));

  // Copy data from host to device
  cudaMemcpy(dev_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_y, y, n * sizeof(float), cudaMemcpyHostToDevice);

  // Launch kernel
  axpyKernel<<<numBlocks, threadsPerBlock>>>(dev_x, dev_y, n);

  // Copy data from device to host
  cudaMemcpy(x, dev_x, n * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(y, dev_y, n * sizeof(float), cudaMemcpyDeviceToHost);

  // Free memory on device
  cudaFree(dev_x);
  cudaFree(dev_y);
}

int main() {
  const int N = 1024;
  float* x = new float[N];
  float* y = new float[N];
  for (int i = 0; i < N; i++) {
    x[i] = i / 2.0f;
    y[i] = i + 1.0f;
  }
  axpy(x, y, N);
  for (int i = 0; i < N; i++) {
    std::cout << x[i] << " ";
  }
  std::cout << std::endl;
  delete[] x;
  delete[] y;
  return 0;
}


// VERSION 2 //

#include <cuda_runtime.h>

__global__ void axpy(float* x, float* y, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    x[idx] += y[idx];
  }
}

int main() {
  const int N = 1024;
  float* x = new float[N];
  float* y = new float[N];
  for (int i = 0; i < N; i++) {
    x[i] = i / 2.0f;
    y[i] = i + 1.0f;
  }
  int threadsPerBlock = 128;
  int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
  axpy<<<numBlocks, threadsPerBlock>>>(x, y, N);
  for (int i = 0; i < N; i++) {
    std::cout << x[i] << " ";
  }
  std::cout << std::endl;
  delete[] x;
  delete[] y;
  return 0;
}

// VERSION 3 //

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

__global__ void axpy(thrust::device_vector<float>& x, thrust::device_vector<float>& y, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    x[idx] += y[idx];
  }
}

int main() {
  const int N = 1024;
  thrust::host_vector<float> h_x(N);
  thrust::host_vector<float> h_y(N);
  for (int i = 0; i < N; i++) {
    h_x[i] = i / 2.0f;
    h_y[i] = i + 1.0f;
  }
  thrust::device_vector<float> d_x(N);
  thrust::device_vector<float> d_y(N);
  d_x.transfer_from(h_x);
  d_y.transfer_from(h_y);
  int threadsPerBlock = 128;
  int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
  axpy<<<numBlocks, threadsPerBlock>>>(d_x, d_y, N);
  d_x.transfer_to(h_x);
  for (int i = 0; i < N; i++) {
    std::cout << h_x[i] << " ";
  }
  std::cout << std::endl;



  
  return 0;
}
