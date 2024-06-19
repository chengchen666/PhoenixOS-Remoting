#include <cuda_runtime.h>
#include <cudnn.h>
#include <stdio.h>
#include <chrono>
#include <iostream>

int main() {
  const int iterations = 100000;
  // cudnnTensorDescriptor_t x_desc;
  // cudnnFilterDescriptor_t x_desc;
  cudnnConvolutionDescriptor_t x_desc;

  for (int i = 0; i < 10; ++i) {
    // cudnnCreateTensorDescriptor(&x_desc);
    // cudnnDestroyTensorDescriptor(x_desc);
    // cudnnCreateFilterDescriptor(&x_desc);
    // cudnnDestroyFilterDescriptor(x_desc);
    cudnnCreateConvolutionDescriptor(&x_desc);
    cudnnDestroyConvolutionDescriptor(x_desc);
  }

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; ++i) {
    // cudnnCreateTensorDescriptor(&x_desc);
    // cudnnDestroyTensorDescriptor(x_desc);
    // cudnnCreateFilterDescriptor(&x_desc);
    // cudnnDestroyFilterDescriptor(x_desc);
    cudnnCreateConvolutionDescriptor(&x_desc);
    cudnnDestroyConvolutionDescriptor(x_desc);
  }
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> elapsed = end - start;
  double totalElapsedTime = elapsed.count();
  double averageElapsedTime = totalElapsedTime / iterations;
  std::cout << "Average elapsed time: " << averageElapsedTime << " ms" << std::endl;

  return 0;
}
