#include <cuda_runtime.h>

#include <chrono>
#include <iostream>

int main() {
  long long iter = 10000;
  int device;
  auto tp1 = std::chrono::steady_clock::now();
  for (long long i = 0; i < iter; i++) {
    if (i % 1000 == 0) std::cout << i << std::endl;
    cudaGetDevice(&device);
  }

  auto tp2 = std::chrono::steady_clock::now();
  std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(tp2 - tp1)
                   .count()
            << " milliseconds" << std::endl;
  return 0;
}