#include <cuda_runtime.h>

#include <chrono>
#include <iostream>

int main() {
  long long iter = 100;
  auto tp1 = std::chrono::steady_clock::now();
  for (long long i = 0; i < iter; i++) {
    // if (i % 10 == 0) {
    //     std::cout << i << std::endl;
    // }
    int n = 16 * 1024 * 1024;
    char *h_data = (char *)malloc(n);
    char *d_data;

    for (int i = 0; i < n; i++) {
      h_data[i] = (char)(i % 128);
    }

    cudaMalloc((void **)&d_data, n * sizeof(int));

    // sanity check
    cudaMemcpy(d_data, h_data, n, cudaMemcpyHostToDevice);
    char *h_data2 = (char *)calloc(n, sizeof(char));
    cudaMemcpy(h_data2, d_data, n, cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) {
      if (h_data[i] != h_data2[i]) {
        std::cout << "mismatch at " << i << std::endl;
        return -1;
      }
    }

    cudaFree(d_data);
  }

  auto tp2 = std::chrono::steady_clock::now();
  std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(tp2 - tp1)
                   .count()
            << " milliseconds" << std::endl;
  return 0;
}