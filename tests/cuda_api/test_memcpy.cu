#include <chrono>
#include <cuda_runtime.h>
#include <iostream>

int main()
{
    const int iterations = 1;

    int n = 16*1024*1024;
    char *h_data = (char*)malloc(n);
    char *d_data;

    for (int i = 0; i < n; i++) {
        h_data[i] = (char) (i % 128);
    }

    cudaMalloc((void**)&d_data, n * sizeof(int));
    std::cout << "ptr: " << (long)d_data << std::endl;

    // // remove initial overhead
    // for (int i = 0; i < 10; i++) {
    //     cudaMemcpy(d_data, h_data, n, cudaMemcpyHostToDevice);
    // }
    
    // double totalElapsedTime = 0.0;
    // // Start the timer
    // auto start = std::chrono::high_resolution_clock::now();
    // for (int i = 0; i < iterations; ++i) {
    //     cudaMemcpy(d_data, h_data, n, cudaMemcpyHostToDevice);
    // }
    // // Stop the timer
    // auto end = std::chrono::high_resolution_clock::now();
    // // Calculate the elapsed time in milliseconds
    // std::chrono::duration<double, std::milli> elapsed = end - start;
    // totalElapsedTime += elapsed.count();

    // // Calculate the average elapsed time
    // double averageElapsedTime = totalElapsedTime / iterations;

    // std::cout << "Average elapsed time: " << averageElapsedTime << " ms" << std::endl;

    return 0;
}
