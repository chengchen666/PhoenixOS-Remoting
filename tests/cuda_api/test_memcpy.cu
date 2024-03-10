#include <chrono>
#include <cuda_runtime.h>
#include <iostream>

int main()
{
    // const int iterations = 1;

    int n = 16*1024*1024;
    char *h_data = (char*)malloc(n);
    char *d_data;

    for (int i = 0; i < n; i++) {
        h_data[i] = (char) (i % 128);
    }

    cudaMalloc((void**)&d_data, n * sizeof(int));
    std::cout << "ptr: " << (long)d_data << std::endl;

    // sanity check
    cudaMemcpy(d_data, h_data, n, cudaMemcpyHostToDevice);
    char *h_data2 = (char*)calloc(n, sizeof(char));
    std::cout << (int)h_data2[0] << " " << (int)h_data2[1] << " " << (int)h_data2[2] << " " << (int)h_data2[3] << std::endl;
    cudaMemcpy(h_data2, d_data, n, cudaMemcpyDeviceToHost);
    std::cout << (int)h_data2[0] << " " << (int)h_data2[1] << " " << (int)h_data2[2] << " " << (int)h_data2[3] << std::endl;
    for (int i = 0; i < n; i++) {
        if (h_data[i] != h_data2[i]) {
            std::cout << "mismatch at " << i << std::endl;
            return -1;
        }
    }

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
