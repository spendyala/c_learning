// %%writefile unified_mem_strlen_example.cu
// !nvcc unified_mem_strlen_example.cu -o unified_mem_strlen_example
// !./unified_mem_strlen_example
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>


__device__ int my_strlen(const char *str) {
    int len = 0;
    while (str[len] != '\0') {
        len++;
    }
    return len;
}


__global__ void myKernel_3(int *d_length, int size, char *d_parivesh) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        d_length[index] = my_strlen(d_parivesh);
    }
}

/*
 * due to the incorrect assignment of d_parivesh in host code before kernel execution.
 * Instead of directly assigning parivesh to d_parivesh, you should copy the content
 * from host string parivesh to d_parivesh using cudaMemcpy to ensure the string resides
 * in unified memory accessible by the GPU. The current assignment d_parivesh = parivesh;
 * simply assigns a host memory address to d_parivesh, which does not copy the string
 * data into unified memory space accessible by the device.
 *
 */

int main() {
    int N = 1024; // Number of elements
    char *init = "veera";
    char *parivesh = "parivesh";
    char *d_parivesh;
    int *d_length;

    // Allocate Unified Memory
    cudaMallocManaged(&d_parivesh, (strlen(parivesh) + 1) * sizeof(char));
    cudaMallocManaged(&d_length, N * sizeof(int));
    //memset(d_length, 0, N * sizeof(int));
    memset(d_length, 0, N);

    for (int i = 0; i < N; ++i) {
        printf("%d ->", d_length[i]);
    }
    printf("\n\n");

    // Copy string to device memory
    cudaMemcpy(d_parivesh, parivesh, strlen(parivesh) + 1, cudaMemcpyHostToDevice);

    // Run kernel
    int blockSize = 256;
    myKernel_3<<<(N + blockSize - 1) / blockSize, blockSize>>>(d_length, N, d_parivesh);
    cudaDeviceSynchronize();

    for (int i = 0; i < N; ++i) {
        printf("%d ->", d_length[i]);
    }
    printf("\n\n");

    // Free device memory
    cudaFree(d_parivesh);
    cudaFree(d_length);

    return 0;
}
