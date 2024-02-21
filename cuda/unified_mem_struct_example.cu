// %%writefile unified_mem_struct_example.cu
// !nvcc unified_mem_struct_example.cu -o unified_mem_struct_example
// !./unified_mem_struct_example
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>

// Define a simple struct
struct MyStruct {
    int x;
    float y;
    int str_len;
    char* name;
};

__device__ int my_strlen(const char *str) {

    int len = 0;
    while (str[len] != '\0') {
        len++;
    }
    return len;
}

__global__ void myKernel(MyStruct *data, int size, char *d_parivesh) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        data[index].x += 1; // Arbitrary operation on the struct
        data[index].y += 1.0f;
        data[index].str_len = my_strlen(d_parivesh);
        data[index].name = d_parivesh;
    }
}


/*
__global__ void myKernel_2(MyStruct *data, int size) {
  // Not working?????
    char *parivesh = "parivesh"; // This is a pointer within GPU memory not in CPU memory
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        data[index].x += 1; // Arbitrary operation on the struct
        data[index].y += 1.0f;
        data[index].name = parivesh;
    }
}*/

int main() {
    int N = 1024; // Number of elements
    MyStruct *d_data;
    char *init = "veera";
    char *parivesh = "parivesh"; // Why can't we just use local variable in kernel, or declare for each thread.
    char *d_parivesh;

    int *d_length; // Not used


    printf("%s\n\n", init);

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&d_data, N * sizeof(MyStruct));
    cudaMallocManaged(&d_parivesh, (strlen(parivesh)+1)*sizeof(char)); // +1 is important, to have the null character.

    cudaMallocManaged(&d_length, N * sizeof(int));
    memset(d_length, 0, N*sizeof(int)); // Memset example to initialize the d_length.

    cudaMemcpy(d_parivesh, parivesh, strlen(parivesh) + 1, cudaMemcpyHostToDevice);

    // Initialize data on the host or you could do this on the GPU as well
    for (int i = 0; i < N; ++i) {
        d_data[i].x = 1;
        d_data[i].y = 1.0f;
        d_data[i].str_len = 0;
        d_data[i].name = init;
    }

    for (int i = 0; i < N; ++i) {
        printf("%d %f %d %s->", d_data[i].x, d_data[i].y, d_data[i].str_len, d_data[i].name);
    }
    printf("\n\n");


    for (int i = 0; i < N; ++i) {
        printf("%d ->", d_length[i]);
    }
    printf("\n\n");


    // Run kernel on the GPU
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    myKernel<<<numBlocks, blockSize>>>(d_data, N, d_parivesh);


    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    for (int i = 0; i < N; ++i) {
        printf("%d %f %d %s->", d_data[i].x, d_data[i].y, d_data[i].str_len, d_data[i].name);
    }
    printf("\n\n");

    for (int i = 0; i < N; ++i) {
        printf("%d ->", d_length[i]);
    }
    printf("\n\n");

    // Free device memory
    cudaFree(d_data);
    cudaFree(d_parivesh);
    cudaFree(d_length);

    return 0;
}
