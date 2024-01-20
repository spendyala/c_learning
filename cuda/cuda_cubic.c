//
// Created by Subbu Pendyala on 1/19/24.
//
// %%cu
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


int* cube_host(int *a, int len, int *c){
    for(int i=0; i<len; i++){
        c[i] = a[i] * a[i] * a[i];
    }
    return c;
}

__global__ void cube_device(int *a, int *c) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    c[i] = a[i] * a[i] * a[i];
}

int random(int min, int max){
    return min + rand() / (RAND_MAX / (max - min + 1) + 1);
}

int main() {
    clock_t start_t, end_t, start_t_k, end_t_k;
    double total_t;
    int *h_a, *h_c, *h_d;
    int N = 1000000000;
    int *d_a, *d_c;
    int BLOCK_SIZE=1024, GRID_SIZE=65535;

    // Allocating memory on the host
    h_a = (int *)malloc(N*sizeof(int));
    for(int i=0; i<N; i++){
        h_a[i]= random(0, 100);
    }

    for(int i=0; i<3; i++){
        printf("%d -> ", h_a[i]);
    }
    printf("\n");

    // Allocating memory on the device
    cudaMalloc((void **)&d_a, N*sizeof(int));
    cudaMalloc((void **)&d_c, N*sizeof(int));
    // Copy the data to the device

    start_t = clock();
    cudaMemcpy(d_a, h_a, N*sizeof(int), cudaMemcpyHostToDevice);
    // Launch the kernel
    start_t_k = clock();
    cube_device<<<GRID_SIZE, BLOCK_SIZE>>>(d_a, d_c);
    end_t_k = clock();
    total_t = (double)(end_t_k - start_t_k)/ CLOCKS_PER_SEC;
    printf("\n Device Kernel Time %f\n", total_t);

    h_c = (int *)malloc(N*sizeof(int));

    // Copy the result back to the host
    cudaMemcpy(h_c, d_c, N*sizeof(int), cudaMemcpyDeviceToHost);
    end_t = clock();
    total_t = (double)(end_t - start_t)/ CLOCKS_PER_SEC;
    printf("\n Device Time %f\n", total_t);

    // Print the result
    for(int i=0; i<3; i++){
        printf("%d -> ", h_c[i]);
    }
    printf("\n");

    // Free the device memory
    cudaFree(d_a);
    cudaFree(d_c);
    free(h_c);

    h_d = (int *)malloc(N*sizeof(int));
    start_t = clock();
    h_d = cube_host(h_a, N, h_d);
    end_t = clock();
    total_t = (double)(end_t - start_t)/ CLOCKS_PER_SEC;
    // Print the result
    for(int i=0; i<3; i++){
        printf("%d -> ", h_d[i]);
    }
    printf("\n CPU Time %f\n", total_t);

    free(h_a);
    free(h_d);


    return 0;
}