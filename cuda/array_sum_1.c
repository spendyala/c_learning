// %%writefile array_sum_1.cu
// !nvcc array_sum_1.cu -o array_sum_1
// !./array_sum_1
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


#define BLOCK_SIZE 1024
#define GRID_SIZE 65535
#define N BLOCK_SIZE*GRID_SIZE


int sum_array(int *h_a, int len)
{
    int sum = 0;
    for(int i=0; i< len; i++){
        sum += h_a[i];
    }
    return sum;
}


__global__ void gpu_sum_array(int *d_in, int *res_sum){
    __shared__ int partial_sum;

    int global_id = threadIdx.x + blockDim.x * blockIdx.x; // Global index, threadIdx.x is local

    if (threadIdx.x == 0){
        partial_sum = 0;
    }
    __syncthreads();

    atomicAdd(&partial_sum, d_in[global_id]);
    __syncthreads();

    if (threadIdx.x == 0) {
        atomicAdd(res_sum, partial_sum);
    }

}
/*
Every thread should update the sum seen so far for it's block

Cache needed should be partial sums of the shared memory.

0, 1, 2 ... 31 Thread IDS from the same Block example block_1
All these threads go and retrive elements from d_a and update temp[block_1]

Given thread ID we should load the entire d_a to shared memory

*/


int main() {
    int *h_in, *h_sum;

    h_in = (int *)malloc(N*sizeof(int));
    h_sum = (int *)malloc(sizeof(int));

    for(int i=0; i<N; i++){
        h_in[i] = 1;
    }

    printf("CPU Sum: %d\n", sum_array(h_in, N));


    // GPU
    int *d_in, *d_sum;
    cudaMalloc((void **)&d_in, N*sizeof(int));
    cudaMalloc((void **)&d_sum, sizeof(int));
    cudaMemcpy(d_in, h_in, N*sizeof(int), cudaMemcpyHostToDevice);

    // Kernal Code
    gpu_sum_array<<<GRID_SIZE, BLOCK_SIZE>>>(d_in, d_sum);

    cudaMemcpy(d_sum, h_sum, sizeof(int), cudaMemcpyDeviceToHost);


    printf("GPU Sum: %d\n", d_sum[0]);


    cudaFree(d_in); cudaFree(d_sum);
    free(h_in); free(h_sum);

    return 0;
}