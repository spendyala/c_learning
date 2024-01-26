//%%writefile array_sum.cu
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 1024*65535



int sum_array(int *h_a, int len)
{
    int sum = 0;
    for(int i=0; i< len; i++){
        sum += h_a[i];
    }
    return sum;
}


__global__ void gpu_sum_array_1(int *d_a, int len, int *sum, int *init){
    __shared__ int temp;
    init[blockIdx.x] = temp;

    int i = threadIdx.x + blockDim.x * blockIdx.x; // Global index, threadIdx.x is local

    /*while(i < blockDim.x+threadIdx.x){ // Iterate over all the threads of a given block and update temp.
      temp += d_a[i]; // d_a is global memory, sum all the
      i++;
    }*/
    temp += d_a[i];

    __syncthreads();
    sum[blockIdx.x] = temp;
}
/*
Every thread should update the sum seen so far for it's block

Cache needed should be partial sums of the shared memory.

0, 1, 2 ... 31 Thread IDS from the same Block example block_1
All these threads go and retrive elements from d_a and update temp[block_1]

Given thread ID we should load the entire d_a to shared memory

*/


int main() {
    int *h_a, *h_sum, *h_init;
    int BLOCK_SIZE=1024, GRID_SIZE=65535;

    h_a = (int *)malloc(N*sizeof(int));
    h_sum = (int *)malloc(GRID_SIZE*sizeof(int));
    h_init = (int *)malloc(GRID_SIZE*sizeof(int));

    for(int i=0; i<N; i++){
        h_a[i] = 1;
    }

    printf("CPU Sum: %d\n", sum_array(h_a, N));


    // GPU
    int *d_a, *d_sum, *d_init;
    cudaMalloc((void **)&d_a, N*sizeof(int));
    cudaMalloc((void **)&d_sum, GRID_SIZE*sizeof(int));
    cudaMalloc((void **)&d_init, GRID_SIZE*sizeof(int));
    cudaMemcpy(d_a, h_a, N*sizeof(int), cudaMemcpyHostToDevice);

    // Kernal Code
    gpu_sum_array_1<<<GRID_SIZE, BLOCK_SIZE>>>(d_a, N, d_sum, d_init);

    cudaMemcpy(h_sum, d_sum, GRID_SIZE*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_init, d_init, GRID_SIZE*sizeof(int), cudaMemcpyDeviceToHost);

    for(int i=0; i<10; i++){
        printf("Init: %d\n", h_init[i]);
    }
    printf("\n");

    int temp = 0;
    for(int i=0; i<GRID_SIZE; i++){
        temp += h_sum[i];
    }
    printf("GPU Sum: %d\n", temp);


    cudaFree(d_a); cudaFree(d_sum); cudaFree(d_init);
    free(h_a); free(h_sum); free(h_init);

    return 0;
}

/*

12 34 50 100 - Host
Copy to device


assign each number to one thread

[block1] [block2]
all the thread in the block, sum: Called as Block SUM

All blocks to finish

and sum all the blocks

DRAM

*/
