//%%writefile array_sum.cu
// !nvcc array_sum.cu -o array_sum
// !./array_sum
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


__global__ void gpu_sum_array_1(int *d_a, int len, int *sum){
    __shared__ int temp;

    if (threadIdx.x == 0){
        temp = 0;
    }
    __syncthreads();

    int i = threadIdx.x + blockDim.x * blockIdx.x; // Global index, threadIdx.x is local

    atomicAdd(&temp, d_a[i]);
    __syncthreads();

    sum[blockIdx.x] = temp;

}
/*
 *
 *
 1 2 3 4 5 6 0 0
  3   7   11  0  Block 1
    10      11   Block 1
        21       Block 1


# This is called Divergence.
if condition is True:
    do A()  fast
else:
    do B()  slow

C()


Every thread should update the sum seen so far for it's block

Cache needed should be partial sums of the shared memory.

0, 1, 2 ... 31 Thread IDS from the same Block example block_1
All these threads go and retrive elements from d_a and update temp[block_1]

Given thread ID we should load the entire d_a to shared memory

*/


int main() {
    int *h_a, *h_sum;
    int BLOCK_SIZE=1024, GRID_SIZE=65535;

    h_a = (int *)malloc(N*sizeof(int));
    h_sum = (int *)malloc(GRID_SIZE*sizeof(int));

    for(int i=0; i<N; i++){
        h_a[i] = 1;
    }

    printf("CPU Sum: %d\n", sum_array(h_a, N));


    // GPU
    int *d_a, *d_sum;
    cudaMalloc((void **)&d_a, N*sizeof(int));
    cudaMalloc((void **)&d_sum, GRID_SIZE*sizeof(int));
    cudaMemcpy(d_a, h_a, N*sizeof(int), cudaMemcpyHostToDevice);

    // Kernal Code
    gpu_sum_array_1<<<GRID_SIZE, BLOCK_SIZE>>>(d_a, N, d_sum);

    cudaMemcpy(h_sum, d_sum, GRID_SIZE*sizeof(int), cudaMemcpyDeviceToHost);


    int temp = 0;
    for(int i=0; i<GRID_SIZE; i++){
        temp += h_sum[i];
    }
    printf("GPU Sum: %d\n", temp);


    cudaFree(d_a); cudaFree(d_sum);
    free(h_a); free(h_sum);

    return 0;
}