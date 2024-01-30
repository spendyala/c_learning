%%writefile reduce_sum_1.cu
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BLOCK_SIZE 1024
#define GRID_SIZE 1024
#define N 1024 * 1024

int sum(int *arr, int num){
    int ans = 0;

    for(int i=0; i<num; i++){
        ans += arr[i];
    }

    return ans;
}

__global__ void reduce_sum_1(int *d_a, int *ans){

    __shared__ int partial_sum[1024];

    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int local_id = threadIdx.x;

    partial_sum[local_id] = d_a[global_id];

    __syncthreads();

    // block level sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (local_id < s) {
            partial_sum[local_id] += partial_sum[local_id + s];
        }
        __syncthreads();
    }

   // Use atomics to update a global variable to store the running sum of blocks
    if (local_id == 0) {
        atomicAdd(ans, partial_sum[0]);
    }
}

int main(){

    int * h_a;
    int *d_ans;


    int *h_ans;
    h_ans = (int *)malloc(sizeof(int));



    int * d_a;


    h_a = (int *)malloc(N*sizeof(int));

    for(int i=0; i<N; i++){
        h_a[i] = 1;
    }

    cudaMalloc((void **)&d_a, N*sizeof(int));
    cudaMalloc((void **)&d_ans, sizeof(int));

    cudaMemcpy(d_a, h_a, N*sizeof(int), cudaMemcpyHostToDevice);

    reduce_sum_1<<<GRID_SIZE, BLOCK_SIZE>>>(d_a, d_ans);
    cudaMemcpy(h_ans, d_ans, sizeof(int), cudaMemcpyDeviceToHost);

    printf("ans: %d\n", *h_ans);
    printf("\n");
    printf("host ans: %d\n", sum(h_a, N));



    //free memory
    cudaFree(d_a);
    free(h_a);
    free(h_ans);
    cudaFree(d_ans);
    return 1;
}