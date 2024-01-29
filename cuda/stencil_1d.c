//%%writefile stencil_1d.cu
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 1024*65535
#define RADIUS 3
#define BLOCK_SIZE 1024
#define GRID_SIZE 65535

__global__ void stencil_1d(int *d_in, int *d_out, int *d_shared){
    __shared__ int temp[BLOCK_SIZE + 2 * RADIUS];
    int gindex = threadIdx.x + blockIdx.x * blockDim.x;
    int lindex = threadIdx.x + RADIUS;

    // Read input elements into shared memory
    temp[lindex] = d_in[gindex];
    if (threadIdx.x < RADIUS) {
        temp[lindex - RADIUS] = d_in[gindex - RADIUS];
        temp[lindex + BLOCK_SIZE] = d_in[gindex + BLOCK_SIZE];
    }

    // Synchronize (ensure all the data is available)
    __syncthreads();

    // Apply the stencil
    int result = 0;
    for (int offset = -1*RADIUS; offset <= RADIUS; offset++){
        // 0 1 2 3 4  Index 2 -3 = -1
        // 1 2 3 4 5
        // RADIUS 2
        // offset -1*RADIUS [1, 2] fall in this range
        // offset           [3] is the offset
        // offset + RADIUS  [4, 5] fall in the right side of radius.
        // we can assume offset is the current local index

        result += temp[lindex+offset];
    }
    // Store the result
    d_out[gindex] = result;

    __syncthreads();

    if(threadIdx.x==0){
        for(int idx=0; idx<(BLOCK_SIZE + 2 * RADIUS); idx++){
            d_shared[idx + blockIdx.x * (BLOCK_SIZE + 2 * RADIUS)] = temp[idx];
        }
    }
}


int main() {
    int *h_in, *h_out, *h_shared;

    // TODO: Get the size correct
    h_in = (int *)malloc(N*sizeof(int));
    h_out = (int *)malloc(N*sizeof(int));
    h_shared = (int *)malloc(GRID_SIZE*(BLOCK_SIZE+2*RADIUS)*sizeof(int));


    for(int i=0; i<N; i++){
        h_in[i] = i;
    }

    for(int i=0; i<15; i++){
        printf("%d -> ", h_in[i]);
    }
    printf("\n");


    // GPU
    int *d_in, *d_out, *d_shared;
    cudaMalloc((void **)&d_in, N*sizeof(int));
    cudaMalloc((void **)&d_out, N*sizeof(int));
    cudaMalloc((void **)&d_shared, GRID_SIZE*(BLOCK_SIZE+2*RADIUS)*sizeof(int));

    cudaMemcpy(d_in, h_in, N*sizeof(int), cudaMemcpyHostToDevice);

    // Kernal Code
    stencil_1d<<<GRID_SIZE, BLOCK_SIZE>>>(d_in, d_out, d_shared);

    cudaMemcpy(h_out, d_out, GRID_SIZE*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_shared, d_shared, GRID_SIZE*(BLOCK_SIZE+2*RADIUS)*sizeof(int), cudaMemcpyDeviceToHost);

    printf("\n");
    for(int i=0; i<15; i++){
        printf("%d -> ", h_shared[i]);
    }
    printf("\n");
    printf("Block2\n");
    for(int i=(BLOCK_SIZE+2*RADIUS); i<(BLOCK_SIZE+2*RADIUS)+15; i++){
        printf("%d -> ", h_shared[i]);
    }
    printf("\n");

    printf("\n");

    for(int i=0; i<15; i++){
        printf("%d -> ", h_out[i]);
    }
    printf("\n");


    cudaFree(d_in); cudaFree(d_out); cudaFree(d_shared);
    free(h_in); free(h_out); free(h_shared);

    return 0;
}