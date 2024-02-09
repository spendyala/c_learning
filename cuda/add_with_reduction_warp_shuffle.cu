%%writefile add_with_reduction_warp_shuffle.cu
// !nvcc add_with_reduction_warp_shuffle.cu -o add_with_reduction_warp_shuffle
// !./add_with_reduction_warp_shuffle
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

//#define N 104857600
//#define N 1048576
//#define N 1047576
//#define N 1024

#define N 671078400 // The input data size to reduce
#define GRID_SIZE 65535 // Number of blocks in the grid
#define THREADS_PER_BLOCK 1024 // 1024 threads per block

__global__ void reduce_ws(float *gdata, float *out) {
    // __shared__ sdata: Each block of threads has its own shared memory space, and this memory is accessible only by the threads within the same block.
    // Shared memory to store partial sums from each warp
    __shared__ float sdata[32];


    int tid = threadIdx.x; // Local thread ID with in a block
    int idx = threadIdx.x + blockDim.x * blockIdx.x; // Global thread index across the grid
    float val = 0.0f; // To accumulate values by each thread, thread is important here
    unsigned mask = 0xFFFFFFFFU; // Mask used for warp shuffle operations
    int lane = threadIdx.x % warpSize; // Lane ID within a warp (0 to warpSize-1)
    int warpID = threadIdx.x / warpSize; // Warp ID within a block

    // Grid-stride loop: each thread processes multiple elements spaced by the grid size
    while (idx < N) {
        val += gdata[idx]; // Accumulate value into val
        idx += gridDim.x * blockDim.x; // Move to the next element this thread is responsible for
    }

    // 1st warp-shuffle reduction
    // Perform warp-level reduction using shuffle operations
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        // Shuffle down and add across the warp
        val += __shfl_down_sync(mask, val, offset);
    }

    // Store the result of each warp's reduction in shared memory
    if (lane == 0) sdata[warpID] = val;
    // Synchronize threads within the block to ensure all warps have written their results
    __syncthreads(); // put warp results in shared mem

    // hereafter, just warp 0
    // Further reduction within the first warp to combine the warp results
    if (warpID == 0) {
        // Load warp's result or zero if outside
        val = (tid < blockDim.x / warpSize) ? sdata[lane] : 0;
        // final warp-shuffle reduction
        // Final shuffle down and add within the first warp
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(mask, val, offset);
        }

        // First thread of the block updates the global sum atomically
        if (tid == 0) atomicAdd(out, val);
    }
}

int main() {

    // Host and device data pointers and output variable
    float *h_data, *d_data, *d_out, h_out = 0.0f;
    size_t bytes = N * sizeof(float);

    // Allocate memory on the host
    h_data = (float*)malloc(bytes);

    // Initialize host array
    for(int i = 0; i < N; i++) {
        h_data[i] = 1.0f; // Example: all elements are 1.0
    }

    // Allocate memory on the device
    cudaMalloc(&d_data, bytes);
    cudaMalloc(&d_out, sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, &h_out, sizeof(float), cudaMemcpyHostToDevice);

    // Calculate grid and block sizes
//    int blockSize = THREADS_PER_BLOCK;
//    int gridSize = (N + blockSize - 1) / blockSize;

    // Launch the kernel with a fixed grid and block size
    reduce_ws<<<GRID_SIZE, THREADS_PER_BLOCK>>>(d_data, d_out);

    // Copy result back to host
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    printf("Total: %f\n", h_out);

    // Cleanup, free host and device memory
    free(h_data);
    cudaFree(d_data);
    cudaFree(d_out);

    return 0;
}