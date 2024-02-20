%%writefile unified_mem_example.cu
// !nvcc unified_mem_example.cu -o unified_mem_example
// !./unified_mem_example
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>

/*
The memset function in both standard C and CUDA takes the value to set as an int,
but it actually sets each byte of the block of memory to that value.
This is straightforward when you're setting memory to 0, as all bytes
 are simply set to 0.

However, when you attempt to use memset to initialize memory with a value other
than 0, such as 1, you might expect each element (for example, each integer if
you're working with an array of integers) to become 1. But what actually happens
is that each byte of memory is set to the value 1. This is not typically what you want,
 especially for types larger than a byte, like integers or floats in CUDA.

For example, if you have an array of integers and you use memset(data, 1, size);,
 instead of setting each integer in the array to 1, you're actually setting each
 byte of the array to 1. Since an integer is typically 4 bytes, this would result
 in each integer being set to 0x01010101 in hexadecimal, which is 16843009 in decimal, not 1.

In CUDA, if you want to initialize an array with a value other than 0, you generally
 need to write a kernel to do so, or use a library function designed for arrays,
 such as thrust::fill if you are using the Thrust library. Here's a simple example
 of how you might write a kernel to initialize an array with the value 1:
*/

__global__ void setValue(int* ptr, int index, int val ){
    ptr[index] = val;
}

__global__ void setDoubleVal(int* ptr){
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    ptr[tid] *= 1;
}
void foo(int size) {
    int *data;
    cudaMallocManaged(&data, size);
    memset(data, 2, size);
    /*for(int i=0; i< size; i++){
        data[i] = 1;
        //printf("%d->", data[i]);
    }*/
    printf("\n");
    //setValue<<<...>>>(data, size/2, 5);
    //setValue<<<1, 1024>>>(data, 3, 5);
    setDoubleVal<<<1, 1024>>>(data);
    cudaDeviceSynchronize();
    for(int i=0; i< size; i++){
        printf("%d->", data[i]);
    }
    printf("\n");
    cudaFree(data);
}

int main(){
    foo(1024);
    return 0;
}