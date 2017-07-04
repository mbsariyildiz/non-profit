#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

bool allocate (void **ptr, size_t size){
    // since cudaMalloc accepts double pointer, casting is required.
    cudaError_t stat = cudaMalloc(ptr, size);

    if (stat == cudaSuccess)
        return true;

    // if no success, print the error
    std::cout << "allocation stat: " <<  stat << std::endl;
    return false;
}

/*
enum cudaMemcpyKind:
cudaMemcpyHostToHost = 0
cudaMemcpyHostToDevice = 1
cudaMemcpyDeviceToHost = 2
cudaMemcpyDeviceToDevice = 3
cudaMemcpyDefault = 4
 */
bool copy(void *dst, void *src, size_t size, int cudaCpyKind){
    cudaMemcpyKind dir;
    switch (cudaCpyKind) {
    case 0:
        dir = cudaMemcpyHostToHost;
        break;
    case 1:
        dir = cudaMemcpyHostToDevice;
        break;
    case 2:
        dir = cudaMemcpyDeviceToHost;
        break;
    case 3:
        dir = cudaMemcpyDeviceToDevice;
        break;
    default:
        dir = cudaMemcpyHostToHost;
        break;
    }
    cudaError_t stat = cudaMemcpy(dst, src, size, dir);

    if (stat == cudaSuccess)
        return true;

    // if no success, print the error
    std::cout << "copy stat: " <<  stat << std::endl;
    return false;
}

void release(void *ptr){ cudaFree(ptr);}
