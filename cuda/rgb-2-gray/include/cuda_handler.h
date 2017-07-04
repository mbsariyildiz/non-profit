#ifndef CUDA_HANDLER_H
#define CUDA_HANDLER_H

#include <cstdio>

bool allocate(void **ptr, size_t size);

/*
enum cudaMemcpyKind:
cudaMemcpyHostToHost = 0
    Host -> Host
cudaMemcpyHostToDevice = 1
    Host -> Device
cudaMemcpyDeviceToHost = 2
    Device -> Host
cudaMemcpyDeviceToDevice = 3
    Device -> Device
cudaMemcpyDefault = 4
 */
bool copy(void *dst, void *src, size_t size, int cudaCpyKind);

void release(void *ptr);

#endif // CUDA_HANDLER_H
