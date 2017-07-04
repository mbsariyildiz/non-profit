#ifndef RGB2GRAY_KERNEL_H
#define RGB2GRAY_KERNEL_H

#include <cuda_runtime.h>

// signiture for kernel caller
void rgb2gray_caller(const uchar3 *rgb,
                     uchar1 *gray,
                     const int n_rows,
                     const int n_cols);

#endif // RGB2GRAY_KERNEL_H
