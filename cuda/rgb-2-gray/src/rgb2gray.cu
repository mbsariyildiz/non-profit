#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

/*
 * kernel function to be run parallel on device
 */
__global__
void rgb2gray_kernel(const uchar3 *rgb,
                   uchar1 *gray,
                   const int n_rows,
                   const int n_cols
                   ){
    // maximum index accessible in the image
    const int img_size = n_rows * n_cols;

    int pixelIdx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pixelIdx_y = blockIdx.y * blockDim.y + threadIdx.y;

    // be careful not to exceed image boundary
    if (pixelIdx_x < n_cols && pixelIdx_y < n_rows){

        int pixelIdx = pixelIdx_y * n_cols + pixelIdx_x;

        // gray = .299f * red + .587f * geen + .114f * blue;
        gray[pixelIdx].x = (unsigned char)(
                    0.299 * rgb[pixelIdx].x +
                    0.587 * rgb[pixelIdx].y +
                    0.114 * rgb[pixelIdx].z );
    }
}

void rgb2gray_caller(const uchar3 *rgb,
                     uchar1 *gray,
                     const int n_rows,
                     const int n_cols){
    
    // n_rows = n_threads_y
    // n_cols = n_threads_x
    const int n_threads_per_dim = 32;
    
    // number of blocks along x axis
    const int n_blocks_x = n_cols % n_threads_per_dim == 0 ? n_cols / n_threads_per_dim : n_cols / n_threads_per_dim + 1 ;
    const int n_blocks_y = n_rows % n_threads_per_dim == 0 ? n_rows / n_threads_per_dim : n_rows / n_threads_per_dim + 1 ;

    std::cout << "n_threads_per_dim: " << n_threads_per_dim << std::endl;
    std::cout << "n_threads_per_block: " << n_threads_per_dim * n_threads_per_dim << std::endl;
    std::cout << "n_blocks_x: " << n_blocks_x << ", ";
    std::cout << "n_blocks_y: " << n_blocks_y << std::endl;

    const dim3 grid(n_blocks_x, n_blocks_y, 1);
    const dim3 block(n_threads_per_dim, n_threads_per_dim, 1);

    rgb2gray_kernel<<<grid, block>>>(rgb, gray, n_rows, n_cols);
}
