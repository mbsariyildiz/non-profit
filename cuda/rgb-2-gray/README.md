### C++ progam that converts an rgb image to gray-scale on both device (GPU) and host (CPU).

It is tested on Ubuntu 16.04 with NVIDIA-GeForce 940Mx by using Cuda-8.0.

The program loads a view of Atlantic Ocean:
![RGB Atlantic](https://github.com/mbsariyildiz/non-profit/blob/master/cuda/rgb-2-gray/resources/atlantic.jpeg)

then converts it to gray-scale by using both CPU and GPU. Below is the one converted on CPU:
![Gray-scale on CPU](https://github.com/mbsariyildiz/non-profit/blob/master/cuda/rgb-2-gray/resources/grayscale_cpu.png)

And the below is converted on GPU:
![Gray-scale on GPU](https://github.com/mbsariyildiz/non-profit/blob/master/cuda/rgb-2-gray/resources/grayscale_gpu.png)

Mean absolute difference (MAD) between them is ~1e-6, which is I believe reasonable.