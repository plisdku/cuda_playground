#include "utils.h"

void myCudaCaller(int arrayLength, float a, const float* x, const float* y, float* out, int gridSize, int blockSize)
{
    float* d_x;
    float* d_y;
    float* d_out;
    
    checkCudaErrors(cudaMalloc(&d_x, arrayLength));
    checkCudaErrors(cudaMalloc(&d_y, arrayLength));
    checkCudaErrors(cudaMalloc(&d_out, arrayLength));
    checkCudaErrors(cudaMemcpy(d_x, x, arrayLength, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_y, y, arrayLength, cudaMemcpyHostToDevice));
    myCudaKernel<<<gridSize, blockSize>>>(arrayLength, a, d_x, d_y, d_out);
    checkCudaErrors(cudaMemcpy(out, d_out, arrayLength, cudaMemcpyDeviceToHost));
    
    checkCudaErrors(cudaFree(d_x));
    checkCudaErrors(cudaFree(d_y));
    checkCudaErrors(cudaFree(d_out));
    
    checkCudaErrors(cudaDeviceSynchronize()); // only if I need it...
//    checkCudaErrors(cudaGetLastError()); // another approach
}

__global__
void myCudaKernel(int arrayLength, float a, const float* x, const float* y, float* out)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (index < arrayLength)
    {
        out[index] = a*x[index] + y[index];
    }
}


