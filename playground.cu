#include "utils.hpp"
#include <stdio.h>

__global__
void saxpy_kernel(int arrayLength, float a, const float* x, const float* y, float* out)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (index < arrayLength)
    {
        out[index] = a*x[index] + y[index];
    }
}

void saxpy(int arrayLength, float a, const float* x, const float* y, float* out, int gridSize, int blockSize)
{
    float* d_x;
    float* d_y;
    float* d_out;
    
    checkCudaErrors(cudaMalloc(&d_x, arrayLength));
    checkCudaErrors(cudaMalloc(&d_y, arrayLength));
    checkCudaErrors(cudaMalloc(&d_out, arrayLength));
    checkCudaErrors(cudaMemcpy(d_x, x, arrayLength, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_y, y, arrayLength, cudaMemcpyHostToDevice));
    saxpy_kernel<<<gridSize, blockSize>>>(arrayLength, a, d_x, d_y, d_out);
    checkCudaErrors(cudaMemcpy(out, d_out, arrayLength, cudaMemcpyDeviceToHost));
    
    checkCudaErrors(cudaFree(d_x));
    checkCudaErrors(cudaFree(d_y));
    checkCudaErrors(cudaFree(d_out));
    
    checkCudaErrors(cudaDeviceSynchronize()); // only if I need it...
//    checkCudaErrors(cudaGetLastError()); // another approach
}




// Prefix sum is an inclusive scan: Hillis & Steele.  This is the easy algorithm.
// Exclusive scan starts with a zero: Blelloch scan.  This is the crazy algorithm.

// Hillis & Steele scan
__global__
void prefix_sum_kernel(int arrayLength, float* data)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (idx >= arrayLength)
    {
        return;
    }
    
    for (int distToLeft = 1; idx - distToLeft >= 0; distToLeft *= 2)
    {
        float tmp = data[idx] + data[idx - distToLeft];
        __syncthreads();
        
        data[idx] = tmp;
        __syncthreads();
        
        // can i possibly really need TWO syncthreads calls per iteration?
    }
}

void prefix_sum(int arrayLength, const float* x, float* out, int gridSize, int blockSize)
{
    float* d_out;
    
    checkCudaErrors(cudaMalloc(&d_out, arrayLength*sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_out, x, arrayLength*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaDeviceSynchronize());
    prefix_sum_kernel<<<gridSize, blockSize>>>(arrayLength, d_out);
    checkCudaErrors(cudaMemcpy(out, d_out, arrayLength*sizeof(float), cudaMemcpyDeviceToHost));
    
    checkCudaErrors(cudaFree(d_out));
}

