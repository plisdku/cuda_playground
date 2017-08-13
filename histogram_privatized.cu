#include "histogram_privatized.hpp"
#include "utils.hpp"
#include <stdio.h>



__global__
void histogram_privatized_kernel(int arrayLength, const float* x, int numBins, float firstEdge, float lastEdge, int* outHist)
{
    // Approaches:
    // 1. One thread per element, writing into a single histogram.  I can figure this out now.
    // 2. One thread per bin, reading the entire array.  I can figure this out now.
    // 3. Something else
    
    // histogram_thread_bin_kernel: each thread handles one bin
    
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (idx >= numBins)
    {
        return;
    }
    
    float binSize = (lastEdge-firstEdge)/numBins;
    
    int myBinCount = 0;
    for (int ii = 0; ii < arrayLength; ii++)
    {
        int iBin = (x[ii]-firstEdge)/binSize;
        if (iBin < 0)
        {
            iBin = 0;
        }
        if (iBin >= numBins)
        {
            iBin = numBins - 1;
        }
        
        if (iBin == idx)
        {
            myBinCount++;
        }
    }
    
    outHist[idx] = myBinCount;
}


void histogram_privatized(int arrayLength, const float* x, int numBins, float firstEdge, float lastEdge, int* outHist, int gridSize, int blockSize)
{
    float* d_x;
    int* d_histogram;
    
    checkCudaErrors(cudaMalloc(&d_x, arrayLength*sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_histogram, numBins*sizeof(int)));
    
    checkCudaErrors(cudaMemcpy(d_x, x, arrayLength*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(d_histogram, 0, numBins*sizeof(int)));
    
    histogram_privatized_kernel<<<gridSize, blockSize>>>(arrayLength, d_x, numBins, firstEdge, lastEdge, d_histogram);
    checkCudaErrors(cudaMemcpy(outHist, d_histogram, numBins*sizeof(float), cudaMemcpyDeviceToHost));
    
    checkCudaErrors(cudaFree(d_x));
    checkCudaErrors(cudaFree(d_histogram));
}
