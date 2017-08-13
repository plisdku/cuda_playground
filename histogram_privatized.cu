#include "histogram_privatized.hpp"
#include "utils.hpp"
#include <stdio.h>



__global__
void histogram_privatized_kernel(int arrayLength, const float* x, int numBins, float firstEdge, float lastEdge, int* outHist)
{   
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
    int* d_histograms;
    
    // We want each thread to have its own histogram.  I can stuff them all into one
    // histogram array.  Each thread is also responsible for a certain number of array
    // elements...
    
    numThreads = gridSize*blockSize;
    elemsPerThread = arrayLength/numThreads;
    
    I had gotten this far.  Really you should think through the whole algorithm from scratch.
    
    checkCudaErrors(cudaMalloc(&d_x, arrayLength*sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_histogram, numBins*numThreads*sizeof(int)));
    
    checkCudaErrors(cudaMemcpy(d_x, x, arrayLength*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(d_histogram, 0, numBins*numThreads*sizeof(int)));
    
    // In this method we have each thread run a sub-histogram and then combine.
    // Each thread needs its own memory.  Hmm.
    
    histogram_privatized_kernel<<<gridSize, blockSize>>>(arrayLength, d_x, numBins, firstEdge, lastEdge, d_histogram);
    checkCudaErrors(cudaMemcpy(outHist, d_histogram, numBins*sizeof(float), cudaMemcpyDeviceToHost));
    
    checkCudaErrors(cudaFree(d_x));
    checkCudaErrors(cudaFree(d_histogram));
}
