#ifndef _HISTOGRAM_NAIVE_HPP_
#define _HISTOGRAM_NAIVE_HPP_

void histogram_thread_element(int arrayLength, const float* x, int numBins, float firstEdge, float lastEdge, int* outHist, int gridSize, int blockSize);
void histogram_thread_bin(int arrayLength, const float* x, int numBins, float firstEdge, float lastEdge, int* outHist, int gridSize, int blockSize);

#endif
