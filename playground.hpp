#ifndef _PLAYGROUND_HPP_
#define _PLAYGROUND_HPP_

void saxpy(int arrayLength, float a, const float* x, const float* y,
    float* out, int gridSize, int blockSize);
    
void prefix_sum(int arrayLength, const float* x, float* out, int gridSize, int blockSize);
    
#endif