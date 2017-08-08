//Udacity HW1 Solution

#include <iostream>
#include <vector>

#include "playground.hpp"


int main(int argc, char **argv)
{
    int numElems = 1000;
    
    float a = 3.0;
    std::vector<float> x(numElems);
    std::vector<float> y(numElems);
    std::vector<float> z(numElems);
    
    int gridSize = 10;
    int blockSize = 100;
    myCudaCaller(numElems, a, x.data(), y.data(), z.data(), gridSize, blockSize);
    
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    
    return 0;
}
