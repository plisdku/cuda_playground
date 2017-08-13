//Udacity HW1 Solution

#include <iostream>
#include <vector>

#include "utils.hpp"
#include "playground.hpp"
#include "histogram_naive.hpp"


void test_saxpy();
void test_scan();
void test_histogram();

int main(int argc, char **argv)
{
//     test_saxpy();
//     test_scan();
    test_histogram();
    return 0;
}


void test_saxpy()
{
    int numElems = 1000;
    
    float a = 3.0;
    std::vector<float> x(numElems);
    std::vector<float> y(numElems);
    std::vector<float> z(numElems);
    
    for (int nn = 0; nn < 10; nn++)
    {
        x[nn] = nn;
        y[nn] = -nn;
    }
    
    int gridSize = 10;
    int blockSize = 100;
    saxpy(numElems, a, x.data(), y.data(), z.data(), gridSize, blockSize);
    
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    
    std::cout << "x\ty\tz\n";
    for (int nn = 0; nn < 10; nn++)
    {
        std::cout << x[nn] << "\t" << y[nn] << "\t" << z[nn] << "\n";
    }
    
    // Now test it
    for (int nn = 0; nn < numElems; nn++)
    {
        float expected = a*x[nn] + y[nn];
        
        if (z[nn] != expected)
        {
            std::cerr << "Error at index " << nn << ": got z = " << z[nn] << ", expected " << expected << ".\n";
        }
    }
}

void test_scan()
{
    int numElems = 1000;
    
    std::vector<float> x(numElems);
    std::vector<float> y(numElems);
    
    for (int nn = 0; nn < numElems; nn++)
    {
        x[nn] = nn;
    }
    
    int gridSize = 10;
    int blockSize = 100;
    prefix_sum(numElems, x.data(), y.data(), gridSize, blockSize);
    
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    
    std::cout << "x\ty\n";
    for (int nn = 0; nn < 10; nn++)
    {
        std::cout << x[nn] << "\t" << y[nn] << "\n";
    }
    
    // Now test it:
    float total = 0.0;
    for (int nn = 0; nn < numElems; nn++)
    {
        total += x[nn];
        
        if (total != y[nn])
        {
            std::cerr << "Error at index " << nn << ": got y = " << y[nn] << ", expected " << total << ".\n";
        }
    }   
}

void test_histogram()
{
    int numElems = 10000;
    
    int numBins = 10;
    float firstEdge = 0.0;
    float lastEdge = 10.0;
    
    std::vector<float> x(numElems);
    std::vector<int> histogramValues(numBins);
    
    for (int nn = 0; nn < numElems; nn++)
    {
        x[nn] = nn/3.0 + 0.5;
    }
    
    int gridSize = 10;
    int blockSize = 100;
    histogram_thread_bin(x.size(), x.data(), numBins, firstEdge, lastEdge, histogramValues.data(), gridSize, blockSize);
    
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    
    std::cout << "x\n";
    for (int nn = 0; nn < 10; nn++)
    {
        std::cout << x[nn] <<  "\n";
    }
    
    float binSize = (lastEdge-firstEdge)/numBins;
    std::cout << "histogram\n";
    for (int nn = 0; nn < numBins; nn++)
    {
        float binLeft = firstEdge + nn*binSize;
        float binRight = binLeft + binSize;
        
        std::cout << binLeft << " to " << binRight << ": " << histogramValues.at(nn) << " values\n";
    } 
}