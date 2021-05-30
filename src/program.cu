//
//  program.cpp
//
//
//  Created by Andreas Dreyer Hysing on 05/09/2021.
//
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>
#include "softmax.cuh"
#include "helper_cuda.h"
#include "stdin_reader.h"

#if __cplusplus == 201103L
// https://stackoverflow.com/questions/7038357/make-unique-and-perfect-forwarding
template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args)
{
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}
#else
using std::make_unique;
#endif

void printHelpPages()
{
    std::cout << "Reads numbers from standard in or command line arguments." << std::endl;
    std::cout << "Perform softmax function on those numbers." << std::endl;
    std::cout << "Print the results to screen" << std::endl;
    std::cout << std::endl;
    std::cout << "Usage:" << std::endl;
    std::cout << "softmax [-v] [-h] [--device=...] [--cpu] [numbers]" << std::endl;
    std::cout << "     -h and --help prints this help" << std::endl;
    std::cout << "     -v and --verbose turns on verbose output" << std::endl;
    std::cout << "     --device=<deviceId> use the n-th GPU device. <deviceId> is an index between 0 and the number of CPUs in the system." << std::endl;
    std::cout << "     -c and --cpu perform the computation on CPU." << std::endl; 
}

bool printHelpIfFlagExists(int argc, const char** argv)
{
    for (int i = 1; i < argc; i++)
    {
        const char* currentArg = argv[i];
        if (strncmp(currentArg, "-h", 2) == 0 || strncmp(currentArg, "--help", 6) == 0)
        {
            printHelpPages();
            return true;
        }
    }

    return false;
}

bool detectVerboseFlag(int argc, const char** argv)
{
    for (int i = 1; i < argc; i++)
    {
        const char* currentArg = argv[i];
        if (strncmp(currentArg, "-v", 2) == 0 || strncmp(currentArg, "--verbose", 9) == 0)
        {
            return true;
        }
    }

    return false;
}

bool detectCPUFlag(int argc, const char** argv)
{
    for (int i = 1; i < argc; i++)
    {
        const char* currentArg = argv[i];
        if (strncmp(currentArg, "-c", 2) == 0 || strncmp(currentArg, "--cpu", 5) == 0)
        {
            return true;
        }
    }

    return false;
}

void printArray(const std::unique_ptr<std::vector<double>> valuesPtr)
{
    std::cout << std::setprecision(std::numeric_limits<long double>::digits10 + 1);
    std::vector<double>* values = valuesPtr.get();
    if (values != nullptr)
    {
        for (int i = 0; i < values->size(); i++)
        {
            if (i > 0)
            {
                std::cout << " ";
            }

            std::cout << values->at(i);
        }

        std::cout << std::endl;
    }
}

void printArray(const std::weak_ptr<std::vector<double>> weak)
{
    auto values = weak.lock();
    if (values == nullptr)
    {
        std::cerr << "Failed on printArray(...). weak_ptr was deallocated." << std::endl;
        return;
    }

    auto valuesPtr = values.get();
    std::cout << std::setprecision(std::numeric_limits<long double>::digits10 + 1);
    int at = valuesPtr->size();
    for (int i = 0; i < at; i++)
    {
        if (i > 0)
        {
            std::cout << " ";
        }

        std::cout << valuesPtr->at(i);
    }

    std::cout << std::endl;
}

int main(int argc, const char** argv)
{
    if (printHelpIfFlagExists(argc, argv))
    {
        return EXIT_SUCCESS;
    }

    bool verbose = detectVerboseFlag(argc, argv);
    bool useCPU = detectCPUFlag(argc, argv);
    // parse input numbers
    auto valuesPtr = std::make_shared<std::vector<double>>(argc - 1);
    int at = 0;
    for (int i = 1; i < argc; i++)
    {
        const char* currentArg = argv[i];
        if (strncmp(currentArg, "-v", 2) == 0
         || strncmp(currentArg, "--verbose", 9) == 0
         || strncmp(currentArg, "-h", 2) == 0 
         || strncmp(currentArg, "--help", 6) == 0 
         || strncmp(currentArg, "-c", 2) == 0 
         || strncmp(currentArg, "--cpu", 5) == 0 
         || strncmp(currentArg, "--device", 8) == 0)
        {
            continue;
        }

        char *parseError = nullptr;
        double value = strtod(currentArg, &parseError);
        if (parseError != currentArg)
        {
            (*valuesPtr.get())[at] = value;
            at ++;
            std::cout << value << std::endl;
        } else {
            std::cerr << "Failed parsing command line flag " << currentArg << std::endl;
            return EXIT_FAILURE;
        }
    }

   
    
    auto numValuesParsed = at;
    if (valuesPtr->size() != numValuesParsed)
    {
       valuesPtr->resize(numValuesParsed);
    }

    putValuesFromStdin(valuesPtr, verbose);



    int CUDAdevice;
    if (useCPU || (CUDAdevice = findCudaDevice(argc, argv, verbose)) == -1)
    {
        if (verbose)
        {
            std::cout << "[CPU]" << std::endl;
        }

        auto scaledValues = softMaxCPU(valuesPtr, verbose);
        std::weak_ptr<std::vector<double>> valuesWeakPtr = valuesPtr;
        printArray(std::move(scaledValues));
    } else {
        cudaError_t status;
        size_t nvPrintfSize = 1024 * 1024;
        if ((status = cudaDeviceSetLimit(cudaLimitPrintfFifoSize, nvPrintfSize)) != cudaSuccess)
        {
            std::cerr << "Failed allocating printf buffers in cuda. " << cudaGetErrorString(status) << std::endl;
            return EXIT_FAILURE;
        }

        auto scaledValues = softMax(valuesPtr, CUDAdevice, verbose);
        std::weak_ptr<std::vector<double>> valuesWeakPtr = valuesPtr;
        printArray(std::move(scaledValues));
    }

    return EXIT_SUCCESS;
}
