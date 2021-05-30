//
//  softmax.cu
//
//
//  Created by Andreas Dreyer Hysing on 09/05/2021.
//
#include "softmax.cuh"
#include <cstddef>
#include <vector>
#include <algorithm>
#include <iostream>
#include <numeric>

#if __cplusplus == 201103L
// https://stackoverflow.com/questions/7038357/make-unique-and-perfect-forwarding
template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args)
{
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}
#endif

// https://github.com/mark-poscablo/gpu-sum-reduction/blob/master/sum_reduction/reduce.cu
template<typename T>
void check(T err, const char* const func, const char* const file, const int line)
{
  if (err != cudaSuccess)
  {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
  }
}

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

unsigned int nextPowerOfTwo(unsigned int n)
{
    // Round up to the next highest power of 2
    // In 12 operations, this code computes the next highest power of 2 for a 32-bit integer. The result may be expressed by the formula 1U << (lg(v - 1) + 1). Note that in the edge case where v is 0, it returns 0, which isn't a power of 2; you might append the expression v += (v == 0) to remedy this if it matters. It would be faster by 2 operations to use the formula and the log base 2 method that uses a lookup table, but in some situations, lookup tables are not suitable, so the above code may be best. (On a Athlon™ XP 2100+ I've found the above shift-left and then OR code is as fast as using a single BSR assembly language instruction, which scans in reverse to find the highest set bit.) It works by copying the highest set bit to all of the lower bits, and then adding one, which results in carries that set all of the lower bits to 0 and one bit beyond the highest set bit to 1. If the original number was a power of 2, then the decrement will reduce it to one less, so that we round up to the same original value.
    // You might alternatively compute the next higher power of 2 in only 8 or 9 operations using a lookup table for floor(lg(v)) and then evaluating 1<<(1+floor(lg(v))); Atul Divekar suggested I mention this on September 5, 2010.
    // Devised by Sean Anderson, Sepember 14, 2001. Pete Hart pointed me to a couple newsgroup posts by him and William Lewis in February of 1997, where they arrive at the same algorithm.
    // https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
    unsigned int v = n; // compute the next highest power of 2 of 32-bit v
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

bool detectPowerOfTwo(unsigned int n)
{
    return (n & (n - 1)) == 0;
}

// source: https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
template<unsigned int blockSize>
__inline__ __device__ void warpAdd(volatile double* sharedValues, unsigned int tid)
{
    if (blockSize >= 64)
    {
        sharedValues[tid] += sharedValues[tid + 32];
    }

    if (blockSize >= 32)
    {
        sharedValues[tid] += sharedValues[tid + 16];
    }

    if (blockSize >= 16)
    {
        sharedValues[tid] += sharedValues[tid + 8];
    }

    if (blockSize >= 8)
    {
        sharedValues[tid] += sharedValues[tid + 4];
    }

    if (blockSize >= 4)
    {
        sharedValues[tid] += sharedValues[tid + 2];
    }

    if (blockSize >= 2)
    {
        sharedValues[tid] += sharedValues[tid + 1];
    }
}

template<unsigned int blockSize, bool isPowerOfTwo>
__device__ double exponentialSumKernel(const double* values, unsigned int n)
{
    extern __shared__ double sharedValues[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
    unsigned int gridSize = blockSize * 2 * gridDim.x;

    double partialExponentialSum = 0.0;
    while (i < n)
    {
        partialExponentialSum += exp(values[i]);
        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (isPowerOfTwo || i + blockSize < n)
        {
            partialExponentialSum += exp(values[i + blockSize]);
        }

        i += gridSize;
    }

    sharedValues[tid] = partialExponentialSum;
    __syncthreads();

    if (blockSize >= 2048)
    {
        if (tid < 1024)
        {
            sharedValues[tid] += sharedValues[tid + 1024];
        }

        __syncthreads();
    }

    if (blockSize >= 1024)
    {
        if (tid < 512)
        {
            sharedValues[tid] += sharedValues[tid + 512];
        }

        __syncthreads();
    }

    if (blockSize >= 512)
    {
        if (tid < 256)
        {
            sharedValues[tid] += sharedValues[tid + 256];
        }

        __syncthreads();
    }

    if (blockSize >= 256)
    {
        if (tid < 128)
        {
            sharedValues[tid] += sharedValues[tid + 128];
        }

        __syncthreads();
    }

    if (blockSize >= 128)
    {
        if (tid < 64)
        {
            sharedValues[tid] += sharedValues[tid + 64];
        }

        __syncthreads();
    }

    if (tid < 32)
    {
        warpAdd<blockSize>(sharedValues, tid);
    }

    __syncthreads();
    return sharedValues[0];
}

template<unsigned int blockSize, bool isPowerOfTwo>
__global__ void softMaxKernel(const double* values, double* scaledValues, unsigned int n)
{
    // The isPowerOfTwo optimization does not handle n = 1. So we hard code the answer for n = 1 and return immediately.
    if (isPowerOfTwo && n == 1)
    {
        scaledValues[0] = 1.0;
        return;
    }

    double exponentialSum = exponentialSumKernel<blockSize, isPowerOfTwo>(values, n);

    unsigned int i = threadIdx.x;
    unsigned int gridSize = blockSize * 2 * gridDim.x;
    while (i < n)
    {
        scaledValues[i] = exp(values[i]) / exponentialSum;
        if (isPowerOfTwo || i + blockSize < n)
        {
            scaledValues[i + blockSize] = exp(values[i + blockSize]) / exponentialSum;
        }

        i += gridSize;
    }
}

unsigned int lookupMaxThreadsPerBlock(unsigned int deviceId)
{
    cudaDeviceProp device;
    checkCudaErrors(cudaGetDeviceProperties(&device, deviceId));
    return device.maxThreadsPerBlock;
}

std::unique_ptr<std::vector<double>> softMax(const std::weak_ptr<std::vector<double>> values, unsigned int deviceId, bool verbose)
{
    std::unique_ptr<std::vector<double>> scaledValues = nullptr;
    // cudaDeviceSetCacheConfig() or on a per-kernel basis using cudaFuncSetCacheConfig(). These accept one of three
    double* d_values = nullptr;
    double* d_scaledValues = nullptr;

    auto valuesShared = values.lock();
    if (valuesShared == nullptr)
    {
        std::cerr << "Unable to access weak_ptr. Parent pointer has been deallocated..." << std::endl;
        return std::move(scaledValues);
    } else if (valuesShared->size() == 0)
    {
        return std::move(scaledValues);
    }

    unsigned int n = valuesShared->size();

    checkCudaErrors(cudaMalloc(&d_values, n * sizeof(double)));
    checkCudaErrors(cudaMalloc(&d_scaledValues, n * sizeof(double)));
    checkCudaErrors(cudaMemcpy(d_values, (void*)valuesShared->data(), n * sizeof(double), cudaMemcpyHostToDevice));

    unsigned int maxThreads = lookupMaxThreadsPerBlock(deviceId);

    unsigned int threads = (n < maxThreads * 2) ? nextPowerOfTwo((n + 1) / 2) : maxThreads;
    unsigned int numBlocks = (n + (threads * 2 - 1)) / (threads * 2);
    dim3 dimBlock(threads, 1, 1);   // threads per block
    dim3 dimGrid(numBlocks, 1, 1);  // blocks
    unsigned int sharedMemSize = threads > 64 ? threads * sizeof(double) : 64 * sizeof(double);

    if (verbose)
    {
        std::cout << "threads: " << threads << " blocks: " << numBlocks << std::endl;
    }

    bool isNPowerOfTwo = detectPowerOfTwo(n);
    bool fallbackToCPU = false;
    if (isNPowerOfTwo)
    {
        switch (threads)
        {
            case 2048:
                softMaxKernel<2048, true><<<dimGrid, dimBlock, sharedMemSize>>>(d_values, d_scaledValues, n);
                break;
            case 1024:
                softMaxKernel<1024, true><<<dimGrid, dimBlock, sharedMemSize>>>(d_values, d_scaledValues, n);
                break;
            case 512:
                softMaxKernel<512, true><<<dimGrid, dimBlock, sharedMemSize>>>(d_values, d_scaledValues, n);
                break;
            case 256:
                softMaxKernel<256, true><<<dimGrid, dimBlock, sharedMemSize>>>(d_values, d_scaledValues, n);
                break;
            case 128:
                softMaxKernel<128, true><<<dimGrid, dimBlock, sharedMemSize>>>(d_values, d_scaledValues, n);
                break;
            case 64:
                softMaxKernel<64, true><<<dimGrid, dimBlock, sharedMemSize>>>(d_values, d_scaledValues, n);
                break;
            case 32:
                softMaxKernel<32, true><<<dimGrid, dimBlock, sharedMemSize>>>(d_values, d_scaledValues, n);
                break;
            case 16:
                softMaxKernel<16, true><<<dimGrid, dimBlock, sharedMemSize>>>(d_values, d_scaledValues, n);
                break;
            case 8:
                softMaxKernel<8, true><<<dimGrid, dimBlock, sharedMemSize>>>(d_values, d_scaledValues, n);
                break;
            case 4:
                softMaxKernel<4, true><<<dimGrid, dimBlock, sharedMemSize>>>(d_values, d_scaledValues, n);
                break;
            case  2:
                softMaxKernel<2, true><<<dimGrid, dimBlock, sharedMemSize>>>(d_values, d_scaledValues, n);
                break;
            case  1:
                softMaxKernel<1, true><<<dimGrid, dimBlock, sharedMemSize>>>(d_values, d_scaledValues, n);
                break;
            default:
                fallbackToCPU = true;
        }
    } else {
        switch (threads)
        {
            case 2048:
                softMaxKernel<2048, false><<<dimGrid, dimBlock, sharedMemSize>>>(d_values, d_scaledValues, n);
                break;
            case 1024:
                softMaxKernel<1024, false><<<dimGrid, dimBlock, sharedMemSize>>>(d_values, d_scaledValues, n);
                break;
            case 512:
                softMaxKernel<512, false><<<dimGrid, dimBlock, sharedMemSize>>>(d_values, d_scaledValues, n);
                break;
            case 256:
                softMaxKernel<256, false><<<dimGrid, dimBlock, sharedMemSize>>>(d_values, d_scaledValues, n);
                break;
            case 128:
                softMaxKernel<128, false><<<dimGrid, dimBlock, sharedMemSize>>>(d_values, d_scaledValues, n);
                break;
            case 64:
                softMaxKernel<64, false><<<dimGrid, dimBlock, sharedMemSize>>>(d_values, d_scaledValues, n);
                break;
            case 32:
                softMaxKernel<32, false><<<dimGrid, dimBlock, sharedMemSize>>>(d_values, d_scaledValues, n);
                break;
            case 16:
                softMaxKernel<16, false><<<dimGrid, dimBlock, sharedMemSize>>>(d_values, d_scaledValues, n);
                break;
            case 8:
                softMaxKernel<8, false><<<dimGrid, dimBlock, sharedMemSize>>>(d_values, d_scaledValues, n);
                break;
            case 4:
                softMaxKernel<4, false><<<dimGrid, dimBlock, sharedMemSize>>>(d_values, d_scaledValues, n);
                break;
            case  2:
                softMaxKernel<2, false><<<dimGrid, dimBlock, sharedMemSize>>>(d_values, d_scaledValues, n);
                break;
            case  1:
                softMaxKernel<1, false><<<dimGrid, dimBlock, sharedMemSize>>>(d_values, d_scaledValues, n);
                break;
            default:
                fallbackToCPU = true;
        }
    }

    if (!fallbackToCPU)
    {
        scaledValues = make_unique<std::vector<double>>(n);
        checkCudaErrors(cudaMemcpy((void*)scaledValues->data(), d_scaledValues, n * sizeof(double), cudaMemcpyDeviceToHost));
    }

    checkCudaErrors(cudaFree((void*)d_values));
    checkCudaErrors(cudaFree((void*)d_scaledValues));
    if (fallbackToCPU)
    {
        return std::move(softMaxCPU(values, verbose));
    }

    return std::move(scaledValues);
}

std::unique_ptr<std::vector<double>> softMax(const std::weak_ptr<std::vector<double>> values, bool verbose)
{
    return softMax(values, 0, verbose);
}

std::unique_ptr<std::vector<double>> softMaxCPU(const std::weak_ptr<std::vector<double>> values, bool verbose)
{
    std::unique_ptr<std::vector<double>> scaledValues = nullptr;
    auto valuesShared = values.lock();
    if (valuesShared == nullptr)
    {
        std::cerr << "Unable to access weak_ptr. Parent pointer has been deallocated..." << std::endl;
        return std::move(scaledValues);
    } else if (valuesShared->size() == 0)
    {
        return std::move(make_unique<std::vector<double>>(0));
    }

    std::vector<double> immediate(valuesShared->size());

    auto lambda = [&](double a, double b)
    {
        return a + exp(b);
    };
    double partialExponentialSum = std::accumulate(valuesShared->begin(), valuesShared->end(), 0.0, lambda);

    std::cout << "Partial" << partialExponentialSum << std::endl;
    {
        std::vector<double>::iterator toIt = immediate.begin();
        for(std::vector<double>::iterator it = valuesShared->begin(); it != valuesShared->end() && toIt != immediate.end(); ++it)
        {
            *toIt = exp(*it) / partialExponentialSum;
            ++toIt;
        }
    }

    scaledValues = make_unique<std::vector<double>>(immediate.begin(), immediate.end());
    return std::move(scaledValues);
}
