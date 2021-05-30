/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

////////////////////////////////////////////////////////////////////////////////
// These are CUDA Helper functions for initialization and error checking

#ifndef HELPER_CUDA_H
#define HELPER_CUDA_H

#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "helper_string.h"


// Note, it is required that your SDK sample to include the proper header files, please
// refer the CUDA examples for examples of the needed CUDA headers, which may change depending
// on which CUDA functions are used.

// CUDA Runtime error messages
#ifdef __DRIVER_TYPES_H__
static const char *_cudaGetErrorEnum(cudaError_t error)
{
    switch (error)
    {
        case cudaSuccess:
            return "cudaSuccess";

        case cudaErrorMissingConfiguration:
            return "cudaErrorMissingConfiguration";

        case cudaErrorMemoryAllocation:
            return "cudaErrorMemoryAllocation";

        case cudaErrorInitializationError:
            return "cudaErrorInitializationError";

        case cudaErrorLaunchFailure:
            return "cudaErrorLaunchFailure";

        case cudaErrorPriorLaunchFailure:
            return "cudaErrorPriorLaunchFailure";

        case cudaErrorLaunchTimeout:
            return "cudaErrorLaunchTimeout";

        case cudaErrorLaunchOutOfResources:
            return "cudaErrorLaunchOutOfResources";

        case cudaErrorInvalidDeviceFunction:
            return "cudaErrorInvalidDeviceFunction";

        case cudaErrorInvalidConfiguration:
            return "cudaErrorInvalidConfiguration";

        case cudaErrorInvalidDevice:
            return "cudaErrorInvalidDevice";

        case cudaErrorInvalidValue:
            return "cudaErrorInvalidValue";

        case cudaErrorInvalidPitchValue:
            return "cudaErrorInvalidPitchValue";

        case cudaErrorInvalidSymbol:
            return "cudaErrorInvalidSymbol";

        case cudaErrorMapBufferObjectFailed:
            return "cudaErrorMapBufferObjectFailed";

        case cudaErrorUnmapBufferObjectFailed:
            return "cudaErrorUnmapBufferObjectFailed";

        case cudaErrorInvalidHostPointer:
            return "cudaErrorInvalidHostPointer";

        case cudaErrorInvalidDevicePointer:
            return "cudaErrorInvalidDevicePointer";

        case cudaErrorInvalidTexture:
            return "cudaErrorInvalidTexture";

        case cudaErrorInvalidTextureBinding:
            return "cudaErrorInvalidTextureBinding";

        case cudaErrorInvalidChannelDescriptor:
            return "cudaErrorInvalidChannelDescriptor";

        case cudaErrorInvalidMemcpyDirection:
            return "cudaErrorInvalidMemcpyDirection";

        case cudaErrorAddressOfConstant:
            return "cudaErrorAddressOfConstant";

        case cudaErrorTextureFetchFailed:
            return "cudaErrorTextureFetchFailed";

        case cudaErrorTextureNotBound:
            return "cudaErrorTextureNotBound";

        case cudaErrorSynchronizationError:
            return "cudaErrorSynchronizationError";

        case cudaErrorInvalidFilterSetting:
            return "cudaErrorInvalidFilterSetting";

        case cudaErrorInvalidNormSetting:
            return "cudaErrorInvalidNormSetting";

        case cudaErrorMixedDeviceExecution:
            return "cudaErrorMixedDeviceExecution";

        case cudaErrorCudartUnloading:
            return "cudaErrorCudartUnloading";

        case cudaErrorUnknown:
            return "cudaErrorUnknown";

        case cudaErrorNotYetImplemented:
            return "cudaErrorNotYetImplemented";

        case cudaErrorMemoryValueTooLarge:
            return "cudaErrorMemoryValueTooLarge";

        case cudaErrorInvalidResourceHandle:
            return "cudaErrorInvalidResourceHandle";

        case cudaErrorNotReady:
            return "cudaErrorNotReady";

        case cudaErrorInsufficientDriver:
            return "cudaErrorInsufficientDriver";

        case cudaErrorSetOnActiveProcess:
            return "cudaErrorSetOnActiveProcess";

        case cudaErrorInvalidSurface:
            return "cudaErrorInvalidSurface";

        case cudaErrorNoDevice:
            return "cudaErrorNoDevice";

        case cudaErrorECCUncorrectable:
            return "cudaErrorECCUncorrectable";

        case cudaErrorSharedObjectSymbolNotFound:
            return "cudaErrorSharedObjectSymbolNotFound";

        case cudaErrorSharedObjectInitFailed:
            return "cudaErrorSharedObjectInitFailed";

        case cudaErrorUnsupportedLimit:
            return "cudaErrorUnsupportedLimit";

        case cudaErrorDuplicateVariableName:
            return "cudaErrorDuplicateVariableName";

        case cudaErrorDuplicateTextureName:
            return "cudaErrorDuplicateTextureName";

        case cudaErrorDuplicateSurfaceName:
            return "cudaErrorDuplicateSurfaceName";

        case cudaErrorDevicesUnavailable:
            return "cudaErrorDevicesUnavailable";

        case cudaErrorInvalidKernelImage:
            return "cudaErrorInvalidKernelImage";

        case cudaErrorNoKernelImageForDevice:
            return "cudaErrorNoKernelImageForDevice";

        case cudaErrorIncompatibleDriverContext:
            return "cudaErrorIncompatibleDriverContext";

        case cudaErrorPeerAccessAlreadyEnabled:
            return "cudaErrorPeerAccessAlreadyEnabled";

        case cudaErrorPeerAccessNotEnabled:
            return "cudaErrorPeerAccessNotEnabled";

        case cudaErrorDeviceAlreadyInUse:
            return "cudaErrorDeviceAlreadyInUse";

        case cudaErrorProfilerDisabled:
            return "cudaErrorProfilerDisabled";

        case cudaErrorProfilerNotInitialized:
            return "cudaErrorProfilerNotInitialized";

        case cudaErrorProfilerAlreadyStarted:
            return "cudaErrorProfilerAlreadyStarted";

        case cudaErrorProfilerAlreadyStopped:
            return "cudaErrorProfilerAlreadyStopped";

#if __CUDA_API_VERSION >= 0x4000

        case cudaErrorAssert:
            return "cudaErrorAssert";

        case cudaErrorTooManyPeers:
            return "cudaErrorTooManyPeers";

        case cudaErrorHostMemoryAlreadyRegistered:
            return "cudaErrorHostMemoryAlreadyRegistered";

        case cudaErrorHostMemoryNotRegistered:
            return "cudaErrorHostMemoryNotRegistered";
#endif

        case cudaErrorStartupFailure:
            return "cudaErrorStartupFailure";

        case cudaErrorApiFailureBase:
            return "cudaErrorApiFailureBase";
    }

    return "<unknown>";
}
#endif

#ifdef __cuda_cuda_h__
// CUDA Driver API errors
static const char *_cudaGetErrorEnum(CUresult error)
{
    switch (error)
    {
        case CUDA_SUCCESS:
            return "CUDA_SUCCESS";

        case CUDA_ERROR_INVALID_VALUE:
            return "CUDA_ERROR_INVALID_VALUE";

        case CUDA_ERROR_OUT_OF_MEMORY:
            return "CUDA_ERROR_OUT_OF_MEMORY";

        case CUDA_ERROR_NOT_INITIALIZED:
            return "CUDA_ERROR_NOT_INITIALIZED";

        case CUDA_ERROR_DEINITIALIZED:
            return "CUDA_ERROR_DEINITIALIZED";

        case CUDA_ERROR_PROFILER_DISABLED:
            return "CUDA_ERROR_PROFILER_DISABLED";

        case CUDA_ERROR_PROFILER_NOT_INITIALIZED:
            return "CUDA_ERROR_PROFILER_NOT_INITIALIZED";

        case CUDA_ERROR_PROFILER_ALREADY_STARTED:
            return "CUDA_ERROR_PROFILER_ALREADY_STARTED";

        case CUDA_ERROR_PROFILER_ALREADY_STOPPED:
            return "CUDA_ERROR_PROFILER_ALREADY_STOPPED";

        case CUDA_ERROR_NO_DEVICE:
            return "CUDA_ERROR_NO_DEVICE";

        case CUDA_ERROR_INVALID_DEVICE:
            return "CUDA_ERROR_INVALID_DEVICE";

        case CUDA_ERROR_INVALID_IMAGE:
            return "CUDA_ERROR_INVALID_IMAGE";

        case CUDA_ERROR_INVALID_CONTEXT:
            return "CUDA_ERROR_INVALID_CONTEXT";

        case CUDA_ERROR_CONTEXT_ALREADY_CURRENT:
            return "CUDA_ERROR_CONTEXT_ALREADY_CURRENT";

        case CUDA_ERROR_MAP_FAILED:
            return "CUDA_ERROR_MAP_FAILED";

        case CUDA_ERROR_UNMAP_FAILED:
            return "CUDA_ERROR_UNMAP_FAILED";

        case CUDA_ERROR_ARRAY_IS_MAPPED:
            return "CUDA_ERROR_ARRAY_IS_MAPPED";

        case CUDA_ERROR_ALREADY_MAPPED:
            return "CUDA_ERROR_ALREADY_MAPPED";

        case CUDA_ERROR_NO_BINARY_FOR_GPU:
            return "CUDA_ERROR_NO_BINARY_FOR_GPU";

        case CUDA_ERROR_ALREADY_ACQUIRED:
            return "CUDA_ERROR_ALREADY_ACQUIRED";

        case CUDA_ERROR_NOT_MAPPED:
            return "CUDA_ERROR_NOT_MAPPED";

        case CUDA_ERROR_NOT_MAPPED_AS_ARRAY:
            return "CUDA_ERROR_NOT_MAPPED_AS_ARRAY";

        case CUDA_ERROR_NOT_MAPPED_AS_POINTER:
            return "CUDA_ERROR_NOT_MAPPED_AS_POINTER";

        case CUDA_ERROR_ECC_UNCORRECTABLE:
            return "CUDA_ERROR_ECC_UNCORRECTABLE";

        case CUDA_ERROR_UNSUPPORTED_LIMIT:
            return "CUDA_ERROR_UNSUPPORTED_LIMIT";

        case CUDA_ERROR_CONTEXT_ALREADY_IN_USE:
            return "CUDA_ERROR_CONTEXT_ALREADY_IN_USE";

        case CUDA_ERROR_INVALID_SOURCE:
            return "CUDA_ERROR_INVALID_SOURCE";

        case CUDA_ERROR_FILE_NOT_FOUND:
            return "CUDA_ERROR_FILE_NOT_FOUND";

        case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND:
            return "CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND";

        case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED:
            return "CUDA_ERROR_SHARED_OBJECT_INIT_FAILED";

        case CUDA_ERROR_OPERATING_SYSTEM:
            return "CUDA_ERROR_OPERATING_SYSTEM";

        case CUDA_ERROR_INVALID_HANDLE:
            return "CUDA_ERROR_INVALID_HANDLE";

        case CUDA_ERROR_NOT_FOUND:
            return "CUDA_ERROR_NOT_FOUND";

        case CUDA_ERROR_NOT_READY:
            return "CUDA_ERROR_NOT_READY";

        case CUDA_ERROR_LAUNCH_FAILED:
            return "CUDA_ERROR_LAUNCH_FAILED";

        case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
            return "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES";

        case CUDA_ERROR_LAUNCH_TIMEOUT:
            return "CUDA_ERROR_LAUNCH_TIMEOUT";

        case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING:
            return "CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING";

        case CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED:
            return "CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED";

        case CUDA_ERROR_PEER_ACCESS_NOT_ENABLED:
            return "CUDA_ERROR_PEER_ACCESS_NOT_ENABLED";

        case CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE:
            return "CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE";

        case CUDA_ERROR_CONTEXT_IS_DESTROYED:
            return "CUDA_ERROR_CONTEXT_IS_DESTROYED";

        case CUDA_ERROR_ASSERT:
            return "CUDA_ERROR_ASSERT";

        case CUDA_ERROR_TOO_MANY_PEERS:
            return "CUDA_ERROR_TOO_MANY_PEERS";

        case CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED:
            return "CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED";

        case CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED:
            return "CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED";

        case CUDA_ERROR_UNKNOWN:
            return "CUDA_ERROR_UNKNOWN";
    }

    return "<unknown>";
}
#endif

#ifdef CUBLAS_API_H_
// cuBLAS API errors
static const char *_cudaGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}
#endif

template< typename T >
bool check(T result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
                file, line, static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
        /*
                std::stringstream ss;
                std::string msg("CUDA error at ");
                msg += file;
                msg += ":";
                ss << line;
                msg += ss.str();
                msg += " code=";
                ss << static_cast<unsigned int>(result);
                msg += ss.str();
                msg += " (";
                msg += _cudaGetErrorEnum(result);
                msg += ") \"";
                msg += func;
                msg += "\"";
                //throw msg;
                std::cerr  << msg <<"\n";
        */
        return true;
    }
    else
    {
        return false;
    }
}

#ifdef __DRIVER_TYPES_H__
// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(val)           check ( (val), #val, __FILE__, __LINE__ )

// This will output the proper error string when calling cudaGetLastError
#define getLastCudaError(msg)      __getLastCudaError (msg, __FILE__, __LINE__)

inline void __getLastCudaError(const char *errorMessage, const char *file, const int line)
{
    cudaError_t err = cudaGetLastError();

    if (cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
                file, line, errorMessage, (int)err, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}
#endif

#ifndef MAX
#define MAX(a,b) (a > b ? a : b)
#endif

// Beginning of GPU Architecture definitions
inline int _ConvertSMVer2Cores(int major, int minor, bool verbose)
{
    // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
    typedef struct
    {
        int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] =
    {
        { 0x10,  8 }, // Tesla Generation (SM 1.0) G80 class
        { 0x11,  8 }, // Tesla Generation (SM 1.1) G8x class
        { 0x12,  8 }, // Tesla Generation (SM 1.2) G9x class
        { 0x13,  8 }, // Tesla Generation (SM 1.3) GT200 class
        { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
        { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
        { 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
        { 0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
        {   -1, -1 }
    };

    int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1)
    {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
        {
            return nGpuArchCoresPerSM[index].Cores;
        }

        index++;
    }

    // If we don't find the values, we default use the previous one to run properly
    if (verbose)
    {
      printf("MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n", major, minor, nGpuArchCoresPerSM[7].Cores);
    }
    return nGpuArchCoresPerSM[7].Cores;
}
// end of GPU Architecture definitions

#ifdef __CUDA_RUNTIME_H__
// General GPU Device CUDA Initialization
inline int gpuDeviceInit(int devID, bool verbose)
{
    int deviceCount;
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0)
    {
        fprintf(stderr, "gpuDeviceInit() CUDA error: no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }

    if (devID < 0)
    {
        devID = 0;
    }

    if (devID > deviceCount-1)
    {
        fprintf(stderr, "\n");
        fprintf(stderr, ">> %d CUDA capable GPU device(s) detected. <<\n", deviceCount);
        fprintf(stderr, ">> gpuDeviceInit (-device=%d) is not a valid GPU device. <<\n", devID);
        fprintf(stderr, "\n");
        return -devID;
    }

    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

    if (deviceProp.computeMode == cudaComputeModeProhibited)
    {
        fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
        return -1;
    }

    if (deviceProp.major < 1)
    {
        fprintf(stderr, "gpuDeviceInit(): GPU device does not support CUDA.\n");
        exit(EXIT_FAILURE);
    }

    checkCudaErrors(cudaSetDevice(devID));
    if (verbose)
    {
        printf("gpuDeviceInit() CUDA Device [%d]: \"%s\n", devID, deviceProp.name);
    }

    return devID;
}

// This function returns the best GPU (with maximum GFLOPS)
inline int gpuGetMaxGflopsDeviceId(bool verbose)
{
    int current_device     = 0, sm_per_multiproc  = 0;
    int max_compute_perf   = 0, max_perf_device   = 0;
    int device_count       = 0, best_SM_arch      = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceCount(&device_count);

    // Find the best major SM Architecture GPU device
    while (current_device < device_count)
    {
        cudaGetDeviceProperties(&deviceProp, current_device);

        // If this GPU is not running on Compute Mode prohibited, then we can add it to the list
        if (deviceProp.computeMode != cudaComputeModeProhibited)
        {
            if (deviceProp.major > 0 && deviceProp.major < 9999)
            {
                best_SM_arch = MAX(best_SM_arch, deviceProp.major);
            }
        }

        current_device++;
    }

    // Find the best CUDA capable GPU device
    current_device = 0;

    while (current_device < device_count)
    {
        cudaGetDeviceProperties(&deviceProp, current_device);

        // If this GPU is not running on Compute Mode prohibited, then we can add it to the list
        if (deviceProp.computeMode != cudaComputeModeProhibited)
        {
            if (deviceProp.major == 9999 && deviceProp.minor == 9999)
            {
                sm_per_multiproc = 1;
            }
            else
            {
                sm_per_multiproc = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor, verbose);
            }

            int compute_perf  = deviceProp.multiProcessorCount * sm_per_multiproc * deviceProp.clockRate;

            if (compute_perf  > max_compute_perf)
            {
                // If we find GPU with SM major > 2, search only these
                if (best_SM_arch > 2)
                {
                    // If our device==dest_SM_arch, choose this, or else pass
                    if (deviceProp.major == best_SM_arch)
                    {
                        max_compute_perf  = compute_perf;
                        max_perf_device   = current_device;
                    }
                }
                else
                {
                    max_compute_perf  = compute_perf;
                    max_perf_device   = current_device;
                }
            }
        }

        ++current_device;
    }

    return max_perf_device;
}


// Initialization code to find the best CUDA Device
inline int findCudaDevice(int argc, const char **argv, bool verbose)
{
    cudaDeviceProp deviceProp;
    int devID = 0;

    // If the command-line has a device number specified, use it
    if (checkCmdLineFlag(argc, argv, "device"))
    {
        devID = getCmdLineArgumentInt(argc, argv, "device=");

        if (devID < 0)
        {
            fprintf(stderr, "Invalid command line parameter\n ");
            exit(EXIT_FAILURE);
        }
        else
        {
            devID = gpuDeviceInit(devID, verbose);

            if (devID < 0)
            {
                fprintf(stderr, "exiting...\n");
                exit(EXIT_FAILURE);
            }
        }
    }
    else
    {
        // Otherwise pick the device with highest Gflops/s
        devID = gpuGetMaxGflopsDeviceId(verbose);
        checkCudaErrors(cudaSetDevice(devID));
        checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));
        if (verbose)
        {
            printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);          
        }
    }

    return devID;
}

// General check for CUDA GPU SM Capabilities
inline bool checkCudaCapabilities(int major_version, int minor_version, bool verbose)
{
    cudaDeviceProp deviceProp;
    deviceProp.major = 0;
    deviceProp.minor = 0;
    int dev;

    checkCudaErrors(cudaGetDevice(&dev));
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));

    if ((deviceProp.major > major_version) ||
        (deviceProp.major == major_version && deviceProp.minor >= minor_version))
    {
        if (verbose)
        {
            printf("> Device %d: <%16s >, Compute SM %d.%d detected\n", dev, deviceProp.name, deviceProp.major, deviceProp.minor);
        }

        return true;
    }
    else
    {
        if (verbose)
        {
            printf("No GPU device was found that can support CUDA compute capability %d.%d.\n", major_version, minor_version);
        }

        return false;
    }
}
#endif

// end of CUDA Helper Functions


#endif
