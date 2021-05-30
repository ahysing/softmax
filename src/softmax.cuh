//
//  softmax.cuh
//  
//
//  Created by Andreas Dreyer Hysing on 09/05/2021.
//
#ifndef __SOFTMAX_CUH__
#define __SOFTMAX_CUH__

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <vector>

std::unique_ptr<std::vector<double>> softMax(const std::weak_ptr<std::vector<double>> values, unsigned int deviceId, bool verbose);
std::unique_ptr<std::vector<double>> softMax(const std::weak_ptr<std::vector<double>> values, bool verbose);
std::unique_ptr<std::vector<double>> softMaxCPU(const std::weak_ptr<std::vector<double>> values, bool verbose);

#endif