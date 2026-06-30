// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "hip/hip_runtime.h"

#define hip_try(error)                                                    \
  if (error != hipSuccess) {                                              \
    std::cerr << "Hip error: " << hipGetErrorString(error) << " at line " \
              << __LINE__ << std::endl;                                   \
    exit(-1);                                                             \
  }

class gpu_timer {
 private:
  hipEvent_t start_;
  hipEvent_t stop_;

 public:
  gpu_timer() {
    hip_try(hipEventCreate(&start_));
    hip_try(hipEventCreate(&stop_));
  }

  ~gpu_timer() {
    hip_try(hipEventDestroy(start_));
    hip_try(hipEventDestroy(stop_));
  }

  void start() { hip_try(hipEventRecord(start_, 0)); }

  void stop() {
    hip_try(hipEventRecord(stop_, 0));
    hip_try(hipEventSynchronize(stop_));
  }

  float elapsed_ms() {
    float milliseconds = 0;
    hip_try(hipEventElapsedTime(&milliseconds, start_, stop_));
    return milliseconds;
  }
  void reset() { hip_try(hipEventRecord(start_, 0)); }
};
