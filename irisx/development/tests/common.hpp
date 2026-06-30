// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include "hip/hip_runtime.h"

#define hip_try(error)                                                         \
  if (error != hipSuccess) {                                                   \
    std::cerr << "Hip error: " << hipGetErrorString(error) << " at line "      \
              << __LINE__ << std::endl;                                        \
    exit(-1);                                                                  \
  }

