# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

function(add_iris_benchmark NAME)
    add_executable(${NAME} ${NAME}.hip)
    target_link_libraries(${NAME} PRIVATE iris::iris)
endfunction()

# Add benchmarks
add_iris_benchmark(put)
add_iris_benchmark(all_put)
add_iris_benchmark(moe_dispatch_pack)
add_iris_benchmark(moe_dispatch_pack_quant)
add_iris_benchmark(moe_dispatch_pack_quant_pipelined)
