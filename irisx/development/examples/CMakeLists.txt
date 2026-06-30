# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.


set(EXAMPLES
    01_message_passing)


function(add_iris_example EXAMPLE)
    string(FIND ${EXAMPLE} "_" underscore_pos)
    math(EXPR underscore_pos "${underscore_pos} + 1")
    string(SUBSTRING ${EXAMPLE} ${underscore_pos} -1 example_name)
    add_executable(${example_name} ${EXAMPLE}/${example_name}.hip)
    target_link_libraries(${example_name} PRIVATE iris::iris)
endfunction()

foreach(example ${EXAMPLES})
    add_iris_example(${example})
endforeach()