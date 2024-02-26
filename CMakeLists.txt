cmake_policy(SET CMP0048 NEW)
project(HayStack VERSION 0.1.0 LANGUAGES C CXX CUDA)

cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(transpose_cuda cuda/transpose.cu)
set_target_properties(transpose_cuda PROPERTIES CUDA_ARCHITECTURES OFF)

