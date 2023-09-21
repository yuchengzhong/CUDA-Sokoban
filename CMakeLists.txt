cmake_minimum_required(VERSION 3.10)
project(CudaSokoban LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)

enable_language(CUDA)

include_directories(${CMAKE_SOURCE_DIR})
file(GLOB SOURCES "*.cpp" "*.cu")
file(GLOB HEADERS "*.h" "*.hpp" "*.cuh")

add_executable(CudaSokoban ${SOURCES} ${HEADERS})

set_target_properties(CudaSokoban PROPERTIES CUDA_ARCHITECTURES 86)
set_target_properties(CudaSokoban PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(CudaSokoban ${CUDA_LIBRARIES})
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")