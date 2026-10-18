cmake_minimum_required(VERSION 3.20)

project(LLM_Inference LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_EXPORT_COMPILE_COMMANDS on)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

# include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CUDA_INCLUDE_DIRS})
# aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC)
aux_source_directory(${CMAKE_SOURCE_DIR}/src SRC)

find_package(CUDAToolkit REQUIRED)

add_executable(main src/main.cpp ${SRC})
target_include_directories(main PUBLIC ${CMAKE_SOURCE_DIR}/include)

set_target_properties(main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(main PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_link_libraries(main PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
