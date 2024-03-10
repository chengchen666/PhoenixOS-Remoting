cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(remoting_test LANGUAGES CXX CUDA)

# Find CUDA
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF CACHE INTERNAL "")
find_package(CUDA REQUIRED)

# Include directories for CUDA
include_directories(${CUDA_INCLUDE_DIRS})
message(STATUS "CUDA Libraries: ${CUDA_CUDART_LIBRARY}")

file(GLOB_RECURSE sources RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")
foreach(source IN LISTS sources)
    # Get file name without directory
    get_filename_component(source_name ${source} NAME_WE)
    # Specify the executable to be built and its sources
    add_executable(${source_name} ${source})
    # Link against the CUDA libraries
    target_link_libraries(${source_name} PRIVATE ${CUDA_LIBRARIES} ${CUDA_CUDART_LIBRARY})
    target_link_libraries(${source_name} PRIVATE nvidia-ml)
    # Specify C++11 as the standard for the host compiler
    set_target_properties(${source_name} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CXX_STANDARD 11
        CXX_STANDARD_REQUIRED YES
    )
endforeach()

# Set optimization level and pass other flags to the CUDA NVCC compiler
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O2;-std=c++11;-lcudnn;-lcuda;-lcudart;-lnvidia-ml)

# If you need to specify architecture-specific flags, for example:
# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_30,code=sm_30)

# Clean-up command (you might not need this as CMake has a 'clean' target)
add_custom_target(clean-all
   COMMAND ${CMAKE_BUILD_TOOL} clean
   COMMAND ${CMAKE_COMMAND} -P clean-all.cmake
)

# The clean-all.cmake file (which you need to create) should contain:
# file(REMOVE_RECURSE ${CMAKE_BINARY_DIR}/CMakeCache.txt
#                     ${CMAKE_BINARY_DIR}/CMakeFiles
#                     ${CMAKE_BINARY_DIR}/${PROJECT_NAME}${CMAKE_EXECUTABLE_SUFFIX}
#                     ${CMAKE_BINARY_DIR}/Makefile
#                     ${CMAKE_BINARY_DIR}/cmake_install.cmake
# )
