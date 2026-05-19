cmake_minimum_required(VERSION 3.18)

project(cuda_learning LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(PCL REQUIRED COMPONENTS common io)

add_executable(sphere src/sphere.c)
target_link_libraries(sphere PRIVATE m)

add_executable(fps src/fps.cu)
target_include_directories(fps PRIVATE ${PCL_INCLUDE_DIRS})
target_compile_definitions(fps PRIVATE ${PCL_DEFINITIONS})
target_link_libraries(fps PRIVATE ${PCL_LIBRARIES})

set_target_properties(fps PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/sphere.pcd")
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/src/sphere.pcd"
        "${CMAKE_CURRENT_BINARY_DIR}/sphere.pcd"
        COPYONLY
    )
endif()
