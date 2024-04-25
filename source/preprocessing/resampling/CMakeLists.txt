

cmake_minimum_required(VERSION 2.5)

project(resampling)

# Find ITK
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

add_executable(resampling ./source/resample.cpp)

target_link_libraries(resampling ${ITK_LIBRARIES})