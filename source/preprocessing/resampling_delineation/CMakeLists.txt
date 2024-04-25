

cmake_minimum_required(VERSION 2.5)

project(resampling_delineation)

# Find ITK
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

add_executable(resampling_delineation ./source/resample_delineation.cpp)

target_link_libraries(resampling_delineation ${ITK_LIBRARIES})