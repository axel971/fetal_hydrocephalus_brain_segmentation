

cmake_minimum_required(VERSION 2.5)

project(relabeling)

# Find ITK
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

add_executable(relabeling ./source/relabeling.cpp)

target_link_libraries(relabeling ${ITK_LIBRARIES})