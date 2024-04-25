

cmake_minimum_required(VERSION 2.5)

project(fill_lateral_ventricle)

# Find ITK
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

add_executable(fill_lateral_ventricle ./source/fill_lateral_ventricle.cpp)

target_link_libraries(fill_lateral_ventricle ${ITK_LIBRARIES})