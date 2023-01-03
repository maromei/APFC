cmake_minimum_required(VERSION 3.22.1)

project(apfc)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

add_executable(${PROJECT_NAME} src/fft.cpp)
add_executable(stencil src/stencil.cpp)

set(ENABLE_DOUBLE ON)
set(XTENSOR_FFTW_USE_DOUBLE ON)

find_package(xtl REQUIRED)
find_package(xtensor REQUIRED)

include_directories(/home/max/statics/xtensor-fftw/include)

target_link_libraries(${PROJECT_NAME} PRIVATE fftw3)
