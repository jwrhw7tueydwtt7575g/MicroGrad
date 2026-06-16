cmake_minimum_required(VERSION 3.20)
project(micrograd LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(MICROGRAD_BUILD_TESTS "Build C++ tests" OFF)
option(MICROGRAD_BUILD_PYTHON "Build Python bindings" ON)

if(MICROGRAD_BUILD_PYTHON)
    find_package(Python REQUIRED COMPONENTS Interpreter Development)
    find_package(pybind11 CONFIG REQUIRED)
endif()

find_package(Threads REQUIRED)

add_subdirectory(src)

if(MICROGRAD_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests/cpp)
endif()
