cmake_minimum_required(VERSION 3.18)
project(ssapy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# The build backend selects the interpreter and installs pybind11 in the
# isolated build environment on every platform.
find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

add_library(ssapy STATIC src/ssapy.cpp include/ssapy.h)
target_include_directories(ssapy PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

pybind11_add_module(_ssapy pysrc/ssapy.cpp)
target_link_libraries(_ssapy PRIVATE ssapy)

install(TARGETS _ssapy
    LIBRARY DESTINATION ssapy
    RUNTIME DESTINATION ssapy
)
