# Minimum required CMake version
cmake_minimum_required(VERSION 3.15)

# Set policies for compatibility
cmake_policy(SET CMP0048 NEW)  # Project version policy
cmake_policy(SET CMP0063 NEW)  # Visibility preset policy

# Project settings
project(ssapy VERSION 1.1.10 LANGUAGES CXX)

# Compiler settings
set(CMAKE_CXX_STANDARD 17)                # Use C++17 standard
set(CMAKE_CXX_STANDARD_REQUIRED ON)       # Require C++17
set(CMAKE_CXX_EXTENSIONS OFF)             # Disable compiler-specific extensions
set(CMAKE_CXX_VISIBILITY_PRESET hidden)   # Hide symbols by default
set(CMAKE_POSITION_INDEPENDENT_CODE ON)   # Enable position-independent code (PIC)

# macOS-specific settings
set(CMAKE_MACOSX_RPATH 1)

# Output directories
if(NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)  # Set default output directory for libraries
endif()

# Find Python
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
set(PYTHON_MODULE_INSTALL_DIR "${Python3_SITEARCH}/ssapy")

# Use the bundled pybind11 subdirectory
add_subdirectory(pybind11)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Define the Python module target (_ssapy)
pybind11_add_module(_ssapy pysrc/ssapy.cpp)

# Ensure the Python module is installed in the correct location
set(CMAKE_POSITION_INDEPENDENT_CODE True)
install(TARGETS _ssapy
    LIBRARY DESTINATION ssapy # Install the compiled module in Python's site-packages
)

# Define a static library target (ssapy)
add_library(ssapy STATIC src/ssapy.cpp)

# Link the static library to the Python module
target_link_libraries(_ssapy PUBLIC ssapy)

# Add the header file explicitly to the static library target
target_sources(ssapy PRIVATE include/ssapy.h)

# Ensure the header file is included in the Python module target
target_include_directories(_ssapy PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
