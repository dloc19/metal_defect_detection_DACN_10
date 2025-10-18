cmake_minimum_required(VERSION 3.16)
project(metal_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Output bin to build/bin for convenience
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Find required packages
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui imgcodecs)
find_package(Threads REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS} include)

# Create executable
add_executable(mdproc src/main.cpp src/mdproc.cpp)

# Compiler-specific options
if(MSVC)
  # MSVC optimizations
  target_compile_options(mdproc PRIVATE 
    /W4 /permissive- /O2 /Ob2 /Oi /Ot /Oy /GL /arch:AVX2
    /DNDEBUG /D_CRT_SECURE_NO_WARNINGS)
  target_link_options(mdproc PRIVATE /LTCG /OPT:REF /OPT:ICF)
else()
  # GCC/Clang optimizations
  target_compile_options(mdproc PRIVATE 
    -Wall -Wextra -Wpedantic -O3 -march=native -mtune=native
    -ffast-math -funroll-loops -flto -DNDEBUG)
  target_link_options(mdproc PRIVATE -flto)
endif()

# Link libraries
target_link_libraries(mdproc PRIVATE 
  ${OpenCV_LIBS} 
  Threads::Threads)

# Set properties for better performance
set_target_properties(mdproc PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  POSITION_INDEPENDENT_CODE ON)

# Install rule
install(TARGETS mdproc RUNTIME DESTINATION bin)

# Optional: Create a shared library version
option(BUILD_SHARED_LIB "Build shared library" OFF)
if(BUILD_SHARED_LIB)
  add_library(mdproc_lib SHARED src/mdproc.cpp)
  target_include_directories(mdproc_lib PUBLIC include)
  target_link_libraries(mdproc_lib PUBLIC ${OpenCV_LIBS} Threads::Threads)
  set_target_properties(mdproc_lib PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    VERSION 1.0.0
    SOVERSION 1)
  install(TARGETS mdproc_lib LIBRARY DESTINATION lib)
endif()

# Print configuration info
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Threading support: ${CMAKE_THREAD_LIBS_INIT}")
