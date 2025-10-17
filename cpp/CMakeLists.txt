cmake_minimum_required(VERSION 3.16)
project(metal_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Output bin to build/bin for convenience
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Find OpenCV (set OpenCV_DIR when needed on Windows)
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui)

include_directories(${OpenCV_INCLUDE_DIRS} include)

add_executable(mdproc src/main.cpp src/mdproc.cpp)

# Warnings
if(MSVC)
  target_compile_options(mdproc PRIVATE /W4 /permissive-)
else()
  target_compile_options(mdproc PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Link OpenCV
target_link_libraries(mdproc PRIVATE ${OpenCV_LIBS})

# Install rule
install(TARGETS mdproc RUNTIME DESTINATION bin)
