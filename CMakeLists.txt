#  This is main CMake file for Neural Network project.
#
#  To generate Makefile for Debug or Release version run:
#
#      cmake -D CMAKE_BUILD_TYPE=X ..
#
#  where X is 'Debug' or 'Release'. To see compilation flags run:
#
#      make VERBOSE=1
#
#  See www.cmake.org to learn more about CMake.

cmake_minimum_required(VERSION 3.1)

# Set project's name and programming language.

project(geonn LANGUAGES CXX)

# Set C++14 standard.

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define a few constant values.

set(MAIN_SOURCE_FILE           "main.cpp")
set(CXX_CODE_QUALITY_FLAGS     "-Wall -pedantic -Wextra -Wno-long-long")

# Print some information about detected system.

message(STATUS "Detected system: ${CMAKE_SYSTEM_NAME} ${CMAKE_HOST_SYSTEM_PROCESSOR}.")

# Find third party libraries provided by CMake.

find_package(OpenCV REQUIRED)

# Include headers.

include_directories(${OpenCV_INCLUDE_DIRS})

# Generate executable.

add_executable(${PROJECT_NAME} ${MAIN_SOURCE_FILE})

# Link against libraries.

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

# Set restrictive compilation flags.

if (CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXX_CODE_QUALITY_FLAGS}")
endif ()
