cmake_minimum_required(VERSION 3.22)
project(aether_search)

set(CMAKE_CXX_STANDARD 20)

set(TENSORFLOW_DIR ${CMAKE_SOURCE_DIR}/external/tensorflow)

include_directories(
        ${TENSORFLOW_DIR}/include
        ${CMAKE_SOURCE_DIR}/include
)

link_directories(
        ${TENSORFLOW_DIR}/lib
)

file(GLOB_RECURSE SOURCES
        src/*.cpp
)

add_executable(aether_search ${SOURCES})

target_link_libraries(aether_search
        tensorflow
        tensorflow_framework
)
