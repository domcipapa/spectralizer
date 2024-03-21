cmake_minimum_required(VERSION 3.28)
project(spectralizer)

set(CMAKE_C_STANDARD 23)
set(BUILD_SHARED_LIBS OFF)

add_executable(spectralizer src/main.c)
target_link_libraries(spectralizer PRIVATE raylib m)
