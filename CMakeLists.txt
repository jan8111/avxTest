cmake_minimum_required(VERSION 3.10)
project(avxTest)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_FLAGS "-mavx -static")

add_executable(avxTest main.c)
target_link_libraries(avxTest -lm)