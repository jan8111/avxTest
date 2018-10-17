cmake_minimum_required(VERSION 3.12)
project(avxTest)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-mavx -static")

add_executable(avxTest main.cpp)