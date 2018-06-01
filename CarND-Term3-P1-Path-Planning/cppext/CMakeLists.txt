cmake_minimum_required(VERSION 2.8.12)
project(cppext)

add_subdirectory(pybind11)
pybind11_add_module(cppext src/main.cpp)
