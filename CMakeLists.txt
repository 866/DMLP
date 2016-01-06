cmake_minimum_required(VERSION 2.0)
project(DynamicMLP)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -o3")
set(SOURCE_FILES main.cpp)
add_executable(DynamicMLP ${SOURCE_FILES} Node.cpp Node.h
common.h IActivationFunction.cpp IActivationFunction.h Layer.cpp
Layer.h MultiLayerPerceptron.cpp MultiLayerPerceptron.h
Trainer.cpp Trainer.h ProcessingUnit.h)
