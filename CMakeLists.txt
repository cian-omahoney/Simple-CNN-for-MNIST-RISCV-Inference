cmake_minimum_required(VERSION 3.26)
project(Simple_CNN_for_MNIST_RISCV_Inference C)

set(CMAKE_C_STANDARD 11)

add_executable(Simple_CNN_for_MNIST_RISCV main.c cnn.c)
target_link_libraries(Simple_CNN_for_MNIST_RISCV m)

