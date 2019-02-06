cmake_minimum_required(VERSION 3.10)
project(origin)

set(CMAKE_CXX_STANDARD 11)

include_directories(.)

add_executable(origin
        Config.h
        InputSpecificActiveNetworkFactory.h
        Layer.cpp
        Layer.h
        main.cpp
        Network.cpp
        Network.h
        Node.cpp
        Node.h)
