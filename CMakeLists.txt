cmake_minimum_required(VERSION 3.10)
project(Seldon VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

file(GLOB SOURCES "src/*.cpp")

# Find and enable OpenMP
find_package(OpenMP REQUIRED)

# Seldon Application
add_executable(seldon ${SOURCES})
target_link_libraries(seldon PUBLIC OpenMP::OpenMP_CXX)
