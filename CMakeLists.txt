cmake_minimum_required(VERSION 3.10)
project(Seldon VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(SELDON_ENABLE_LEGACY "Build legacy Dataset/LogicEngine/StatsEngine/TerminalUI modules" OFF)

file(GLOB SOURCES "src/*.cpp")
list(FILTER SOURCES EXCLUDE REGEX "src/tests\\.cpp$")
if(NOT SELDON_ENABLE_LEGACY)
    list(FILTER SOURCES EXCLUDE REGEX "src/Dataset\\.cpp$")
    list(FILTER SOURCES EXCLUDE REGEX "src/LogicEngine\\.cpp$")
    list(FILTER SOURCES EXCLUDE REGEX "src/StatsEngine\\.cpp$")
    list(FILTER SOURCES EXCLUDE REGEX "src/TerminalUI\\.cpp$")
endif()

add_executable(seldon ${SOURCES})
target_include_directories(seldon PRIVATE include)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(seldon PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(seldon PRIVATE USE_OPENMP)
endif()
