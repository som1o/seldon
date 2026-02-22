cmake_minimum_required(VERSION 3.10)
project(Seldon VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

file(GLOB SOURCES "src/*.cpp")
list(FILTER SOURCES EXCLUDE REGEX "src/tests\\.cpp$")

add_executable(seldon ${SOURCES})
target_include_directories(seldon PRIVATE include)

option(SELDON_ENABLE_OPENMP "Enable OpenMP acceleration for supported kernels" ON)

if(SELDON_ENABLE_OPENMP)
    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(seldon PRIVATE OpenMP::OpenMP_CXX)
        target_compile_definitions(seldon PRIVATE USE_OPENMP)
        message(STATUS "Seldon: OpenMP enabled")
    else()
        message(STATUS "Seldon: OpenMP requested but not found; building without OpenMP")
    endif()
else()
    message(STATUS "Seldon: OpenMP explicitly disabled (SELDON_ENABLE_OPENMP=OFF)")
endif()
