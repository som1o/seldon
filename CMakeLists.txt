cmake_minimum_required(VERSION 3.10)
project(Seldon VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

file(GLOB SOURCES "src/*.cpp")

# Find and enable OpenMP (Optional)
find_package(OpenMP)

# Seldon Application
add_executable(seldon ${SOURCES})

if(OpenMP_CXX_FOUND)
    target_link_libraries(seldon PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(seldon PRIVATE USE_OPENMP)
endif()

# Seldon Tests
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/tests.cpp")
    set(TEST_SOURCES src/tests.cpp src/Dataset.cpp src/MathUtils.cpp)
    add_executable(seldon_tests ${TEST_SOURCES})
    if(OpenMP_CXX_FOUND)
        target_link_libraries(seldon_tests PUBLIC OpenMP::OpenMP_CXX)
        target_compile_definitions(seldon_tests PRIVATE USE_OPENMP)
    endif()
    target_include_directories(seldon_tests PRIVATE include)
endif()
