cmake_minimum_required(VERSION 3.21)
project(ByteTrackCPP)

set(CMAKE_CXX_STANDARD 17)

option(BUILD_PYBIND "Build pybind11 Python module" OFF)

# OpenCV was built with CUDA 12.1; ensure FindCUDA locates the correct toolkit
if(EXISTS /usr/local/cuda-12.1)
    set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-12.1 CACHE PATH "" FORCE)
endif()

find_package(OpenCV REQUIRED COMPONENTS core)
if(NOT BUILD_PYBIND)
    find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc videoio imgcodecs)
endif()
find_package(Eigen3 REQUIRED)

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
)

set(CORE_SRC_FILES
    ${PROJECT_SOURCE_DIR}/src/BYTETracker.cpp
    ${PROJECT_SOURCE_DIR}/src/STrack.cpp
    ${PROJECT_SOURCE_DIR}/src/kalmanFilter.cpp
    ${PROJECT_SOURCE_DIR}/src/lapjv.cpp
    ${PROJECT_SOURCE_DIR}/src/utils.cpp
)

if(BUILD_PYBIND)
    find_package(pybind11 REQUIRED)
    pybind11_add_module(_bytetrack
        ${CORE_SRC_FILES}
        ${PROJECT_SOURCE_DIR}/src/PyBYTETrack.cpp
    )
    target_link_libraries(_bytetrack PRIVATE
        ${OpenCV_LIBS}
        Eigen3::Eigen
    )
    install(TARGETS _bytetrack DESTINATION bytetrack)
else()
    file(GLOB TEST_SRC_FILES ${PROJECT_SOURCE_DIR}/test/*.cpp)
    add_executable(test_app ${CORE_SRC_FILES} ${TEST_SRC_FILES})
    target_link_libraries(test_app
        ${OpenCV_LIBS}
        Eigen3::Eigen
    )
    target_include_directories(test_app PRIVATE ${OpenCV_INCLUDE_DIRS})
endif()
