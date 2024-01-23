cmake_minimum_required(VERSION 3.28.0) # 指定CMake的最低版本要求

project(
    hippoHpcCppTestBase
)

if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/testCommon)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/testCommon)
else()
    message(WARNING "${CMAKE_CURRENT_SOURCE_DIR}/testCommon dir does not exist")
endif()

if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/testCudaDevice)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/testCudaDevice)
else()
    message(WARNING "${CMAKE_CURRENT_SOURCE_DIR}/testCudaDevice dir does not exist")
endif()
