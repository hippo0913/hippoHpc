cmake_minimum_required(VERSION 3.16) # 指定CMake的最低版本要求

project(testImageProcessing
    VERSION 1.1.1
    LANGUAGES CXX C
)

find_package(OpenCV REQUIRED)

# 头文件
include_directories(
    ${CPP_BASE}/inc
    ${CPP_BASE}/inc/image
    ${OpenCV_INCLUDE_DIRS}
)

# 源文件
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} TEST_IMAGE_PROCESSING_SRC_FILES)
add_executable(${PROJECT_NAME}
    ${TEST_IMAGE_PROCESSING_SRC_FILES}
)

# 库文件
target_link_libraries(${PROJECT_NAME}
    ${GLOG_LIBS}
    ${OpenCV_LIBS}
)
