cmake_minimum_required(VERSION 3.28.0) # 指定CMake的最低版本要求

project(testCommon
    VERSION 1.2.3
    LANGUAGES CXX C
)

# 源文件
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} TEST_COMMON_SRC_FILES)
add_executable(${PROJECT_NAME}
    ${TEST_COMMON_SRC_FILES}
)

# 库目录 库文件
target_link_libraries(${PROJECT_NAME}
    ${GLOG_LIBS}
)
