cmake_minimum_required(VERSION 3.28.0) # 指定CMake的最低版本要求

project(testCommon
    VERSION 1.2.3
    LANGUAGES CXX C
)

# CMAKE_CXX_STANDARD: 设置C++编译器的标准版本为C++14
set(CMAKE_CXX_STANDARD 14)

# CMAKE_CXX_STANDARD_REQUIRED: 强制要求所有C++源文件使用指定的C++标准
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CMAKE_CXX_EXTENSIONS: 要求编译器只使用标准的C++特性来编译代码
set(CMAKE_CXX_EXTENSIONS OFF)

# 添加当前目录到头文件的搜索路径
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# -Wall: 打开所有警告
# -g: 生成调试信息，可使用gdb等工具
# -Werror: 将所有警告信息转换为错误
# -Wno-error=pedantic: 用于禁止pedantic警告
# -Wno-error=switch: 用户禁止switch警告
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror -g")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-error=pedantic -Wno-error=switch")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=pedantic -Wno-error=switch")

set(3RDPARTY_BASE ${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdParty)
set(CPP_BASE ${CMAKE_CURRENT_SOURCE_DIR}/../../../cpp)
message(WARNING "CPP_BASE=${CPP_BASE}")
message(WARNING "3RDPARTY_BASE=${3RDPARTY_BASE}")

add_executable(${PROJECT_NAME}
    ${CPP_BASE}/test/testCommon
)

option(HIPPO_GLOG "whether enable glog flags." ON)
if(HIPPO_GLOG)
    message(WARNING "HIPPO_GLOG FLAG IS ON.")
    add_definitions(-DHIPPO_GLOG=1)
    target_include_directories(${PROJECT_NAME} PUBLIC
        ${3RDPARTY_BASE}
    )
    target_link_directories(${PROJECT_NAME} PUBLIC
        ${3RDPARTY_BASE}/glog/glog0.6.0/lib
    )
else()
    message(WARNING "HIPPO_GLOG FLAG IS OFF.")
endif()

# 头文件
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/inc/common
)

# 库目录 库文件
add_subdirectory(
    ${CPP_BASE}/test/testCommon
)
