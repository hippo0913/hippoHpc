cmake_minimum_required(VERSION 3.16) # 指定CMake的最低版本要求

project(
    hippoHpcCpp
)

# 添加当前目录到头文件的搜索路径
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# -Wall: 打开所有警告
# -g: 生成调试信息，可使用gdb等工具
# -Werror: 将所有警告信息转换为错误
# -Wno-error=pedantic: 用于禁止pedantic警告
# -Wno-error=switch: 用户禁止switch警告
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror -g")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-error=pedantic -Wno-error=switch")
message("CMAKE_C_FLAGS${CMAKE_C_FLAGS}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=pedantic -Wno-error=switch")
message("CMAKE_CXX_FLAGS${CMAKE_CXX_FLAGS}")

# CMAKE_CXX_STANDARD: 设置C++编译器的标准版本为C++14
set(CMAKE_CXX_STANDARD 14)
message("CMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}")

# CMAKE_CXX_STANDARD_REQUIRED: 强制要求所有C++源文件使用指定的C++标准
set(CMAKE_CXX_STANDARD_REQUIRED ON)
message("CMAKE_CXX_STANDARD_REQUIRED=${CMAKE_CXX_STANDARD_REQUIRED}")

# CMAKE_CXX_EXTENSIONS: 要求编译器只使用标准的C++特性来编译代码
set(CMAKE_CXX_EXTENSIONS OFF)
message("CMAKE_CXX_EXTENSIONS=${CMAKE_CXX_EXTENSIONS}")

set(CPP_BASE ${CMAKE_CURRENT_SOURCE_DIR})
message("CPP_BASE=${CPP_BASE}")

set(3RDPARTY_BASE ${CPP_BASE}/3rdParty)
message("3RDPARTY_BASE=${3RDPARTY_BASE}")

set(TEST_BASE ${CPP_BASE}/testBase)
message("TEST_BASE=${TEST_BASE}")

set(CUDA_HOME "$ENV{CUDA_HOME}")
message("CUDA_HOME=${CUDA_HOME}")
enable_language(CUDA)
include_directories(${CUDA_HOME}/targets/x86_64-linux/include)
link_directories(${CUDA_HOME}/targets/x86_64-linux/lib)
set(CUDA_LIBS cuda cudart)

option(HIPPO_DEBUG "whether enable debug flags." OFF)
if(HIPPO_DEBUG)
    message(WARNING "HIPPO DEBUG FLAG IS ON.")
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of Debug, options are: None Debug Release RelWithDebInfo MinSizeRel." FORCE)
    add_definitions(-DHIPPO_DEBUG=1)
else()
    message(WARNING "HIPPO RELEASE FLAG IS ON.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of Release, options are: None Debug Release RelWithDebInfo MinSizeRel." FORCE)
    add_definitions(-DHIPPO_RELEASE=1)
endif()

# 日志库目前只使用了GLOG，不能关闭
option(HIPPO_GLOG "whether enable glog flags." ON)
if(HIPPO_GLOG)
    message(WARNING "HIPPO_GLOG FLAG IS ON.")
    add_definitions(-DHIPPO_GLOG=1)
    include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdParty/glog//glog0.6.0/include
    )
    link_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/3rdParty/glog/glog0.6.0/lib
    )
    set(GLOG_LIBS glog)
else()
    message(WARNING "HIPPO_GLOG FLAG IS OFF.")
endif()

option(HIPPO_MULTI_CUDACONTEXT "whether enable multiple cuda context." ON)
if (HIPPO_MULTI_CUDACONTEXT)
  message(WARNING "HIPPO_MULTI_CUDACONTEXT is ON.")
  add_definitions(-DHIPPO_MULTI_CUDACONTEXT=1)
else()
  message(WARNING "HIPPO_MULTI_CUDACONTEXT is OFF.")
endif()

# 头文件
include_directories(
    ${CPP_BASE}/inc
    ${CPP_BASE}/inc/common
    ${CPP_BASE}/inc/cuda
)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src SRC_FILES)
add_executable(${PROJECT_NAME}
    ${SRC_FILES}
)
target_link_libraries(${PROJECT_NAME}
    ${GLOG_LIBS}
    ${CUDA_LIBS}
)

if (EXISTS ${CPP_BASE}/testBase)
    add_subdirectory(${CPP_BASE}/testBase)
else()
    message(WARNING "$TEST_BASE dir does not exist")
endif()
