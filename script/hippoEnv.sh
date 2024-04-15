#!/bin/bash

# script env
PNORMAL='\e[0m' # 重置文本颜色为默认颜色，表示 NORMAL
PDEBUG='\e[35m' # 洋红色表示 DEBUG
PINFO='\e[36m' # 青色表示 INFORMATION
PWARNING='\e[33m' # 黄色表示 WARNING
PERROR='\e[31m' # 红色表示 ERROR

hippoHpcScript=$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)
hippoHpcBase=$hippoHpcScript/../
hippoHpcBase=$(realpath "$hippoHpcBase")
hippoHpcLog=$hippoHpcBase/log
hippoHpcTools=$hippoHpcBase/tools
hippoHpcCpp=$hippoHpcBase/cpp
hippoHpcPython=$hippoHpcBase/python
hippoBin=$hippoHpcBase/bin

echo -e $PDEBUG "\b\c"
echo "hippoHpcScript = $hippoHpcScript"
echo "hippoHpcBase = $hippoHpcBase"
echo "hippoHpcLog = $hippoHpcLog"
echo "hippoHpcTools = $hippoHpcTools"
echo "hippoHpcCpp = $hippoHpcCpp"
echo "hippoHpcPython = $hippoHpcPython"
echo "hippoBin = $hippoBin"
echo -e $PNORMAL "\b\c"

cmake_folder="cmake-3.28.0-linux-x86_64"
cmake_tar_name="$cmake_folder.tar.gz"
cmake_url="https://cmake.org/files/LatestRelease/$cmake_tar_name"
cmake_bin=$hippoHpcTools/cmake/bin

function common_ensure_folder_exists() {
    if [ ! -d "$1" ]; then
        mkdir -p $1
    fi
}

common_ensure_folder_exists $hippoBin
common_ensure_folder_exists $hippoHpcLog

# cmake
function build_cmake() {
    echo -e $PINFO "\b\c"
    common_ensure_folder_exists $hippoHpcTools
    pushd $hippoHpcTools > /dev/null
    if [ ! -d cmake/bin ]; then
        if [ ! -d $cmake_folder ]; then
            if [ ! -f $cmake_tar_name ]; then
                wget ${cmake_url}
                tar -zxf "$cmake_tar_name" -C "$hippoHpcTools"
                rm $cmake_tar_name
            fi
        fi
        ln -s $cmake_folder cmake
    fi
    popd > /dev/null
    echo -e $PNORMAL "\b\c"
}

function export_cmake() {
    build_cmake
    export PATH=${cmake_bin}:$PATH
    cmake_version=$(cmake --version | grep -oP '(\d+\.\d+\.\d+)')
    echo "cmake_version = $cmake_version"
}

# export_cmake

# cuda
CUDA_HOME=/usr/local/cuda
CUDNN_HOME=/usr/local/cuda

function export_hpc() {
    export CUDA_HOME CUDNN_HOME
}

export_hpc
