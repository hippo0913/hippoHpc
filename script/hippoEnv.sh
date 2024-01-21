#!/bin/bash

# script env
PNORMAL='\e[0m' # 重置文本颜色为默认颜色，表示 NORMAL
PDEBUG='\e[35m' # 洋红色表示 DEBUG
PINFO='\e[36m' # 青色表示 INFORMATION
PWARNING='\e[33m' # 黄色表示 WARNING
PERROR='\e[31m' # 红色表示 ERROR

echo "hippoEnv start"

hippoHpcScript=$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)
hippoHpcBase=$hippoHpcScript/../
hippoHpcBase=$(realpath "$hippoHpcBase")
hippo3rdParty=$hippoHpcBase/3rdParty
hippo3rdPartyBin=$hippo3rdParty/bin

echo -e $PDEBUG
echo "hippoHpcScript = $hippoHpcScript"
echo "hippoHpcBase = $hippoHpcBase"
echo "hippo3rdParty = $hippo3rdParty"
echo "hippo3rdPartyBin = $hippo3rdPartyBin"
echo -e $PNORMAL

cmake_folder="cmake-3.28.0-linux-x86_64"
cmake_tar_name="$cmake_folder.tar.gz"
cmake_url="https://cmake.org/files/LatestRelease/$cmake_tar_name"
cmake_bin=$hippo3rdPartyBin/cmake/bin

function common_ensure_folder_exists() {
    if [ ! -d "$1" ]; then
        mkdir -p $1
    fi
}

common_ensure_folder_exists $hippo3rdParty
common_ensure_folder_exists $hippo3rdPartyBin

function build_cmake() {
    echo -e $PINFO
    echo "Calling cmake_autoConfig.sh"

    common_ensure_folder_exists $hippo3rdPartyBin
    pushd $hippo3rdPartyBin > /dev/null
    if [ ! -d cmake/bin ]; then
        if [ ! -d $cmake_folder ]; then
            if [ ! -f $cmake_tar_name ]; then
                wget ${cmake_url}
                tar -zxf "$cmake_tar_name" -C "$hippo3rdPartyBin"
                rm $cmake_tar_name
            fi
        fi
        ln -s $cmake_folder cmake
    fi
    popd > /dev/null

    echo "Calling cmake_autoConfig.sh end"
    echo -e $PNORMAL
}

function export_cmake() {
    build_cmake
    export PATH=${cmake_bin}:$PATH
    cmake_version=$(cmake --version | grep -oP '(\d+\.\d+\.\d+)')
    echo "cmake_version = $cmake_version"
}

while [ -n "$1" ]
do
    case "$1" in
    -cmake)
        export_cmake
        ;;
esac
    shift
done

echo "hippoEnv end"
