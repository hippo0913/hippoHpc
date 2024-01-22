#!/bin/bash

scriptBase=$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)
source $scriptBase/hippoEnv.sh

pushd $hippoBin > /dev/null

cmake ..
make -j${nproc}

popd > /dev/null
