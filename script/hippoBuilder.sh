#!/bin/bash

source hippoEnv.sh

pushd $hippoBin > /dev/null

cmake ..
make -j${nproc}

popd > /dev/null
