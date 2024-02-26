#!/bin/bash

export SYCLOMATIC_HOME=/workspace
export PATH_TO_C2S_INSTALL_FOLDER=/workspace/c2s_install
source $PATH_TO_C2S_INSTALL_FOLDER/setvars.sh
icpx -fsycl -std=c++17 -O3 transpose.dp.cpp 
