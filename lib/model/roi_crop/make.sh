#!/usr/bin/env bash

<<<<<<< HEAD
#CUDA_PATH=/usr/local/cuda/

cd src
echo "Compiling my_lib kernels by nvcc..."
nvcc -c -o roi_crop_cuda_kernel.cu.o roi_crop_cuda_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_50
=======
CUDA_PATH=/usr/local/cuda/

cd src
echo "Compiling my_lib kernels by nvcc..."
nvcc -c -o roi_crop_cuda_kernel.cu.o roi_crop_cuda_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52
>>>>>>> 793eeda709a4483589939795954491531204c768

cd ../
python build.py
