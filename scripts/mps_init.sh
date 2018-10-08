#!/bin/bash
# the following must be performed with root privilege
export CUDA_VISIBLE_DEVICES="0"
nvidia-smi -i 0 -c EXCLUSIVE_PROCESS
nvidia-cuda-mps-control -d
nvidia-smi -pm ENABLED
