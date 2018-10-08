#!/bin/bash
echo quit | nvidia-cuda-mps-control
nvidia-smi -i 0 -c DEFAULT
nvidia-smi -pm 0

