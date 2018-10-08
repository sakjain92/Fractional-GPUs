#Kill mps processes if it gets hanged
#Call this script with root priviledges
killall nvidia-cuda-mps-control
killall nvidia-cuda-mps-server
nvidia-smi -i 0 -c DEFAULT
nvidia-smi -pm 0
