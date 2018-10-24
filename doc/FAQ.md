# Frequently Asked Questions

## How do I debug FGPU code?

Modify the file *$PROJ_DIR/CMakelists.txt* such that it contains
```
set(CMAKE_BUILD_TYPE Debug)         # Uncomment this line
#set(CMAKE_BUILD_TYPE Release)      # Comment this line
```

After the changes, recompile FGPU. 
```
cd $PROJ_DIR/build
make
```
FGPU code should be compiled with debug symbols. You can now use gdb to single-step
through FGPU API. Revert back the changes and compile again after completion of debugging.

## My application is complaining that "FGPU:Couldn't open shmem"

This indicates that the fgpu_server is not running. Please see the document *$PROJ_DIR/doc/PORT.md*.

## FGPU server is complaining that "FGPU:Couldn't get device color info"

This indicates that the device driver has not been confifured for memory partitioning but FGPU has been.
Recompile FGPU with appropriate configuration and then install the device driver. 
Please see the document *$PROJ_DIR/doc/BUILD.md*.

## FGPU server is complaining that "FGPU:MPS is not enabled"
FGPU server requires Nvidia MPS. Run the following command
```
sudo $PROJ_DIR/scripts/mps_init.sh
```

## How do I check if CUDA SDK and Nvidia device driver have been installed correctly?
By running *nvidia-smi* tool, we can do an end-to-end check. 
```
nvidia-smi
```
It should report the details about all the GPUs it could find attached to current machine. For example
```
Tue Oct 23 17:29:12 2018       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 390.48                 Driver Version: 390.48                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 1080    Off  | 00000000:01:00.0 Off |                  N/A |
|  0%   46C    P0    37W / 180W |      0MiB /  8119MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

## I cannot run the application as it complains "error while loading shared libraries: libcuda.so" or "error while loading shared libraries: libcudart.so"

There can be four reasons for this:

* Environment variables *PATH* or *LD_LIBRARY_PATH* are not configured properly <br/>
    Both these variables should point to cuda installation path. Refer to *$PROJ_DIR/doc/BUILD.md*

* CUDA SDK is not installed  <br/>
    Check that the CUDA SDK is installed by running following command
    ```
    nvcc --version
    ```
    It should state the CUDA SDK version as 9.1

* Nvidia device driver is not installed <br/>
    Check that the device driver is installed.

* *ldconfig* cache is not populated correctly. <br/>
    Try running the following command:
    ```
    sudo ldconfig /usr/local/cuda/lib64
    ```


## How do I uninstall the Nvidia device driver?
```
sudo $PROJ_DIR/driver/NVIDIA-Linux-x86_64-390.48/nvidia-installer --uninstall
```

## *mps_stop.sh* script is stuck. What should I do?
To stop MPS, it is first required that all applications using CUDA are stopped/killed
(including *fgpu_server*).
You might also want to try running the following command
```
sudo $PROJ_DIR/scripts/mps_kill.sh
```

## My application is stuck at the start and not making any progress

There can be three reasons for this:
* Pre-Volta MPS only allows a single user at a time  <br/>
    If you are using a GPU that is of pre-Volta architecture, only single user can run applications
    at a time. Check that all the running applications (including *fgpu_server*) have been launched by the same user.
    Note that an application that is ran using *sudo* runs as root.

* MPS is stuck <br/>
    Pre-Volta MPS is finicky. If an application crashes while running on GPU, it might cause issues with MPS. Subsequent applications
    might not be able to run. Sometimes MPS might also become unresponsive. Hence in this situation, stop all applications using CUDA and then run
    either of the following commands:
    ```
    sudo $PROJ_DIR/scripts/mps_stop.sh
    ```
    or
    ```
    sudo $PROJ_DIR/scripts/mps_kill.sh
    ```

* Bug in device driver <br/>
    There might be some bugs in the Nvidia device driver. To check if this is the cause, the following command outputs the linux kernel
    log. Check if the log shows any kernel panics.
    ```
    dmesg
    ```
    Some bugs in the device driver might cause reboots. Please open a new issue with the log attached incase you notice any reboots or kernel panics.


## I have read this FAQ and all the supplement documents in this repository. I am still facing an issue not covered. What should I do?

Please open a new issue on the github repository.
