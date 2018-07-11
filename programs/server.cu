/* Thie file contains the server that initializes the persistent module */
#include <stdio.h>

#include <fractional_gpu.h>
#include <persistent.h>

int main()
{
    int ret;

    ret = fgpu_server_init();
    if (ret < 0)
        return ret;

    printf("Server Started. Press any key to terminate server\n");

    getchar();

    fgpu_server_deinit();

    return 0;
}