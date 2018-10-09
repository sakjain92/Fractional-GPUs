/* Thie file contains the server that initializes the persistent module */
#include <stdio.h>
#include <unistd.h>

#include <fractional_gpu.hpp>
#include <fgpu_internal_persistent.hpp>

int main()
{
    int ret;

    ret = fgpu_server_init();
    if (ret < 0)
        return ret;

    printf("Server Started. Press any key to terminate server\n");

    getchar();

    /* When running as background process, stdin might be closed */
    if (feof(stdin))
        pause();

    fgpu_server_deinit();

    return 0;
}
