/*******************************************************************************
    Copyright (c) 2015 NVIDIA Corporation

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to
    deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
    sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be
        included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.

*******************************************************************************/

#ifndef __UVM8_PROCFS_H__
#define __UVM8_PROCFS_H__

#include "uvm8_forward_decl.h"
#include "uvm_linux.h"
#include "nv-procfs.h"

NV_STATUS uvm_procfs_init(void);
void uvm_procfs_exit(void);

// Is procfs enabled at all?
bool uvm_procfs_is_enabled(void);

// Is debug procfs enabled? This indicates that debug procfs files should be created.
bool uvm_procfs_is_debug_enabled(void);

struct proc_dir_entry *uvm_procfs_get_gpu_base_dir(void);

void uvm_procfs_destroy_entry(struct proc_dir_entry *entry);

// Helper for printing into a seq_file if it's not NULL and UVM_DBG_PRINT otherwise.
// Useful when sharing a print function for both debug output and procfs output.
#define UVM_SEQ_OR_DBG_PRINT(seq_file, format, ...)             \
    do {                                                        \
        if (seq_file != NULL)                                   \
            seq_printf(seq_file, format, ##__VA_ARGS__);        \
        else                                                    \
            UVM_DBG_PRINT(format, ##__VA_ARGS__);               \
    } while (0)

#endif // __UVM8_PROCFS_H__
