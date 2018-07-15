/*******************************************************************************
    Copyright (c) 2016 NVIDIA Corporation

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

#include "uvm_linux.h"
#include "uvm8_hal_types.h"
#include "uvm8_hal.h"
#include "uvm8_gpu.h"
#include "clb06f.h"

void uvm_hal_maxwell_host_tlb_invalidate_all(uvm_push_t *push, uvm_gpu_phys_address_t pdb, NvU32 depth, uvm_membar_t membar)
{
    NvU32 target;
    NvU32 pdb_lo;
    NvU32 pdb_hi;

    UVM_ASSERT_MSG(pdb.aperture == UVM_APERTURE_VID || pdb.aperture == UVM_APERTURE_SYS, "aperture: %u", pdb.aperture);


    // Only Pascal+ supports invalidating down from a specific depth.
    (void)depth;

    (void)membar;

    if (pdb.aperture == UVM_APERTURE_VID)
        target = HWCONST(B06F, MEM_OP_C, TLB_INVALIDATE_TARGET, VID_MEM);
    else
        target = HWCONST(B06F, MEM_OP_C, TLB_INVALIDATE_TARGET, SYS_MEM_COHERENT);

    UVM_ASSERT_MSG(IS_ALIGNED(pdb.address, 1 << 12), "pdb 0x%llx\n", pdb.address);
    pdb.address >>= 12;
    pdb_lo = pdb.address & HWMASK(B06F, MEM_OP_C, TLB_INVALIDATE_ADDR_LO);
    pdb_hi = pdb.address >> HWSIZE(B06F, MEM_OP_C, TLB_INVALIDATE_ADDR_LO);

    NV_PUSH_2U(B06F, MEM_OP_C, target |
                               HWCONST(B06F, MEM_OP_C, TLB_INVALIDATE_PDB, ONE) |
                               HWCONST(B06F, MEM_OP_C, TLB_INVALIDATE_GPC, ENABLE) |
                               HWVALUE(B06F, MEM_OP_C, TLB_INVALIDATE_ADDR_LO, pdb_lo),
                     MEM_OP_D, HWCONST(B06F, MEM_OP_D, OPERATION, MMU_TLB_INVALIDATE) |
                               HWVALUE(B06F, MEM_OP_D, TLB_INVALIDATE_ADDR_HI, pdb_hi));
}
