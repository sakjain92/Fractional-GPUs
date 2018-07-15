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

#ifndef __UVM8_PROCESSORS_H__
#define __UVM8_PROCESSORS_H__

#include "uvm_linux.h"
#include "uvm_common.h"

#define UVM_MAX_UNIQUE_GPU_PAIRS SUM_FROM_0_TO_N(UVM_MAX_GPUS - 1)

// Static processor id for the CPU
#define UVM_CPU_ID (0)

// A unique uvm internal gpu id, guaranteed to be in the [1, UVM_MAX_PROCESSORS) range
typedef NvU32 uvm_gpu_id_t;

// Helper to iterate over all valid gpu ids
#define for_each_gpu_id(i) for (i = 1; i < UVM_MAX_PROCESSORS; ++i)

// Same as gpu id type but can also include the cpu id (UVM_CPU_ID)
typedef NvU32 uvm_processor_id_t;

// Helper for converting uvm_gpu_id_t to an array index.
static NvU32 uvm_gpu_index(uvm_gpu_id_t gpu_id)
{
    UVM_ASSERT(gpu_id >= 1);
    return gpu_id - 1;
}

// A collection of uvm processor ids
// Operated on with the uvm_processor_mask_*() functions below
typedef struct
{
    DECLARE_BITMAP(bitmap, UVM_MAX_PROCESSORS);
} uvm_processor_mask_t;

static bool uvm_processor_uuid_eq(const NvProcessorUuid *uuid1, const NvProcessorUuid *uuid2)
{
    return memcmp(uuid1, uuid2, sizeof(*uuid1)) == 0;
}

// Copies a UUID from source (src) to destination (dst).
static void uvm_processor_uuid_copy(NvProcessorUuid *dst, const NvProcessorUuid *src)
{
    memcpy(dst, src, sizeof(*dst));
}

void uvm_processor_uuid_from_id(NvProcessorUuid *uuid, uvm_processor_id_t id);

static bool uvm_processor_mask_test(const uvm_processor_mask_t *mask, uvm_processor_id_t id)
{
    UVM_ASSERT_MSG(id < UVM_MAX_PROCESSORS, "id %u\n", id);

    return test_bit(id, mask->bitmap);
}

// uvm_processor_mask_set and uvm_processor_mask_clear assume to modify the mask
// under a lock, so they use the non-atomic variants __set_bit and __clear_bit.
static void uvm_processor_mask_set(uvm_processor_mask_t *mask, uvm_processor_id_t id)
{
    UVM_ASSERT_MSG(id < UVM_MAX_PROCESSORS, "id %u\n", id);

    __set_bit(id, mask->bitmap);
}

static void uvm_processor_mask_clear(uvm_processor_mask_t *mask, uvm_processor_id_t id)
{
    UVM_ASSERT_MSG(id < UVM_MAX_PROCESSORS, "id %u\n", id);

    __clear_bit(id, mask->bitmap);
}

static bool uvm_processor_mask_test_and_set(uvm_processor_mask_t *mask, uvm_processor_id_t id)
{
    UVM_ASSERT_MSG(id < UVM_MAX_PROCESSORS, "id %u\n", id);

    return __test_and_set_bit(id, mask->bitmap);
}

static bool uvm_processor_mask_test_and_clear(uvm_processor_mask_t *mask, uvm_processor_id_t id)
{
    UVM_ASSERT_MSG(id < UVM_MAX_PROCESSORS, "id %u\n", id);

    return __test_and_clear_bit(id, mask->bitmap);
}

static void uvm_processor_mask_zero(uvm_processor_mask_t *mask)
{
    bitmap_zero(mask->bitmap, UVM_MAX_PROCESSORS);
}

static bool uvm_processor_mask_empty(const uvm_processor_mask_t *mask)
{
    return bitmap_empty(mask->bitmap, UVM_MAX_PROCESSORS);
}

static void uvm_processor_mask_copy(uvm_processor_mask_t *dst, const uvm_processor_mask_t *src)
{
    bitmap_copy(dst->bitmap, src->bitmap, UVM_MAX_PROCESSORS);
}

static void uvm_processor_mask_or(uvm_processor_mask_t *dst,
                                  const uvm_processor_mask_t *src1,
                                  const uvm_processor_mask_t *src2)
{
    bitmap_or(dst->bitmap, src1->bitmap, src2->bitmap, UVM_MAX_PROCESSORS);
}

static bool uvm_processor_mask_andnot(uvm_processor_mask_t *dst,
                                      const uvm_processor_mask_t *src1,
                                      const uvm_processor_mask_t *src2)
{
    return bitmap_andnot(dst->bitmap, src1->bitmap, src2->bitmap, UVM_MAX_PROCESSORS);
}

static void uvm_processor_mask_xor(uvm_processor_mask_t *dst,
                                   const uvm_processor_mask_t *src1,
                                   const uvm_processor_mask_t *src2)
{
    bitmap_xor(dst->bitmap, src1->bitmap, src2->bitmap, UVM_MAX_PROCESSORS);
}

// Returns the first id that's set in the mask or UVM_MAX_PROCESSORS if there isn't one.
static uvm_processor_id_t uvm_processor_mask_find_first_id(const uvm_processor_mask_t *mask)
{
    return find_first_bit(mask->bitmap, UVM_MAX_PROCESSORS);
}

// Returns the first id greater or equal to min_id that's set in the mask or
// UVM_MAX_PROCESSORS if there isn't one.
static uvm_processor_id_t uvm_processor_mask_find_next_id(const uvm_processor_mask_t *mask,
                                                          uvm_processor_id_t min_id)
{
    return find_next_bit(mask->bitmap, UVM_MAX_PROCESSORS, min_id);
}

// Returns the first id that's not set in the mask or UVM_MAX_PROCESSORS if there isn't one.
static uvm_processor_id_t uvm_processor_mask_find_first_unset_id(const uvm_processor_mask_t *mask)
{
    return find_first_zero_bit(mask->bitmap, UVM_MAX_PROCESSORS);
}

// Returns the first id greater or equal to min_id that's not set in the mask or
// UVM_MAX_PROCESSORS if there isn't one.
static uvm_processor_id_t uvm_processor_mask_find_next_unset_id(const uvm_processor_mask_t *mask,
                                                                uvm_processor_id_t min_id)
{
    return find_next_zero_bit(mask->bitmap, UVM_MAX_PROCESSORS, min_id);
}

static int uvm_processor_mask_and(uvm_processor_mask_t *mask_out,
                                  const uvm_processor_mask_t *mask_in1,
                                  const uvm_processor_mask_t *mask_in2)

{
    return bitmap_and(mask_out->bitmap, mask_in1->bitmap, mask_in2->bitmap, UVM_MAX_PROCESSORS);
}

static int uvm_processor_mask_equal(const uvm_processor_mask_t *mask_in1,
                                    const uvm_processor_mask_t *mask_in2)

{
    return bitmap_equal(mask_in1->bitmap, mask_in2->bitmap, UVM_MAX_PROCESSORS);
}

static int uvm_processor_mask_subset(const uvm_processor_mask_t *subset,
                                     const uvm_processor_mask_t *mask)

{
    return bitmap_subset(subset->bitmap, mask->bitmap, UVM_MAX_PROCESSORS);
}

// Get the number of processors set in the mask
static NvU32 uvm_processor_mask_get_count(const uvm_processor_mask_t *mask)
{
    return bitmap_weight(mask->bitmap, UVM_MAX_PROCESSORS);
}

// Get the number of GPUs set in the mask
static NvU32 uvm_processor_mask_get_gpu_count(const uvm_processor_mask_t *mask)
{
    NvU32 gpu_count = uvm_processor_mask_get_count(mask);
    if (uvm_processor_mask_test(mask, UVM_CPU_ID))
        --gpu_count;
    return gpu_count;
}

// Helper to iterate over all processor ids set in a mask
#define for_each_id_in_mask(id, mask)                              \
    for ((id) = uvm_processor_mask_find_first_id(mask);            \
         (id) != UVM_MAX_PROCESSORS;                              \
         (id) = uvm_processor_mask_find_next_id((mask), (id) + 1))

// Helper to iterate over all GPU ids set in a mask
#define for_each_gpu_id_in_mask(gpu_id, mask)                                \
    for ((gpu_id) = uvm_processor_mask_find_next_id((mask), UVM_CPU_ID + 1); \
         (gpu_id) != UVM_MAX_PROCESSORS;                                    \
         (gpu_id) = uvm_processor_mask_find_next_id((mask), (gpu_id) + 1))

#endif
