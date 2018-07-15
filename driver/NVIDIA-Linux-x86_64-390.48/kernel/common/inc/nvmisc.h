 /***************************************************************************\
|*                                                                           *|
|*       Copyright 1993-2015 NVIDIA, Corporation.  All rights reserved.      *|
|*                                                                           *|
|*     NOTICE TO USER:   The source code  is copyrighted under  U.S. and     *|
|*     international laws.  NVIDIA, Corp. of Sunnyvale,  California owns     *|
|*     copyrights, patents, and has design patents pending on the design     *|
|*     and  interface  of the NV chips.   Users and  possessors  of this     *|
|*     source code are hereby granted a nonexclusive, royalty-free copy-     *|
|*     right  and design patent license  to use this code  in individual     *|
|*     and commercial software.                                              *|
|*                                                                           *|
|*     Any use of this source code must include,  in the user documenta-     *|
|*     tion and  internal comments to the code,  notices to the end user     *|
|*     as follows:                                                           *|
|*                                                                           *|
|*     Copyright  1993-2015  NVIDIA,  Corporation.   NVIDIA  has  design     *|
|*     patents and patents pending in the U.S. and foreign countries.        *|
|*                                                                           *|
|*     NVIDIA, CORPORATION MAKES NO REPRESENTATION ABOUT THE SUITABILITY     *|
|*     OF THIS SOURCE CODE FOR ANY PURPOSE. IT IS PROVIDED "AS IS" WITH-     *|
|*     OUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.  NVIDIA, CORPORATION     *|
|*     DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOURCE CODE, INCLUD-     *|
|*     ING ALL IMPLIED WARRANTIES  OF MERCHANTABILITY  AND FITNESS FOR A     *|
|*     PARTICULAR  PURPOSE.  IN NO EVENT  SHALL NVIDIA,  CORPORATION  BE     *|
|*     LIABLE FOR ANY SPECIAL,  INDIRECT,  INCIDENTAL,  OR CONSEQUENTIAL     *|
|*     DAMAGES, OR ANY DAMAGES  WHATSOEVER  RESULTING  FROM LOSS OF USE,     *|
|*     DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR     *|
|*     OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION  WITH THE     *|
|*     USE OR PERFORMANCE OF THIS SOURCE CODE.                               *|
|*                                                                           *|
|*     RESTRICTED RIGHTS LEGEND:  Use, duplication, or disclosure by the     *|
|*     Government is subject  to restrictions  as set forth  in subpara-     *|
|*     graph (c) (1) (ii) of the Rights  in Technical Data  and Computer     *|
|*     Software  clause  at DFARS  52.227-7013 and in similar clauses in     *|
|*     the FAR and NASA FAR Supplement.                                      *|
|*                                                                           *|
 \***************************************************************************/

/*
 * nvmisc.h
 */
#ifndef __NV_MISC_H
#define __NV_MISC_H

#ifdef __cplusplus
extern "C" {
#endif //__cplusplus

#include "nvtypes.h"

#if !defined(NVIDIA_UNDEF_LEGACY_BIT_MACROS)
//
// Miscellaneous macros useful for bit field manipulations
//
// STUPID HACK FOR CL 19434692.  Will revert when fix CL is delivered bfm -> chips_a.    
#ifndef BIT    
#define BIT(b)                  (1<<(b))
#endif    
#define BIT32(b)                ((NvU32)1<<(b))
#define BIT64(b)                ((NvU64)1<<(b))

#endif

//
// It is recommended to use the following bit macros to avoid macro name
// collisions with other src code bases.
//
#define NVBIT(b)                  (1<<(b))
#define NVBIT32(b)                ((NvU32)1<<(b))
#define NVBIT64(b)                ((NvU64)1<<(b))

// Index of the 'on' bit (assuming that there is only one).
// Even if multiple bits are 'on', result is in range of 0-31.
#define BIT_IDX_32(n)                   \
   ((((n) & 0xFFFF0000)? 0x10: 0) |     \
    (((n) & 0xFF00FF00)? 0x08: 0) |     \
    (((n) & 0xF0F0F0F0)? 0x04: 0) |     \
    (((n) & 0xCCCCCCCC)? 0x02: 0) |     \
    (((n) & 0xAAAAAAAA)? 0x01: 0) )

// Index of the 'on' bit (assuming that there is only one).
// Even if multiple bits are 'on', result is in range of 0-63.
#define BIT_IDX_64(n)                              \
   ((((n) & 0xFFFFFFFF00000000ULL)? 0x20: 0) |     \
    (((n) & 0xFFFF0000FFFF0000ULL)? 0x10: 0) |     \
    (((n) & 0xFF00FF00FF00FF00ULL)? 0x08: 0) |     \
    (((n) & 0xF0F0F0F0F0F0F0F0ULL)? 0x04: 0) |     \
    (((n) & 0xCCCCCCCCCCCCCCCCULL)? 0x02: 0) |     \
    (((n) & 0xAAAAAAAAAAAAAAAAULL)? 0x01: 0) )

// Desctructive, assumes only one bit set in n32
// Deprecated in favor of BIT_IDX_32.
#define IDX_32(n32)                     \
{                                       \
    NvU32 idx = 0;                      \
    if ((n32) & 0xFFFF0000) idx += 16;  \
    if ((n32) & 0xFF00FF00) idx += 8;   \
    if ((n32) & 0xF0F0F0F0) idx += 4;   \
    if ((n32) & 0xCCCCCCCC) idx += 2;   \
    if ((n32) & 0xAAAAAAAA) idx += 1;   \
    (n32) = idx;                        \
}

// tegra mobile uses nvmisc_macros.h and can't access nvmisc.h... and sometimes both get included.
#ifndef _NVMISC_MACROS_H
// Use Coverity Annotation to mark issues as false positives/ignore when using single bit defines.
#define DRF_ISBIT(bitval,drf)                \
        ( /* coverity[identical_branches] */ \
          bitval ? drf )
#define DEVICE_BASE(d)          (0?d)  // what's up with this name? totally non-parallel to the macros below
#define DEVICE_EXTENT(d)        (1?d)  // what's up with this name? totally non-parallel to the macros below
#define DRF_BASE(drf)           (0?drf)  // much better
#define DRF_EXTENT(drf)         (1?drf)  // much better
#ifdef NV_MISRA_COMPLIANCE_REQUIRED
#define DRF_SHIFT(drf)          (((NvU32)DRF_BASE(drf)) % 32U)
#define DRF_SHIFT_RT(drf)       (((NvU32)DRF_EXTENT(drf)) % 32U)
#define DRF_MASK(drf)           (0xFFFFFFFFU>>(31U-(((NvU32)DRF_EXTENT(drf)) % 32U)+(((NvU32)DRF_BASE(drf)) % 32U)))
#define DRF_DEF(d,r,f,c)        (((NvU32)(NV ## d ## r ## f ## c))<<DRF_SHIFT(NV ## d ## r ## f))
#else
#define DRF_SHIFT(drf)          ((DRF_ISBIT(0,drf)) % 32)
#define DRF_SHIFT_RT(drf)       ((DRF_ISBIT(1,drf)) % 32)
#define DRF_MASK(drf)           (0xFFFFFFFF>>(31-((DRF_ISBIT(1,drf)) % 32)+((DRF_ISBIT(0,drf)) % 32)))
#define DRF_DEF(d,r,f,c)        ((NV ## d ## r ## f ## c)<<DRF_SHIFT(NV ## d ## r ## f))
#endif
#define DRF_SHIFTMASK(drf)      (DRF_MASK(drf)<<(DRF_SHIFT(drf)))
#define DRF_SIZE(drf)           (DRF_EXTENT(drf)-DRF_BASE(drf)+1)

#define DRF_NUM(d,r,f,n)        (((n)&DRF_MASK(NV ## d ## r ## f))<<DRF_SHIFT(NV ## d ## r ## f))
#define DRF_VAL(d,r,f,v)        (((v)>>DRF_SHIFT(NV ## d ## r ## f))&DRF_MASK(NV ## d ## r ## f))
#endif

// Signed version of DRF_VAL, which takes care of extending sign bit.
#define DRF_VAL_SIGNED(d,r,f,v) (((DRF_VAL(d,r,f,v) ^ (NVBIT(DRF_SIZE(NV ## d ## r ## f)-1)))) - (NVBIT(DRF_SIZE(NV ## d ## r ## f)-1)))
#define DRF_IDX_DEF(d,r,f,i,c)  ((NV ## d ## r ## f ## c)<<DRF_SHIFT(NV##d##r##f(i)))
#define DRF_IDX_OFFSET_DEF(d,r,f,i,o,c)  ((NV ## d ## r ## f ## c)<<DRF_SHIFT(NV##d##r##f(i,o)))
#define DRF_IDX_NUM(d,r,f,i,n)  (((n)&DRF_MASK(NV##d##r##f(i)))<<DRF_SHIFT(NV##d##r##f(i)))
#define DRF_IDX_VAL(d,r,f,i,v)  (((v)>>DRF_SHIFT(NV##d##r##f(i)))&DRF_MASK(NV##d##r##f(i)))
#define DRF_IDX_OFFSET_VAL(d,r,f,i,o,v)  (((v)>>DRF_SHIFT(NV##d##r##f(i,o)))&DRF_MASK(NV##d##r##f(i,o)))
// Fractional version of DRF_VAL which reads Fx.y fixed point number (x.y)*z
#define DRF_VAL_FRAC(d,r,x,y,v,z) ((DRF_VAL(d,r,x,v)*z) + ((DRF_VAL(d,r,y,v)*z) / (1<<DRF_SIZE(NV##d##r##y))))

//
// 64 Bit Versions
//
#define DRF_SHIFT64(drf)                ((DRF_ISBIT(0,drf)) % 64)
#define DRF_MASK64(drf)                 (NV_U64_MAX>>(63-((DRF_ISBIT(1,drf)) % 64)+((DRF_ISBIT(0,drf)) % 64)))
#define DRF_SHIFTMASK64(drf)            (DRF_MASK64(drf)<<(DRF_SHIFT64(drf)))

#define DRF_DEF64(d,r,f,c)              (((NvU64)(NV ## d ## r ## f ## c))<<DRF_SHIFT64(NV ## d ## r ## f))
#define DRF_NUM64(d,r,f,n)              ((((NvU64)(n))&DRF_MASK64(NV ## d ## r ## f))<<DRF_SHIFT64(NV ## d ## r ## f))
#define DRF_VAL64(d,r,f,v)              ((((NvU64)(v))>>DRF_SHIFT64(NV ## d ## r ## f))&DRF_MASK64(NV ## d ## r ## f))

#define DRF_VAL_SIGNED64(d,r,f,v)       (((DRF_VAL64(d,r,f,v) ^ (NVBIT64(DRF_SIZE(NV ## d ## r ## f)-1)))) - (NVBIT64(DRF_SIZE(NV ## d ## r ## f)-1)))
#define DRF_IDX_DEF64(d,r,f,i,c)        (((NvU64)(NV ## d ## r ## f ## c))<<DRF_SHIFT64(NV##d##r##f(i)))
#define DRF_IDX_OFFSET_DEF64(d,r,f,i,o,c) ((NvU64)(NV ## d ## r ## f ## c)<<DRF_SHIFT64(NV##d##r##f(i,o)))
#define DRF_IDX_NUM64(d,r,f,i,n)        ((((NvU64)(n))&DRF_MASK64(NV##d##r##f(i)))<<DRF_SHIFT64(NV##d##r##f(i)))
#define DRF_IDX_VAL64(d,r,f,i,v)        ((((NvU64)(v))>>DRF_SHIFT64(NV##d##r##f(i)))&DRF_MASK64(NV##d##r##f(i)))
#define DRF_IDX_OFFSET_VAL64(d,r,f,i,o,v) (((NvU64)(v)>>DRF_SHIFT64(NV##d##r##f(i,o)))&DRF_MASK64(NV##d##r##f(i,o)))

#define FLD_SET_DRF64(d,r,f,c,v)        (((NvU64)(v) & ~DRF_SHIFTMASK64(NV##d##r##f)) | DRF_DEF64(d,r,f,c))
#define FLD_SET_DRF_NUM64(d,r,f,n,v)    ((((NvU64)(v)) & ~DRF_SHIFTMASK64(NV##d##r##f)) | DRF_NUM64(d,r,f,n))
#define FLD_IDX_SET_DRF64(d,r,f,i,c,v)  (((NvU64)(v) & ~DRF_SHIFTMASK64(NV##d##r##f(i))) | DRF_IDX_DEF64(d,r,f,i,c))
#define FLD_IDX_OFFSET_SET_DRF64(d,r,f,i,o,c,v) (((NvU64)(v) & ~DRF_SHIFTMASK64(NV##d##r##f(i,o))) | DRF_IDX_OFFSET_DEF64(d,r,f,i,o,c))
#define FLD_IDX_SET_DRF_DEF64(d,r,f,i,c,v) (((NvU64)(v) & ~DRF_SHIFTMASK64(NV##d##r##f(i))) | DRF_IDX_DEF64(d,r,f,i,c))
#define FLD_IDX_SET_DRF_NUM64(d,r,f,i,n,v) (((NvU64)(v) & ~DRF_SHIFTMASK64(NV##d##r##f(i))) | DRF_IDX_NUM64(d,r,f,i,n))
#define FLD_SET_DRF_IDX64(d,r,f,c,i,v)  (((NvU64)(v) & ~DRF_SHIFTMASK64(NV##d##r##f)) | DRF_DEF64(d,r,f,c(i)))

#define FLD_TEST_DRF64(d,r,f,c,v)       (DRF_VAL64(d, r, f, v) == NV##d##r##f##c)
#define FLD_TEST_DRF_AND64(d,r,f,c,v)   (DRF_VAL64(d, r, f, v) & NV##d##r##f##c)
#define FLD_TEST_DRF_NUM64(d,r,f,n,v)   (DRF_VAL64(d, r, f, v) == n)
#define FLD_IDX_TEST_DRF64(d,r,f,i,c,v) (DRF_IDX_VAL64(d, r, f, i, v) == NV##d##r##f##c)
#define FLD_IDX_OFFSET_TEST_DRF64(d,r,f,i,o,c,v) (DRF_IDX_OFFSET_VAL64(d, r, f, i, o, v) == NV##d##r##f##c)

//
// 32 Bit Versions
//

#define FLD_SET_DRF(d,r,f,c,v)            ((v & ~DRF_SHIFTMASK(NV##d##r##f)) | DRF_DEF(d,r,f,c))
#define FLD_SET_DRF_NUM(d,r,f,n,v)        ((v & ~DRF_SHIFTMASK(NV##d##r##f)) | DRF_NUM(d,r,f,n))
// FLD_SET_DRF_DEF is deprecated! Use an explicit assignment with FLD_SET_DRF instead.
#define FLD_SET_DRF_DEF(d,r,f,c,v)        (v = (v & ~DRF_SHIFTMASK(NV##d##r##f)) | DRF_DEF(d,r,f,c))
#define FLD_IDX_SET_DRF(d,r,f,i,c,v)      ((v & ~DRF_SHIFTMASK(NV##d##r##f(i))) | DRF_IDX_DEF(d,r,f,i,c))
#define FLD_IDX_OFFSET_SET_DRF(d,r,f,i,o,c,v)      ((v & ~DRF_SHIFTMASK(NV##d##r##f(i,o))) | DRF_IDX_OFFSET_DEF(d,r,f,i,o,c))
#define FLD_IDX_SET_DRF_DEF(d,r,f,i,c,v)  ((v & ~DRF_SHIFTMASK(NV##d##r##f(i))) | DRF_IDX_DEF(d,r,f,i,c))
#define FLD_IDX_SET_DRF_NUM(d,r,f,i,n,v)  ((v & ~DRF_SHIFTMASK(NV##d##r##f(i))) | DRF_IDX_NUM(d,r,f,i,n))
#define FLD_SET_DRF_IDX(d,r,f,c,i,v)      ((v & ~DRF_SHIFTMASK(NV##d##r##f)) | DRF_DEF(d,r,f,c(i)))

#define FLD_TEST_DRF(d,r,f,c,v)       ((DRF_VAL(d, r, f, v) == NV##d##r##f##c))
#define FLD_TEST_DRF_AND(d,r,f,c,v)   ((DRF_VAL(d, r, f, v) & NV##d##r##f##c))
#define FLD_TEST_DRF_NUM(d,r,f,n,v)   ((DRF_VAL(d, r, f, v) == n))
#define FLD_IDX_TEST_DRF(d,r,f,i,c,v) ((DRF_IDX_VAL(d, r, f, i, v) == NV##d##r##f##c))
#define FLD_IDX_OFFSET_TEST_DRF(d,r,f,i,o,c,v) ((DRF_IDX_OFFSET_VAL(d, r, f, i, o, v) == NV##d##r##f##c))

#define REF_DEF(drf,d)            (((drf ## d)&DRF_MASK(drf))<<DRF_SHIFT(drf))
#define REF_VAL(drf,v)            (((v)>>DRF_SHIFT(drf))&DRF_MASK(drf))
#define REF_NUM(drf,n)            (((n)&DRF_MASK(drf))<<DRF_SHIFT(drf))
#define FLD_TEST_REF(drf,c,v)     (REF_VAL(drf, v) == drf##c)
#define FLD_TEST_REF_AND(drf,c,v) (REF_VAL(drf, v) & drf##c)
#define FLD_SET_REF_NUM(drf,n,v)  (((v) & ~DRF_SHIFTMASK(drf)) | REF_NUM(drf,n))

#define CR_DRF_DEF(d,r,f,c)     ((CR ## d ## r ## f ## c)<<DRF_SHIFT(CR ## d ## r ## f))
#define CR_DRF_NUM(d,r,f,n)     (((n)&DRF_MASK(CR ## d ## r ## f))<<DRF_SHIFT(CR ## d ## r ## f))
#define CR_DRF_VAL(d,r,f,v)     (((v)>>DRF_SHIFT(CR ## d ## r ## f))&DRF_MASK(CR ## d ## r ## f))

// Multi-word (MW) field manipulations.  For multi-word structures (e.g., Fermi SPH),
// fields may have bit numbers beyond 32.  To avoid errors using "classic" multi-word macros,
// all the field extents are defined as "MW(X)".  For example, MW(127:96) means
// the field is in bits 0-31 of word number 3 of the structure.
//
// DRF_VAL_MW() macro is meant to be used for native endian 32-bit aligned 32-bit word data,
// not for byte stream data.
//
// DRF_VAL_BS() macro is for byte stream data used in fbQueryBIOS_XXX().
//
#define DRF_EXPAND_MW(drf)         drf                          // used to turn "MW(a:b)" into "a:b"
#define DRF_PICK_MW(drf,v)         (v?DRF_EXPAND_##drf)         // picks low or high bits
#define DRF_WORD_MW(drf)           (DRF_PICK_MW(drf,0)/32)      // which word in a multi-word array
#define DRF_BASE_MW(drf)           (DRF_PICK_MW(drf,0)%32)      // which start bit in the selected word?
#define DRF_EXTENT_MW(drf)         (DRF_PICK_MW(drf,1)%32)      // which end bit in the selected word
#define DRF_SHIFT_MW(drf)          (DRF_PICK_MW(drf,0)%32)
#define DRF_MASK_MW(drf)           (0xFFFFFFFF>>((31-(DRF_EXTENT_MW(drf))+(DRF_BASE_MW(drf)))%32))
#define DRF_SHIFTMASK_MW(drf)      ((DRF_MASK_MW(drf))<<(DRF_SHIFT_MW(drf)))
#define DRF_SIZE_MW(drf)           (DRF_EXTENT_MW(drf)-DRF_BASE_MW(drf)+1)

#define DRF_DEF_MW(d,r,f,c)        ((NV##d##r##f##c) << DRF_SHIFT_MW(NV##d##r##f))
#define DRF_NUM_MW(d,r,f,n)        (((n)&DRF_MASK_MW(NV##d##r##f))<<DRF_SHIFT_MW(NV##d##r##f))
//
// DRF_VAL_MW is the ONLY multi-word macro which supports spanning. No other MW macro supports spanning currently
//
#define DRF_VAL_MW_1WORD(d,r,f,v)       ((((v)[DRF_WORD_MW(NV##d##r##f)])>>DRF_SHIFT_MW(NV##d##r##f))&DRF_MASK_MW(NV##d##r##f))
#define DRF_SPANS(drf)                  ((DRF_PICK_MW(drf,0)/32) != (DRF_PICK_MW(drf,1)/32))
#define DRF_WORD_MW_LOW(drf)            (DRF_PICK_MW(drf,0)/32)
#define DRF_WORD_MW_HIGH(drf)           (DRF_PICK_MW(drf,1)/32)
#define DRF_MASK_MW_LOW(drf)            (0xFFFFFFFF)
#define DRF_MASK_MW_HIGH(drf)           (0xFFFFFFFF>>(31-(DRF_EXTENT_MW(drf))))
#define DRF_SHIFT_MW_LOW(drf)           (DRF_PICK_MW(drf,0)%32)
#define DRF_SHIFT_MW_HIGH(drf)          (0)
#define DRF_MERGE_SHIFT(drf)            ((32-((DRF_PICK_MW(drf,0)%32)))%32)
#define DRF_VAL_MW_2WORD(d,r,f,v)       (((((v)[DRF_WORD_MW_LOW(NV##d##r##f)])>>DRF_SHIFT_MW_LOW(NV##d##r##f))&DRF_MASK_MW_LOW(NV##d##r##f)) | \
    (((((v)[DRF_WORD_MW_HIGH(NV##d##r##f)])>>DRF_SHIFT_MW_HIGH(NV##d##r##f))&DRF_MASK_MW_HIGH(NV##d##r##f)) << DRF_MERGE_SHIFT(NV##d##r##f)))
#define DRF_VAL_MW(d,r,f,v)             ( DRF_SPANS(NV##d##r##f) ? DRF_VAL_MW_2WORD(d,r,f,v) : DRF_VAL_MW_1WORD(d,r,f,v) )

#define DRF_IDX_DEF_MW(d,r,f,i,c)  ((NV##d##r##f##c)<<DRF_SHIFT_MW(NV##d##r##f(i)))
#define DRF_IDX_NUM_MW(d,r,f,i,n)  (((n)&DRF_MASK_MW(NV##d##r##f(i)))<<DRF_SHIFT_MW(NV##d##r##f(i)))
#define DRF_IDX_VAL_MW(d,r,f,i,v)  ((((v)[DRF_WORD_MW(NV##d##r##f(i))])>>DRF_SHIFT_MW(NV##d##r##f(i)))&DRF_MASK_MW(NV##d##r##f(i)))

//
// Logically OR all DRF_DEF constants indexed from zero to s (semiinclusive).
// Caution: Target variable v must be pre-initialized.
//
#define FLD_IDX_OR_DRF_DEF(d,r,f,c,s,v)                 \
do                                                      \
{   NvU32 idx;                                          \
    for (idx = 0; idx < (NV ## d ## r ## f ## s); ++idx)\
    {                                                   \
        v |= DRF_IDX_DEF(d,r,f,idx,c);                  \
    }                                                   \
} while(0)


#define FLD_MERGE_MW(drf,n,v)               (((v)[DRF_WORD_MW(drf)] & ~DRF_SHIFTMASK_MW(drf)) | n)
#define FLD_ASSIGN_MW(drf,n,v)              ((v)[DRF_WORD_MW(drf)] = FLD_MERGE_MW(drf, n, v))
#define FLD_IDX_MERGE_MW(drf,i,n,v)         (((v)[DRF_WORD_MW(drf(i))] & ~DRF_SHIFTMASK_MW(drf(i))) | n)
#define FLD_IDX_ASSIGN_MW(drf,i,n,v)        ((v)[DRF_WORD_MW(drf(i))] = FLD_MERGE_MW(drf(i), n, v))

#define FLD_SET_DRF_MW(d,r,f,c,v)               FLD_MERGE_MW(NV##d##r##f, DRF_DEF_MW(d,r,f,c), v)
#define FLD_SET_DRF_NUM_MW(d,r,f,n,v)           FLD_ASSIGN_MW(NV##d##r##f, DRF_NUM_MW(d,r,f,n), v)
#define FLD_SET_DRF_DEF_MW(d,r,f,c,v)           FLD_ASSIGN_MW(NV##d##r##f, DRF_DEF_MW(d,r,f,c), v)
#define FLD_IDX_SET_DRF_MW(d,r,f,i,c,v)         FLD_IDX_MERGE_MW(NV##d##r##f, i, DRF_IDX_DEF_MW(d,r,f,i,c), v)
#define FLD_IDX_SET_DRF_DEF_MW(d,r,f,i,c,v)    FLD_IDX_MERGE_MW(NV##d##r##f, i, DRF_IDX_DEF_MW(d,r,f,i,c), v)
#define FLD_IDX_SET_DRF_NUM_MW(d,r,f,i,n,v)     FLD_IDX_ASSIGN_MW(NV##d##r##f, i, DRF_IDX_NUM_MW(d,r,f,i,n), v)

#define FLD_TEST_DRF_MW(d,r,f,c,v)          ((DRF_VAL_MW(d, r, f, v) == NV##d##r##f##c))
#define FLD_TEST_DRF_NUM_MW(d,r,f,n,v)      ((DRF_VAL_MW(d, r, f, v) == n))
#define FLD_IDX_TEST_DRF_MW(d,r,f,i,c,v)    ((DRF_IDX_VAL_MW(d, r, f, i, v) == NV##d##r##f##c))

#define DRF_VAL_BS(d,r,f,v)                 ( DRF_SPANS(NV##d##r##f) ? DRF_VAL_BS_2WORD(d,r,f,v) : DRF_VAL_BS_1WORD(d,r,f,v) )

//-------------------------------------------------------------------------//
//                                                                         //
// Common defines for engine register reference wrappers                   //
//                                                                         //
// New engine addressing can be created like:                              //
// #define ENG_REG_PMC(o,d,r)                      NV##d##r                //
// #define ENG_IDX_REG_CE(o,d,i,r)                 CE_MAP(o,r,i)           //
//                                                                         //
// See FB_FBPA* for more examples                                          //
//-------------------------------------------------------------------------//

#define ENG_RD_REG(g,o,d,r)             GPU_REG_RD32(g, ENG_REG##d(o,d,r))
#define ENG_WR_REG(g,o,d,r,v)           GPU_REG_WR32(g, ENG_REG##d(o,d,r), v)
#define ENG_RD_DRF(g,o,d,r,f)           ((GPU_REG_RD32(g, ENG_REG##d(o,d,r))>>GPU_DRF_SHIFT(NV ## d ## r ## f))&GPU_DRF_MASK(NV ## d ## r ## f))
#define ENG_WR_DRF_DEF(g,o,d,r,f,c)     GPU_REG_WR32(g, ENG_REG##d(o,d,r),(GPU_REG_RD32(g,ENG_REG##d(o,d,r))&~(GPU_DRF_MASK(NV##d##r##f)<<GPU_DRF_SHIFT(NV##d##r##f)))|GPU_DRF_DEF(d,r,f,c))
#define ENG_WR_DRF_NUM(g,o,d,r,f,n)     GPU_REG_WR32(g, ENG_REG##d(o,d,r),(GPU_REG_RD32(g,ENG_REG##d(o,d,r))&~(GPU_DRF_MASK(NV##d##r##f)<<GPU_DRF_SHIFT(NV##d##r##f)))|GPU_DRF_NUM(d,r,f,n))
#define ENG_TEST_DRF_DEF(g,o,d,r,f,c)   (ENG_RD_DRF(g, o, d, r, f) == NV##d##r##f##c)

#define ENG_RD_IDX_DRF(g,o,d,r,f,i)     ((GPU_REG_RD32(g, ENG_REG##d(o,d,r(i)))>>GPU_DRF_SHIFT(NV ## d ## r ## f))&GPU_DRF_MASK(NV ## d ## r ## f))
#define ENG_TEST_IDX_DRF_DEF(g,o,d,r,f,c,i) (ENG_RD_IDX_DRF(g, o, d, r, f, i) == NV##d##r##f##c)

#define ENG_IDX_RD_REG(g,o,d,i,r)       GPU_REG_RD32(g, ENG_IDX_REG##d(o,d,i,r))
#define ENG_IDX_WR_REG(g,o,d,i,r,v)     GPU_REG_WR32(g, ENG_IDX_REG##d(o,d,i,r), v)

#define ENG_IDX_RD_DRF(g,o,d,i,r,f)     ((GPU_REG_RD32(g, ENG_IDX_REG##d(o,d,i,r))>>GPU_DRF_SHIFT(NV ## d ## r ## f))&GPU_DRF_MASK(NV ## d ## r ## f))

//
// DRF_READ_1WORD_BS() and DRF_READ_1WORD_BS_HIGH() do not read beyond the bytes that contain
// the requested value. Reading beyond the actual data causes a page fault panic when the
// immediately following page happened to be protected or not mapped.
//
#define DRF_VAL_BS_1WORD(d,r,f,v)           ((DRF_READ_1WORD_BS(d,r,f,v)>>DRF_SHIFT_MW(NV##d##r##f))&DRF_MASK_MW(NV##d##r##f))
#define DRF_VAL_BS_2WORD(d,r,f,v)           (((DRF_READ_4BYTE_BS(NV##d##r##f,v)>>DRF_SHIFT_MW_LOW(NV##d##r##f))&DRF_MASK_MW_LOW(NV##d##r##f)) | \
    (((DRF_READ_1WORD_BS_HIGH(d,r,f,v)>>DRF_SHIFT_MW_HIGH(NV##d##r##f))&DRF_MASK_MW_HIGH(NV##d##r##f)) << DRF_MERGE_SHIFT(NV##d##r##f)))

#define DRF_READ_1BYTE_BS(drf,v)            ((NvU32)(((const NvU8*)(v))[DRF_WORD_MW(drf)*4]))
#define DRF_READ_2BYTE_BS(drf,v)            (DRF_READ_1BYTE_BS(drf,v)| \
    ((NvU32)(((const NvU8*)(v))[DRF_WORD_MW(drf)*4+1])<<8))
#define DRF_READ_3BYTE_BS(drf,v)            (DRF_READ_2BYTE_BS(drf,v)| \
    ((NvU32)(((const NvU8*)(v))[DRF_WORD_MW(drf)*4+2])<<16))
#define DRF_READ_4BYTE_BS(drf,v)            (DRF_READ_3BYTE_BS(drf,v)| \
    ((NvU32)(((const NvU8*)(v))[DRF_WORD_MW(drf)*4+3])<<24))

#define DRF_READ_1BYTE_BS_HIGH(drf,v)       ((NvU32)(((const NvU8*)(v))[DRF_WORD_MW_HIGH(drf)*4]))
#define DRF_READ_2BYTE_BS_HIGH(drf,v)       (DRF_READ_1BYTE_BS_HIGH(drf,v)| \
    ((NvU32)(((const NvU8*)(v))[DRF_WORD_MW_HIGH(drf)*4+1])<<8))
#define DRF_READ_3BYTE_BS_HIGH(drf,v)       (DRF_READ_2BYTE_BS_HIGH(drf,v)| \
    ((NvU32)(((const NvU8*)(v))[DRF_WORD_MW_HIGH(drf)*4+2])<<16))
#define DRF_READ_4BYTE_BS_HIGH(drf,v)       (DRF_READ_3BYTE_BS_HIGH(drf,v)| \
    ((NvU32)(((const NvU8*)(v))[DRF_WORD_MW_HIGH(drf)*4+3])<<24))

// Calculate 2^n - 1 and avoid shift counter overflow
//
// On Windows amd64, 64 << 64 => 1
//
#define NV_TWO_N_MINUS_ONE(n) (((1ULL<<(n/2))<<((n+1)/2))-1)

#define DRF_READ_1WORD_BS(d,r,f,v) \
    ((DRF_EXTENT_MW(NV##d##r##f)<8)?DRF_READ_1BYTE_BS(NV##d##r##f,(v)): \
    ((DRF_EXTENT_MW(NV##d##r##f)<16)?DRF_READ_2BYTE_BS(NV##d##r##f,(v)): \
    ((DRF_EXTENT_MW(NV##d##r##f)<24)?DRF_READ_3BYTE_BS(NV##d##r##f,(v)): \
    DRF_READ_4BYTE_BS(NV##d##r##f,(v)))))

#define DRF_READ_1WORD_BS_HIGH(d,r,f,v) \
    ((DRF_EXTENT_MW(NV##d##r##f)<8)?DRF_READ_1BYTE_BS_HIGH(NV##d##r##f,(v)): \
    ((DRF_EXTENT_MW(NV##d##r##f)<16)?DRF_READ_2BYTE_BS_HIGH(NV##d##r##f,(v)): \
    ((DRF_EXTENT_MW(NV##d##r##f)<24)?DRF_READ_3BYTE_BS_HIGH(NV##d##r##f,(v)): \
    DRF_READ_4BYTE_BS_HIGH(NV##d##r##f,(v)))))

#define BIN_2_GRAY(n)           ((n)^((n)>>1))
// operates on a 64-bit data type
#define GRAY_2_BIN_64b(n)       (n)^=(n)>>1; (n)^=(n)>>2; (n)^=(n)>>4; (n)^=(n)>>8; (n)^=(n)>>16; (n)^=(n)>>32;

#define LOWESTBIT(x)            ( (x) &  (((x)-1) ^ (x)) )
// Destructive operation on n32
#define HIGHESTBIT(n32)     \
{                           \
    HIGHESTBITIDX_32(n32);  \
    n32 = NVBIT(n32);       \
}
#define ONEBITSET(x)            ( (x) && (((x) & ((x)-1)) == 0) )

// Destructive operation on n32
#define NUMSETBITS_32(n32)                                         \
{                                                                  \
    n32 = n32 - ((n32 >> 1) & 0x55555555);                         \
    n32 = (n32 & 0x33333333) + ((n32 >> 2) & 0x33333333);          \
    n32 = (((n32 + (n32 >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;  \
}

/*!
 * Calculate number of bits set in a 32-bit unsigned integer.
 * Pure typesafe alternative to @ref NUMSETBITS_32.
 */
static NV_FORCEINLINE NvU32
nvPopCount32(const NvU32 x)
{
    NvU32 temp = x;
    temp = temp - ((temp >> 1) & 0x55555555);
    temp = (temp & 0x33333333) + ((temp >> 2) & 0x33333333);
    temp = (((temp + (temp >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
    return temp;
}

/*!
 * Calculate number of bits set in a 64-bit unsigned integer.
 */
static NV_FORCEINLINE NvU32
nvPopCount64(const NvU64 x)
{
    NvU64 temp = x;
    temp = temp - ((temp >> 1) & 0x5555555555555555ull);
    temp = (temp & 0x3333333333333333ull) + ((temp >> 2) & 0x3333333333333333ull);
    temp = (temp + (temp >> 4)) & 0x0F0F0F0F0F0F0F0Full;
    temp = (temp * 0x0101010101010101ull) >> 56;
    return (NvU32)temp;
}

/*!
 * Determine how many bits are set below a bit index within a mask.
 * This assigns a dense ordering to the set bits in the mask.
 *
 * For example the mask 0xCD contains 5 set bits:
 *     nvMaskPos32(0xCD, 0) == 0
 *     nvMaskPos32(0xCD, 2) == 1
 *     nvMaskPos32(0xCD, 3) == 2
 *     nvMaskPos32(0xCD, 6) == 3
 *     nvMaskPos32(0xCD, 7) == 4
 */
static NV_FORCEINLINE NvU32
nvMaskPos32(const NvU32 mask, const NvU32 bitIdx)
{
    return nvPopCount32(mask & (NVBIT32(bitIdx) - 1));
}

// Destructive operation on n32
#define LOWESTBITIDX_32(n32)  \
{                             \
    n32 = LOWESTBIT(n32);     \
    IDX_32(n32);              \
}

// Destructive operation on n32
#define HIGHESTBITIDX_32(n32)   \
{                               \
    NvU32 count = 0;            \
    while (n32 >>= 1)           \
    {                           \
        count++;                \
    }                           \
    n32 = count;                \
}

// Destructive operation on n32
#define ROUNDUP_POW2(n32) \
{                         \
    n32--;                \
    n32 |= n32 >> 1;      \
    n32 |= n32 >> 2;      \
    n32 |= n32 >> 4;      \
    n32 |= n32 >> 8;      \
    n32 |= n32 >> 16;     \
    n32++;                \
}

/*!
 * Round up a 32-bit unsigned integer to the next power of 2.
 * Pure typesafe alternative to @ref ROUNDUP_POW2.
 *
 * param[in] x must be in range [0, 2^31] to avoid overflow.
 */
static NV_FORCEINLINE NvU32
nvNextPow2_U32(const NvU32 x)
{
    NvU32 y = x;
    y--;
    y |= y >> 1;
    y |= y >> 2;
    y |= y >> 4;
    y |= y >> 8;
    y |= y >> 16;
    y++;
    return y;
}


static NV_FORCEINLINE NvU32
nvPrevPow2_U32(const NvU32 x )
{
    NvU32 y = x;
    y |= (y >> 1);
    y |= (y >> 2);
    y |= (y >> 4);
    y |= (y >> 8);
    y |= (y >> 16);
    return y - (y >> 1);
}


// Destructive operation on n64
#define ROUNDUP_POW2_U64(n64) \
{                         \
    n64--;                \
    n64 |= n64 >> 1;      \
    n64 |= n64 >> 2;      \
    n64 |= n64 >> 4;      \
    n64 |= n64 >> 8;      \
    n64 |= n64 >> 16;     \
    n64 |= n64 >> 32;     \
    n64++;                \
}

#define NV_SWAP_U8(a,b) \
{                       \
    NvU8 temp;          \
    temp = a;           \
    a = b;              \
    b = temp;           \
}

/*!
 * @brief   Macros allowing simple iteration over bits set in a given mask.
 *
 * @param[in]       maskWidth   bit-width of the mask (allowed: 8, 16, 32, 64)
 *
 * @param[in,out]   index       lvalue that is used as a bit index in the loop
 *                              (can be declared as any NvU* or NvS* variable)
 * @param[in]       mask        expression, loop will iterate over set bits only
 */
#define FOR_EACH_INDEX_IN_MASK(maskWidth,index,mask)        \
{                                                           \
    NvU##maskWidth lclMsk = (NvU##maskWidth)(mask);         \
    for (index = 0; lclMsk != 0; index++, lclMsk >>= 1)     \
    {                                                       \
        if (((NvU##maskWidth)NVBIT64(0) & lclMsk) == 0)     \
        {                                                   \
            continue;                                       \
        }
#define FOR_EACH_INDEX_IN_MASK_END                          \
    }                                                       \
}

//
// Size to use when declaring variable-sized arrays
//
#define NV_ANYSIZE_ARRAY                                                      1

//
// Returns ceil(a/b)
//
#define NV_CEIL(a,b) (((a)+(b)-1)/(b))

// Clearer name for NV_CEIL
#ifndef NV_DIV_AND_CEIL
#define NV_DIV_AND_CEIL(a, b) NV_CEIL(a,b)
#endif

#ifndef NV_MIN
#define NV_MIN(a, b)        (((a) < (b)) ? (a) : (b))
#endif

#ifndef NV_MAX
#define NV_MAX(a, b)        (((a) > (b)) ? (a) : (b))
#endif

//
// Returns absolute value of provided integer expression
//
#define NV_ABS(a) ((a)>=0?(a):(-(a)))

//
// Returns 1 if input number is positive, 0 if 0 and -1 if negative. Avoid
// macro parameter as function call which will have side effects.
//
#define NV_SIGN(s) ((NvS8)(((s) > 0) - ((s) < 0)))

//
// Returns 1 if input number is >= 0 or -1 otherwise. This assumes 0 has a
// positive sign.
//
#define NV_ZERO_SIGN(s) ((NvS8)((((s) >= 0) * 2) - 1))

// Returns the offset (in bytes) of 'member' in struct 'type'.
#ifndef NV_OFFSETOF
    #if defined(__GNUC__) && __GNUC__ > 3
        #define NV_OFFSETOF(type, member)   ((NvU32)__builtin_offsetof(type, member))
    #else
        #define NV_OFFSETOF(type, member)    ((NvU32)(NvU64)&(((type *)0)->member)) // shouldn't we use PtrToUlong? But will need to include windows header.
    #endif
#endif

//
// Performs a rounded division of b into a (unsigned). For SIGNED version of
// NV_ROUNDED_DIV() macro check the comments in http://nvbugs/769777.
//
#define NV_UNSIGNED_ROUNDED_DIV(a,b)    (((a) + ((b) / 2)) / (b))

/*!
 * Performs a ceiling division of b into a (unsigned).  A "ceiling" division is
 * a division is one with rounds up result up if a % b != 0.
 *
 * @param[in] a    Numerator
 * @param[in] b    Denominator
 *
 * @return a / b + a % b != 0 ? 1 : 0.
 */
#define NV_UNSIGNED_DIV_CEIL(a, b)      (((a) + (b - 1)) / (b))

/*!
 * Performs a rounded right-shift of 32-bit unsigned value "a" by "shift" bits.
 * Will round result away from zero.
 *
 * @param[in] a      32-bit unsigned value to shift.
 * @param[in] shift  Number of bits by which to shift.
 *
 * @return  Resulting shifted value rounded away from zero.
 */
#define NV_RIGHT_SHIFT_ROUNDED(a, shift)                                       \
    (((a) >> (shift)) + !!((NVBIT((shift) - 1) & (a)) == NVBIT((shift) - 1)))

//
// Power of 2 alignment.
//    (Will give unexpected results if 'gran' is not a power of 2.)
//
#ifndef NV_ALIGN_DOWN
#define NV_ALIGN_DOWN(v, gran)      ((v) & ~((gran) - 1))
#endif

#ifndef NV_ALIGN_UP
#define NV_ALIGN_UP(v, gran)        (((v) + ((gran) - 1)) & ~((gran)-1))
#endif

#ifndef NV_ALIGN_DOWN64
#define NV_ALIGN_DOWN64(v, gran)      ((v) & ~(((NvU64)gran) - 1))
#endif

#ifndef NV_ALIGN_UP64
#define NV_ALIGN_UP64(v, gran)        (((v) + ((gran) - 1)) & ~(((NvU64)gran)-1))
#endif

#ifndef NV_IS_ALIGNED
#define NV_IS_ALIGNED(v, gran)      (0 == ((v) & ((gran) - 1)))
#endif

#ifndef NV_IS_ALIGNED64
#define NV_IS_ALIGNED64(v, gran)      (0 == ((v) & (((NvU64)gran) - 1)))
#endif

#ifndef NVMISC_MEMSET
static NV_FORCEINLINE void *NVMISC_MEMSET(void *s, NvU8 c, NvLength n)
{
    NvU8 *b = (NvU8 *) s;
    NvLength i;

    for (i = 0; i < n; i++)
    {
        b[i] = c;
    }

    return s;
}
#endif

#ifndef NVMISC_MEMCPY
static NV_FORCEINLINE void *NVMISC_MEMCPY(void *dest, const void *src, NvLength n)
{
    NvU8 *destByte = (NvU8 *) dest;
    const NvU8 *srcByte = (const NvU8 *) src;
    NvLength i;

    for (i = 0; i < n; i++)
    {
        destByte[i] = srcByte[i];
    }

    return dest;
}
#endif

static NV_FORCEINLINE char *NVMISC_STRNCPY(char *dest, const char *src, NvLength n)
{
    NvLength i;

    for (i = 0; i < n; i++)
    {
        dest[i] = src[i];
        if (src[i] == '\0')
        {
            break;
        }
    }

    for (; i < n; i++)
    {
        dest[i] = '\0';
    }

    return dest;
}

#ifdef __cplusplus
}
#endif //__cplusplus

#endif // __NV_MISC_H

