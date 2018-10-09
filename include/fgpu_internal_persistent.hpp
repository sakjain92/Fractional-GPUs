/* This file is a header used by internal API */
#ifndef __FGPU_INTERNAL_PERSISTENT_HPP__
#define __FGPU_INTERNAL_PERSISTENT_HPP__

#include <fgpu_internal_common.hpp>

typedef struct __align__(FGPU_DEVICE_CACHELINE_SIZE) fgpu_bindex {
    int index[2];
} fgpu_bindex_t;

/* Memory where persistent kernels use atomic operations to get block index */
typedef struct fgpu_bindexes {
    struct fgpu_bindex bindexes[FGPU_MAX_NUM_COLORS];
} fgpu_bindexes_t;

#define FGPU_NOT_PBLOCK_NOT_STARTED         0
#define FGPU_ACTIVE_PBLOCK_STARTED          1
#define FGPU_INACTIVE_PBLOCK_STARTED        2
#define FGPU_GENERIC_PBLOCK_STARTED         3   /* Either active or inactive */

struct __align__(FGPU_DEVICE_CACHELINE_SIZE) fgpu_indicator {
    int started;
};

/* Memory where persistent kernel indicates to host that it successfully launched */
typedef struct fgpu_indicators {
    struct fgpu_indicator indicators[FGPU_MAX_NUM_PBLOCKS];
} fgpu_indicators_t;

/* Forward declaration */
typedef struct fgpu_dev_ctx fgpu_dev_ctx_t;

void fgpu_set_ctx_dims(fgpu_dev_ctx_t *ctx, int _gridDim, int _blockDim);
void fgpu_set_ctx_dims(fgpu_dev_ctx_t *ctx, dim3 _gridDim, dim3 _blockDim);
void fgpu_set_ctx_dims(fgpu_dev_ctx_t *ctx, uint3 _gridDim, uint3 _blockDim);

#endif /* FGPU_INTERNAL_PERSISTENT_HPP */
