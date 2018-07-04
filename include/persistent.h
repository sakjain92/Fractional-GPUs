/* This file is a header used by internal API */
#ifndef __PERSISTENT_H__
#define __PERSISTENT_H__

#include <config.h>

struct __align__(FGPU_DEVICE_CACHELINE_SIZE) fgpu_bindex {
    int index[2];
};

/* Memory where persistent kernels use atomic operations to get block index */
struct fgpu_bindexes {
    struct fgpu_bindex bindexes[FGPU_MAX_NUM_COLORS];
};

struct __align__(FGPU_DEVICE_CACHELINE_SIZE) fgpu_indicator {
    bool started[FGPU_MAX_NUM_COLORS];
};

struct fgpu_indicators {
    struct fgpu_indicator indicators[FGPU_MAX_NUM_BLOCKS_PER_SM * FGPU_MAX_NUM_SM];
};

#endif /* PERSISTENT_H */
