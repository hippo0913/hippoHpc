#ifndef __HIPPOCUDAPUBLIC__H__
#define __HIPPOCUDAPUBLIC__H__

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "hippoCommon.hpp"

#define HPC_CUDRV_CHECK(err)                                                                                          \
    do {                                                                                                              \
        if (err != CUDA_SUCCESS) {                                                                                    \
            char const* pStr = nullptr;                                                                               \
            cuGetErrorString(err, &pStr);                                                                             \
            LOG(ERROR) << "CudaDrv failure at " << __PRETTY_FUNCTION__ << ", err = " << err << ", errstr = " << pStr; \
            sleep(1U);                                                                                                \
            abort();                                                                                                  \
        }                                                                                                             \
    } while (false)

#define HPC_CUDA_CHECK(err)                                                                \
    do {                                                                                   \
        if (err != cudaSuccess) {                                                          \
            LOG(ERROR) << "CudaRT failure at " << __PRETTY_FUNCTION__ << ", err = " << err \
                       << ", errdesc = " << cudaGetErrorString(err);                       \
            sleep(1U);                                                                     \
            abort();                                                                       \
        }                                                                                  \
    } while (false)

#define HPC_CUDLA_CHECK(err)                                                                       \
    do {                                                                                           \
        if (err != cudlaSuccess) {                                                                 \
            hpc::common::HPC_LOG_ERR("Cudla failure at %s:%d, err = %d", __FILE__, __LINE__, err); \
            sleep(1U);                                                                             \
            abort();                                                                               \
        }                                                                                          \
    } while (false)

#endif  //!__HIPPOCUDAPUBLIC__H__