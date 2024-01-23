#ifndef __HIPPOCUDA__H__
#define __HIPPOCUDA__H__

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "hippoCommon.hpp"

#define HPC_CUDRV_CHECK(err)                                                                     \
    do {                                                                                         \
        if (err != CUDA_SUCCESS) {                                                               \
            char const* pStr = nullptr;                                                          \
            cuGetErrorString(err, &pStr);                                                        \
            LOG(ERROR) << "CudaDrv failure at " << __PRETTY_FUNCTION__ << ", err = " << err << ", errstr = " << pStr; \
            sleep(1U);                                                                           \
            abort();                                                                             \
        }                                                                                        \
    } while (false)

#define HPC_CUDA_CHECK(err)                                                                        \
    do {                                                                                           \
        if (err != cudaSuccess) {                                                                  \
            hpc::common::HPC_LOG_ERR("CudaRT failure at %s:%d : err = %d, errdesc = %s", __FILE__, \
                                     __LINE__, err, cudaGetErrorString(err));                      \
            sleep(1U);                                                                             \
            abort();                                                                               \
        }                                                                                          \
    } while (false)

#define HPC_CUDLA_CHECK(err)                                                                       \
    do {                                                                                           \
        if (err != cudlaSuccess) {                                                                 \
            hpc::common::HPC_LOG_ERR("Cudla failure at %s:%d, err = %d", __FILE__, __LINE__, err); \
            sleep(1U);                                                                             \
            abort();                                                                               \
        }                                                                                          \
    } while (false)

NAMESPACE_HIPPO_BEGIN
NAMESPACE_DEVICE_BEGIN

// CudaDevice from CUDA Driver API
struct hippoCudaDevice {
    explicit hippoCudaDevice(int deviceId = 0)
        : mDeviceCount(0), mDeviceId(deviceId) {
        HPC_CUDRV_CHECK(cuInit(0));
        HPC_CUDRV_CHECK(cuDeviceGetCount(&mDeviceCount));
        assert(mDeviceId < mDeviceCount && mDeviceId >= 0);
    }

    int deviceCount() const { return mDeviceCount; }

    void setDeviceFlags()

private:
    int mDeviceCount = -1;
    int mDeviceId = -1;
};

NAMESPACE_DEVICE_END
NAMESPACE_HIPPO_END

#endif  //!__HIPPOCUDA__H__
