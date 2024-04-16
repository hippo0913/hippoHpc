#ifndef __HIPPOCUDADEVICE__H__
#define __HIPPOCUDADEVICE__H__

#include "hippoCudaPublic.hpp"

NAMESPACE_HIPPO_BEGIN
NAMESPACE_DEVICE_BEGIN

// CudaDevice from CUDA Driver API
struct hippoCudaDevice {
    explicit hippoCudaDevice() {
        HPC_CUDRV_CHECK(cuInit(0));
        HPC_CUDRV_CHECK(cuDeviceGetCount(&mDeviceCount));
    }
    ~hippoCudaDevice() { HPC_CUDA_CHECK(cudaDeviceReset()); }

    // set
    void setDevice() { HPC_CUDA_CHECK(cudaSetDevice(mDeviceId)); }
    void setDeviceFlags(int flags = cudaDeviceScheduleAuto) { HPC_CUDA_CHECK(cudaSetDeviceFlags(flags)); }
    void setDeviceId(int deviceId = 0) {
        mDeviceId = deviceId;
        assert(mDeviceId < mDeviceCount && mDeviceId >= 0);
        HPC_CUDRV_CHECK(cuDeviceGet(&mDeviceHandle, mDeviceId));
        HPC_CUDRV_CHECK(cuDeviceGetUuid(&mDeviceUuid, mDeviceId));
        HPC_CUDRV_CHECK(cuDeviceGetName(mDeviceName, sizeof(mDeviceName), mDeviceHandle));
        cudaDriverGetVersion(&mDriverVersion);
        cudaRuntimeGetVersion(&mRuntimeVersion);
    }

    // query
    int queryDeviceAttributes(CUdevice_attribute attribute) {
        HPC_CUDRV_CHECK(cuDeviceGetAttribute(&mDeviceAttributes, attribute, mDeviceHandle));
        return mDeviceAttributes;
    }

    void queryDeviceProperties(cudaDeviceProp& deviceProp) {
        HPC_CUDA_CHECK(cudaGetDeviceProperties(&mDeviceProp, mDeviceId));
    }

    int QueryStreamPriority(int* least, int* greatest) {
        HPC_CUDA_CHECK(cudaDeviceGetStreamPriorityRange(least, greatest));
        return 0;
    }

    // get
    int deviceCount() const { return mDeviceCount; }
    int deviceId() const { return mDeviceId; }
    CUdevice deviceHandle() const { return mDeviceHandle; }
    CUuuid deviceUuid() const { return mDeviceUuid; }
    const char* deviceName() { return mDeviceName; }
    int driverVersion() const { return mDriverVersion; }
    int runtimeVersion() const { return mRuntimeVersion; }

private:
    int mDeviceCount = -1;
    int mDeviceId = -1;
    CUdevice mDeviceHandle = -1;
    CUuuid mDeviceUuid;
    char mDeviceName[256] = {0};
    int mDeviceAttributes = -1;
    cudaDeviceProp mDeviceProp;
    int mDriverVersion = 0;
    int mRuntimeVersion = 0;
};

NAMESPACE_DEVICE_END
NAMESPACE_HIPPO_END

#define hippoCudaDeviceInst (hippo::common::GlobalSingleton<hippo::device::hippoCudaDevice>::instance())

#endif  //!__HIPPOCUDADEVICE__H__