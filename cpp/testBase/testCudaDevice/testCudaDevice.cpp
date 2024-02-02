#include "hippoCommon.hpp"
#include "hippoCuda.hpp"

int main(int argc, char const *argv[]) {
    hippoProcessInst.initialize();
    hippoCudaDeviceInst.setDeviceId(0);
    int driverVersion = hippoCudaDeviceInst.driverVersion();
    int runtimeVersion = hippoCudaDeviceInst.runtimeVersion();
    LOG(INFO) << "driver version: " << driverVersion / 1000 << "." << (driverVersion % 100) / 10
              << ", runtime version: " << runtimeVersion / 1000 << "." << (runtimeVersion % 100) / 10;
    LOG(INFO) << "deviceCount = " << hippoCudaDeviceInst.deviceCount() << ", deviceId = " << hippoCudaDeviceInst.deviceId()
              << ", deviceHandle = " << hippoCudaDeviceInst.deviceHandle()
              << ", deviceUuid = " << hippoCudaDeviceInst.deviceUuid().bytes
              << ", deviceName = " << hippoCudaDeviceInst.deviceName();
    int supportRet = hippoCudaDeviceInst.queryDeviceAttributes(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
    LOG(INFO) << "CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK support ret = " << supportRet;
    supportRet = hippoCudaDeviceInst.queryDeviceAttributes(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X);
    LOG(INFO) << "CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X support ret = " << supportRet;
    supportRet = hippoCudaDeviceInst.queryDeviceAttributes(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y);
    LOG(INFO) << "CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y support ret = " << supportRet;
    supportRet = hippoCudaDeviceInst.queryDeviceAttributes(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z);
    LOG(INFO) << "CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z support ret = " << supportRet;
    supportRet = hippoCudaDeviceInst.queryDeviceAttributes(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X);
    LOG(INFO) << "CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X support ret = " << supportRet;
    supportRet = hippoCudaDeviceInst.queryDeviceAttributes(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y);
    LOG(INFO) << "CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y support ret = " << supportRet;
    supportRet = hippoCudaDeviceInst.queryDeviceAttributes(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z);
    LOG(INFO) << "CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z support ret = " << supportRet;
    // bytes
    supportRet = hippoCudaDeviceInst.queryDeviceAttributes(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK);
    LOG(INFO) << "CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK support ret = " << supportRet;
    // bytes
    supportRet = hippoCudaDeviceInst.queryDeviceAttributes(CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY);
    LOG(INFO) << "CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY support ret = " << supportRet;
    supportRet = hippoCudaDeviceInst.queryDeviceAttributes(CU_DEVICE_ATTRIBUTE_WARP_SIZE);
    LOG(INFO) << "CU_DEVICE_ATTRIBUTE_WARP_SIZE support ret = " << supportRet;
    supportRet = hippoCudaDeviceInst.queryDeviceAttributes(CU_DEVICE_ATTRIBUTE_MAX_PITCH);
    LOG(INFO) << "CU_DEVICE_ATTRIBUTE_MAX_PITCH support ret = " << supportRet;
    supportRet = hippoCudaDeviceInst.queryDeviceAttributes(CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK);
    LOG(INFO) << "CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK support ret = " << supportRet;

    // cuda.h CUcomputemode_enum
    supportRet = hippoCudaDeviceInst.queryDeviceAttributes(CU_DEVICE_ATTRIBUTE_COMPUTE_MODE);
    LOG(INFO) << "CU_DEVICE_ATTRIBUTE_COMPUTE_MODE support ret = " << supportRet;
    if (CU_COMPUTEMODE_DEFAULT != supportRet) {
        LOG(WARNING) << "current device does not support multi cuda context";
    }

    supportRet = hippoCudaDeviceInst.queryDeviceAttributes(CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY);
    LOG(INFO) << "CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY support ret = " << supportRet;
    supportRet = hippoCudaDeviceInst.queryDeviceAttributes(CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM);
    LOG(INFO) << "CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM support ret = " << supportRet;

    supportRet = hippoCudaDeviceInst.queryDeviceAttributes(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR);
    LOG(INFO) << "CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR support ret = " << supportRet;
    supportRet = hippoCudaDeviceInst.queryDeviceAttributes(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR);
    LOG(INFO) << "CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR support ret = " << supportRet;
    supportRet = hippoCudaDeviceInst.queryDeviceAttributes(CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED);
    LOG(INFO) << "CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED support ret = " << supportRet;

    hippoProcessInst.finalize();
    return 0;
}
