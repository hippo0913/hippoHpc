#include "hippoCommon.hpp"
#include "hippoCuda.hpp"


int main(int argc, char const *argv[]) {
    hippoProcessInst.initialize();
    hippo::device::hippoCudaDevice testDevice(0);
    LOG(INFO) << testDevice.deviceCount();
    hippoProcessInst.finalize();
    return 0;
}
