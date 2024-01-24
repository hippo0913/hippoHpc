#include "hippoCommon.hpp"
#include "hippoCuda.hpp"

int main(int argc, char **argv) {
    hippoProcessInst.initialize();
    hippoCudaDeviceInst.setDeviceId(0);
    hippoProcessInst.finalize();
    return 0;
}
