#include "hippoProcess.hpp"

int main(int argc, char** argv) {
    hippoProcessInst.initialize();
    LOG(INFO) << __PRETTY_FUNCTION__ << ": " << __LINE__;
    LOG(WARNING) << __PRETTY_FUNCTION__ << ": " << __LINE__;
    LOG(ERROR) << __PRETTY_FUNCTION__ << ": " << __LINE__;
    hippoProcessInst.finalize();
    return 0;
}
