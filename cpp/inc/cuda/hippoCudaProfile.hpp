#ifndef __HIPPOCUDAPROFILE__H__
#define __HIPPOCUDAPROFILE__H__

#include "hippoPublic.hpp"

#if HIPPO_PROFILING == 1

#include <nvtx3/nvToolsExt.h>

NAMESPACE_HIPPO_BEGIN
NAMESPACE_HPC_BEGIN

struct ProfilingRangeRAII {
    ProfilingRangeRAII(const char* message) { nvtxRangePushA(message); }

    ~ProfilingRangeRAII() { nvtxRangePop(); }
};

NAMESPACE_HPC_END
NAMESPACE_HIPPO_END

#define HPC_PROFILING_AUTORANGE(msg) auto ____l = hpc::device::tegra::ProfilingRangeRAII(msg)
#define HPC_PROFILING_MARK(msg) nvtxMarkA(msg)
#else
#define HPC_PROFILING_AUTORANGE(msg)
#define HPC_PROFILING_MARK(msg)
#endif  // HIPPO_PROFILING == 1

#endif  //!__HIPPOCUDAPROFILE__H__