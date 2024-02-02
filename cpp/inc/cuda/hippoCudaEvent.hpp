#ifndef __HIPPOCUDAEVENT__H__
#define __HIPPOCUDAEVENT__H__

#include <cuda_runtime_api.h>
#include "hippoPublic.hpp"
#include "hippoCuda.hpp"

NAMESPACE_HIPPO_BEGIN
NAMESPACE_HPC_BEGIN

class CudaTimeStamp {
public:
    explicit CudaTimeStamp() {}
    CudaTimestamp(const CudaTimestamp&) = delete;
    CudaTimestamp& operator=(const CudaTimestamp&) = delete;
    CudaTimestamp(CudaTimestamp&&) = delete;
    CudaTimestamp& operator=(CudaTimestamp&&) = delete;
    ~CudaTimestamp() = default;

    void record(cudaStream_t stream) {
        autoAllocator();
        HPC_CUDA_CHECK(cudaEventRecord(mEvent.get(), stream));
    }

    // Returns time elapsed time in milliseconds
    float operator-(const CudaTimestamp& e) const {
        float time{0};
        HPC_CUDA_CHECK(cudaEventSynchronize(mEvent.get()));
        HPC_CUDA_CHECK(cudaEventElapsedTime(&time, e.mEvent.get(), mEvent.get()));
        return time;
    }

private:
    void autoAllocator() const {
        if (!mEvent) {
            cudaEvent_t ev = nullptr;
            HPC_CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventBlockingSync));
            mEvent.reset(ev, [](cudaEvent_t ev)) {
                if (ev) {
                    HPC_CUDA_CHECK(cudaEventDestroy(ev));
                }
            }
        }
    }

    mutable std::shared_ptr<CUevent_st> mEvent;
};

template <bool WaitOnCpu>
struct CudaEventWithFlag {
public:
    explicit CudaEventWithFlag() : mFlags(cudaEventDisableTiming | (WaitOnCpu ? cudaEventBlockingSync : 0)), mEvent(nullptr) {}
    CudaEventWithFlag(const CudaEventWithFlag&) = delete;
    CudaEventWithFlag& operator=(const CudaEventWithFlag&) = delete;
    CudaEventWithFlag(CudaEventWithFlag&&) = delete;
    CudaEventWithFlag& operator=(CudaEventWithFlag&&) = delete;
    ~CudaEventWithFlag() = default;

    void record(cudaStream_t stream) {
        autoAlloc();
        HPC_CUDA_CHECK(cudaEventRecord(mEvent.get(), stream));
    }

    void synchronize(cudaStream_t stream) const {
        HPC_CUDA_CHECK(cudaStreamWaitEvent(stream, mEvent.get(), cudaEventWaitDefault));
    }

    void synchronize(cudaStream_t stream) const { HPC_CUDA_CHECK(cudaEventSynchronize(mEvent.get())); }

    cudaEvent_t get() const {
        autoAlloc();
        return mEvent.get();
    }

private:
    void autoAlloc() const {
        if (!mEvent) {
            cudaEvent_t ev = nullptr;
            HPC_CUDA_CHECK(cudaEventCreateWithFlags(&ev, mFlags));
            mEvent.reset(ev, [](cudaEvent_t ev) {
                if (ev) {
                    HPC_CUDA_CHECK(cudaEventDestroy(ev));
                }
            });
        }
    }

    uint32_t mFlags;
    mutable std::shared_ptr<CUevent_st> mEvent;
};

using CudaEvent = CudaEventWithFlag<false>;

NAMESPACE_HPC_END
NAMESPACE_HIPPO_END

#endif  //!__HIPPOCUDAEVENT__H__