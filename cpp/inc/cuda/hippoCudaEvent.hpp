#ifndef __HIPPOCUDAEVENT__H__
#define __HIPPOCUDAEVENT__H__

#include "hippoCudaPublic.hpp"

NAMESPACE_HIPPO_BEGIN
NAMESPACE_HPC_BEGIN

class CudaTimeStamp {
public:
    explicit CudaTimeStamp() {}
    CudaTimeStamp(const CudaTimeStamp&) = delete;
    CudaTimeStamp& operator=(const CudaTimeStamp&) = delete;
    CudaTimeStamp(CudaTimeStamp&&) = delete;
    CudaTimeStamp& operator=(CudaTimeStamp&&) = delete;
    ~CudaTimeStamp() = default;

    void record(cudaStream_t stream) {
        autoAllocator();
        HPC_CUDA_CHECK(cudaEventRecord(mEvent.get(), stream));
    }

    // Returns time elapsed time in milliseconds
    float operator-(const CudaTimeStamp& e) const {
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
#if HIPPO_DEBUG == 1
            printf("%s:%d, pid = %lu, event = %p, created.\n", __PRETTY_FUNCTION__, __LINE__, pthread_self(), ev);
#endif  // !HIPPO_DEBUG == 1
            mEvent.reset(ev, [](cudaEvent_t ev) {
                if (ev) {
#if HIPPO_DEBUG == 1
                    printf("%s:%d, pid = %lu, event = %p, will be destroied.\n", __PRETTY_FUNCTION__, __LINE__,
                           pthread_self(), ev);
#endif  // !HIPPO_DEBUG == 1
                    HPC_CUDA_CHECK(cudaEventDestroy(ev));
                }
            });
        }
    }

    mutable std::shared_ptr<CUevent_st> mEvent;
};

template <bool WaitOnCpu>
struct CudaEventWithFlag {
public:
    explicit CudaEventWithFlag()
        : mFlags(cudaEventDisableTiming | (WaitOnCpu ? cudaEventBlockingSync : 0)), mEvent(nullptr) {}
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

    void synchronize() const { HPC_CUDA_CHECK(cudaEventSynchronize(mEvent.get())); }

    cudaEvent_t get() const {
        autoAlloc();
        return mEvent.get();
    }

private:
    void autoAlloc() const {
        if (!mEvent) {
            cudaEvent_t ev = nullptr;
            HPC_CUDA_CHECK(cudaEventCreateWithFlags(&ev, mFlags));
#if HIPPO_DEBUG == 1
            printf("%s:%d, pid = %lu, event = %p, created.\n", __PRETTY_FUNCTION__, __LINE__, pthread_self(), ev);
#endif  // !HIPPO_DEBUG == 1
            mEvent.reset(ev, [](cudaEvent_t ev) {
                if (ev) {
#if HIPPO_DEBUG == 1
                    printf("%s:%d, pid = %lu, event = %p, will be destroied.\n", __PRETTY_FUNCTION__, __LINE__,
                           pthread_self(), ev);
#endif  // !HIPPO_DEBUG == 1
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