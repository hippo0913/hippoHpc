#ifndef __HIPPOCUDASTREAM__H__
#define __HIPPOCUDASTREAM__H__

#include "hippoCudaPublic.hpp"

NAMESPACE_HIPPO_BEGIN
NAMESPACE_HPC_BEGIN

class CudaStreamWithFlag {
public:
    // stream priority [0, -5], 0 least, -5 greatest
    explicit CudaStreamWithFlag(int flags = cudaStreamNonBlocking, int priority = 0)
        : flags_(flags), priority_(priority) {}
    CudaStreamWithFlag(const CudaStreamWithFlag&) = delete;
    CudaStreamWithFlag& operator=(const CudaStreamWithFlag&) = delete;
    CudaStreamWithFlag(CudaStreamWithFlag&&) = delete;
    CudaStreamWithFlag& operator=(CudaStreamWithFlag&&) = delete;
    ~CudaStreamWithFlag() = default;

    void QueryStreamPriorityRange(int* leastPriority, int* greatestPriority) {
        HPC_CUDA_CHECK(cudaDeviceGetStreamPriorityRange(leastPriority, greatestPriority));
    }

    int QueryCurStreamPriority() const {
        int priority = 0;
        HPC_CUDA_CHECK(cudaStreamGetPriority(stream_.get(), &priority));
        if (priority != priority_) {
#if HIPPO_DEBUG == 1
            printf("%s:%d, stream = %p, priority = %d, priority_ = %d\n", __PRETTY_FUNCTION__, __LINE__, stream_.get(),
                   priority, priority_);
#endif  // !HIPPO_DEBUG == 1
        }
        return priority;
    }

    int QueryStreamPriority(cudaStream_t stream) const {
        int priority = 0;
        HPC_CUDA_CHECK(cudaStreamGetPriority(stream_.get(), &priority));
#if HIPPO_DEBUG == 1
        printf("%s:%d, stream = %p, priority = %d\n", __PRETTY_FUNCTION__, __LINE__, stream, priority);
#endif  // !HIPPO_DEBUG == 1
        return priority;
    }

    int QueryStreamIsCapturing() const {
        cudaStreamCaptureStatus capturing;
        HPC_CUDA_CHECK(cudaStreamIsCapturing(stream_.get(), &capturing));
        return static_cast<int>(capturing);
    }

    void Synchronize() { HPC_CUDA_CHECK(cudaStreamSynchronize(stream_.get())); }

    cudaStream_t get() const {
        AutoAlloc();
        return stream_.get();
    }

private:
    void AutoAlloc() const {
        if (!stream_) {
            cudaStream_t stream = nullptr;
            HPC_CUDA_CHECK(cudaStreamCreateWithPriority(&stream, flags_, priority_));
#if HIPPO_DEBUG == 1
            printf("%s:%d, pid = %lu, stream = %p, created.\n", __PRETTY_FUNCTION__, __LINE__, pthread_self(), stream);
#endif  // !HIPPO_DEBUG == 1
            stream_.reset(stream, [](cudaStream_t stream) {
                if (stream) {
#if HIPPO_DEBUG == 1
                    printf("%s:%d, pid = %lu, stream = %p, will be destroied.\n", __PRETTY_FUNCTION__, __LINE__,
                           pthread_self(), stream);
#endif  // !HIPPO_DEBUG == 1
                    HPC_CUDA_CHECK(cudaStreamDestroy(stream));
                }
            });
        }
    }

    uint32_t flags_ = 0U;
    int priority_ = 0;
    mutable std::shared_ptr<CUstream_st> stream_;
};

using CudaStream = CudaStreamWithFlag;

NAMESPACE_HPC_END
NAMESPACE_HIPPO_END

#endif  //!__HIPPOCUDASTREAM__H__