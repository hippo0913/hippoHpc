#ifndef __HIPPOCUDAGRAGH__H__
#define __HIPPOCUDAGRAGH__H__

#include "hippoCudaPublic.hpp"

NAMESPACE_HIPPO_BEGIN
NAMESPACE_HPC_BEGIN

class CudaGraghWrapper {
public:
    explicit CudaGraghWrapper() : exec_(0), is_capturing_(false) {
        HPC_CUDA_CHECK(cudaGraphCreate(&graph_, 0));
#if HIPPO_DEBUG == 1
        printf("%s:%d, pid = %lu, graph = %p, created.\n", __PRETTY_FUNCTION__, __LINE__, pthread_self(), graph_);
#endif  // !HIPPO_DEBUG == 1
    }
    CudaGraghWrapper(const CudaGraghWrapper&) = delete;
    CudaGraghWrapper& operator=(const CudaGraghWrapper&) = delete;
    CudaGraghWrapper(CudaGraghWrapper&&) = delete;
    CudaGraghWrapper& operator=(CudaGraghWrapper&&) = delete;
    ~CudaGraghWrapper() {
        if (!graph_) {
#if HIPPO_DEBUG == 1
            printf("%s:%d, pid = %lu, graph = %p, will be destroied.\n", __PRETTY_FUNCTION__, __LINE__, pthread_self(),
                   graph_);
#endif  // !HIPPO_DEBUG == 1
            cudaGraphDestroy(graph_);
        }
    }

    void BeginCapture(cudaStream_t stream) {
        if (!is_capturing_ && graph_ != nullptr) {
            is_capturing_ = true;
            HPC_CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        }
    }

    void EndCapture(cudaStream_t stream) {
        if (!is_capturing_) {
            is_capturing_ = false;
            HPC_CUDA_CHECK(cudaStreamEndCapture(stream, &graph_));
        }
    }

    bool IsCapturing() const { return is_capturing_; }

private:
    mutable cudaGraph_t graph_;
    mutable std::shared_ptr<cudaGraphExec_t> exec_;
    bool is_capturing_;
};

using CudaGragh = CudaGraghWrapper;

NAMESPACE_HPC_END
NAMESPACE_HIPPO_END

#endif  //!__HIPPOCUDAGRAGH__H__