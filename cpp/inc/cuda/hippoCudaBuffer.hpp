#ifndef __HIPPOCUDABUFFER__H__
#define __HIPPOCUDABUFFER__H__

#include "hippoCudaPublic.hpp"

NAMESPACE_HIPPO_BEGIN
NAMESPACE_HPC_BEGIN

struct DeviceAllocator {
    void operator()(void **ptr, size_t size) { HPC_CUDA_CHECK(cudaMalloc(ptr, size)); }
};

struct DeviceDeallocator {
    void operator()(void *ptr) { HPC_CUDA_CHECK(cudaFree(ptr)); }
};

struct ManagedAllocator {
    void operator()(void **ptr, size_t size) { HPC_CUDA_CHECK(cudaMallocManaged(ptr, size)); }
};

struct HostAllocator {
    void operator()(void **ptr, size_t size) { HPC_CUDA_CHECK(cudaMallocHost(ptr, size)); }
};

struct HostDeallocator {
    void operator()(void **ptr) { HPC_CUDA_CHECK(cudaFreeHost(ptr)); }
};

// allocate when create constructor
template <typename Allocator, typename Deallocator>
struct CudaBufferEager {
    explicit CudaBufferEager(size_t size)
        : mSize(size)
        , mPtr[&mSize](
              {
                  void *ptr = nullptr;
                  Allocator{}(mPtr, mSize);
                  return ptr;
              }(),
              Deallocator{}) {}
    CudaBufferEager(CudaBufferEager const &) = delete;
    CudaBufferEager &operator=(CudaBufferEager const &) = delete;
    ~CudaBufferEager() = default;

    void *get() const { return mPtr.get(); }

private:
    std::shared_ptr<void> mPtr;
    size_t mSize = 0;
};

// allocate when call get()
template <typename Allocator, typename Deallocator>
struct CudaBufferLazy {
    explicit CudaBufferLazy(size_t size) : mSize(size), mPtr{nullptr} {}
    CudaBufferLazy(CudaBufferLazy const &) = delete;
    CudaBufferLazy &operator=(CudaBufferLazy const &) = delete;
    ~CudaBufferLazy() = default;

    void *get() const {
        if (!mPtr) {
            void *ptr = nullptr;
            Allocator()(&mPtr, mSize);
            mPtr.set(ptr, Deallocator());
        }
        return mPtr.get();
    }

private:
    mutable std::shared_ptr<void> mPtr;
    size_t mSize = 0;
};

#if HIPPO_MULTI_CUDACONTEXT == 1
template <typename Allocator, typename Deallocator>
using CudaBuffer = CudaBufferLazy<Allocator, Deallocator>;
#else
template <typename Allocator, typename Deallocator>
using CudaBuffer = CudaBufferEager<Allocator, Deallocator>;
#endif

constexpr bool has_single_bit(size_t x) noexcept { return x && !(x & (x - 1)); }

template <typename Scalar, size_t Align>
constexpr size_t align(size_t n) noexcept {
    static_assert(has_single_bit(Align), "Align must be an integral power of two.");
    return ((n + Align - 1) & (~(Align - 1)));
}

template <typename Scalar, size_t N, size_t C, size_t H, size_t W, size_t Align, size_t WStride, size_t HStride>
struct TensorAttr {
    static constexpr size_t elem_bytes = sizeof(Scalar);
    static constexpr size_t n = N;
    static constexpr size_t c = C;
    static constexpr size_t h = H;
    static constexpr size_t w = W;
    static constexpr size_t w_bytes = elem_bytes * w;
    static constexpr size_t align = Align;
    static constexpr size_t w_stride = std::max(align<Scalar, Align>(sizeof(Scalar) * W), WStride);
    static constexpr size_t h_stride = std::max(h, HStride);
    static constexpr size_t alloc_bytes = n * c * w_stride * h_stride;
};

template <typename Scalar, size_t N, size_t C, size_t H, size_t W, size_t Align = 1, size_t WStride = 0, size_t HStride = 0>
struct CudaBufferDevice {
    static constexpr TensorAttr<Scalar, N, C, H, W, Align, WStride, HStride> attr{};
    CudaBufferDevice() : mBuffer(attr.alloc_bytes) {}
    CudaBufferDevice(CudaBufferDevice cosnst &) = delete;
    CudaBufferDevice &operator=(const CudaBufferDevice &) = delete;
    ~CudaBufferDevice() = default;

    Scalar *getGpuPtr() const { return static_cast<Scalar *>(mBuffer.get()); }

private:
    CudaBuffer<DeviceAllocator, DeviceDeallocator> mBuffer;
};

template <typename Scalar, size_t N, size_t C, size_t H, size_t W, size_t Align = 1, size_t WStride = 0, size_t HStride = 0>
struct CudaBufferHost {
    static constexpr TensorAttr<Scalar, N, C, H, W, Align, WStride, HStride> attr{};
    CudaBufferHost() : mBuffer(attr.alloc_bytes) {}
    CudaBufferHost(CudaBufferHost const &) = delete;
    CudaBufferHost &operator=(const CudaBufferHost &) = delete;
    ~CudaBufferHost() = default;

    Scalar *getCpuPtr() const { return static_cast<Scalar *>(mBuffer.get()); }

private:
    CudaBuffer<HostAllocator, HostDeallocator> mBuffer;
};

template <typename Scalar, size_t N, size_t C, size_t H, size_t W, size_t Align = 1, size_t WStride = 0, size_t HStride = 0>
struct CudaBufferManaged {
    static constexpr TensorAttr<Scalar, N, C, H, W, Align, WStride, HStride> attr{};
    CudaBufferManaged() : mBuffer(attr.alloc_bytes) {}
    CudaBufferManaged(CudaBufferManaged const &) = delete;
    CudaBufferManaged &operator=(CudaBufferManaged const &) = delete;
    ~CudaBufferManaged() = default;

    Scalar *getGpuPtr() const { return static_cast<Scalar *>(mBuffer.get()); }
    Scalar *getCpuPtr() const { return static_cast<Scalar *>(mBuffer.get()); }

    void hostToDevice(cudaStream_t stream) const {
        HPC_CUDA_CHECK(cudaStreamAttachMemAsync(stream, mBuffer.get(), 0, cudaMemAttachGlobal));
    }

    void deviceToHost(cudaStream_t stream) const {
        HPC_CUDA_CHECK(cudaStreamAttachMemAsync(stream, mBuffer.get(), 0, cudaMemAttachHost));
    }

private:
    CudaBuffer<ManagedAllocator, DeviceDeallocator> mBuffer;
};

template <typename Scalar, size_t N, size_t C, size_t H, size_t W, size_t Align = 1, size_t WStride = 0, size_t HStride = 0>
struct CudaBufferMirrored {
    static constexpr TensorAttr<Scalar, N, C, H, W, Align, WStride, HStride> attr{};
    CudaBufferMirrored() : mDevBuffer(attr.alloc_bytes), mHostBuffer(attr.alloc_bytes) {}
    CudaBufferMirrored(CudaBufferMirrored const &) = delete;
    CudaBufferMirrored &operator=(CudaBufferMirrored const &) = delete;
    ~CudaBufferMirrored() = default;

    Scalar *getGpuPtr() const { return static_cast<Scalar *>(mDevBuffer.get()); }
    Scalar *getCpuPtr() const { return static_cast<Scalar *>(mHostBuffer.get()); }

    void hostToDevice(cudaStream_t stream) const {
        HPC_CUDA_CHECK(cudaMemcpyAsync(mDevBuffer.get(), mHostBuffer.get(), attr.alloc_bytes, cudaMemcpyHostToDevice, stream));
    }

    void deviceToHost(cudaStream_t stream) const {
        HPC_CUDA_CHECK(cudaMemcpyAsync(mHostBuffer.get(), mDevBuffer.get(), attr.alloc_bytes, cudaMemcpyDeviceToHost, stream));
    }

private:
    CudaBuffer<DeviceAllocator, DeviceDeallocator> mDevBuffer;
    CudaBuffer<HostAllocator, HostDeallocator> mHostBuffer;
};

template <typename Buffer, typename Meta>
struct CudaBufferWithMeta {
    Buffer mBuffer;
    Meta mMeta;
};

NAMESPACE_HPC_END
NAMESPACE_HIPPO_END

#endif  //!__HIPPOCUDABUFFER__H__