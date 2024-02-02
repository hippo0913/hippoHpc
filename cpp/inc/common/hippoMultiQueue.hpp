#ifndef __HIPPOMULTIQUEUE__H__
#define __HIPPOMULTIQUEUE__H__

#include "hippoCommonPublic.hpp"
#include <array>
#include <functional>

NAMESPACE_HIPPO_BEGIN
NAMESPACE_COMMON_BEGIN

template <typename DataT, size_t ItemSize>
struct FixedQueue {
    static constexpr size_t RealCapa = ItemSize + 1;
    using ContainerT = std::array<DataT, RealCapa>;
    struct Iterator {
        Iterator(ContainerT& ref, size_t index) : mDataRef(ref), mIndex(index) {}

        Iterator& operator++() {
            mIndex = (mIndex + 1U) % RealCapa;
            return *this;
        }

        DataT& operator*() const { return mDataRef[mIndex]; }

        bool operator!=(const Iterator& rhs) const { return mIndex != rhs.mIndex; }

    private:
        ContainerT& mDataRef;
        size_t mIndex;
    };

    struct ConstIterator {
        ConstIterator(ContainerT const& ref, size_t index) : mDataRef(ref), mIndex(index) {}

        ConstIterator& operator++() {
            mIndex = (mIndex + 1U) % RealCapa;
            return *this;
        }

        DataT const& operator*() const { return mDataRef[mIndex]; }

        bool operator!=(const ConstIterator& rhs) const { return mIndex != rhs.mIndex; }

    private:
        ContainerT const& mDataRef;
        size_t mIndex;
    };

    FixedQueue() : mData{}, mHead{0U}, mTail{0U} {}
    FixedQueue(std::initializer_list<DataT>&& l) : mData{}, mHead{0U}, mTail{0U} {
        for (auto const& i : l) {
            auto ret = push(i);
            if (!ret) {
                break;
            }
        }
    }
    ~FixedQueue() {}

    DataT& operator[](size_t idx) { return mData[(mHead + idx) % RealCapa]; }

    DataT const& operator[](size_t idx) const { return mData[(mHead + idx) % RealCapa]; }

    Iterator begin() { return Iterator(mData, mHead); }

    Iterator end() { return Iterator(mData, mTail); }

    ConstIterator begin() const { return ConstIterator(mData, mHead); }

    ConstIterator end() const { return ConstIterator(mData, mTail); }

    void clear() {
        mHead = 0U;
        mTail = 0U;
    }

    size_t size() { return (mTail + RealCapa - mHead) % RealCapa; }

    bool empty() const { return (mHead == mTail); }

    bool full() const { return ((mTail + 1) % RealCapa == mHead); }

    bool push(DataT const& data) {
        bool ret = false;
        if (!full()) {
            mData[mTail] = data;
            mTail = (mTail + 1U) % RealCapa;
            ret = true;
        }
        return ret;
    }

    Iterator reserve() {
        Iterator it(mData, mTail);
        if (!full()) {
            mTail = (mTail + 1) % RealCapa;
        }
        return it;
    }

    DataT* pick() {
        DataT* ret = nullptr;
        if (!empty()) {
            ret = &mData[mHead];
        }
        return ret;
    }

    bool pop() {
        bool ret = false;
        if (!empty()) {
            mHead = (mHead + 1) % RealCapa;
            ret = true;
        }
        return ret;
    }

    template <typename Ret, typename Func>
    Ret reduce(Ret const& default_, Func const& f) {
        Ret result = default_;
        for (auto const& i : *this) {
            result = f(result, i);
        }
        return result;
    }

private:
    ContainerT mData;
    size_t mHead;
    size_t mTail;
};

template <typename QueueIdxT, typename DataT, size_t QueueSize, size_t ItemSize>
struct FIFOMultiQueue {
    using ItemT = FixedQueue<DataT, ItemSize>;

    struct Mapping {
        QueueIdxT index;
        ItemT queue;
    };

    FIFOMultiQueue(){};
    ~FIFOMultiQueue(){};

    FIFOMultiQueue(FIFOMultiQueue const&) = delete;
    FIFOMultiQueue& operator=(FIFOMultiQueue const&) = delete;

    // overwrite : force deque old queue.
    bool push(QueueIdxT const& index, DataT const& data, bool overwrite = false, QueueIdxT* overwrited = nullptr) {
        bool ret = false;
        Mapping* qptr = nullptr;
        for (auto& q : mCache) {
            if (q.index == index) {
                qptr = &q;
                break;
            }
        }
        if (qptr) {
            ret = qptr->queue.push(data);
        } else {
            if (overwrite && mCache.full()) {
                if (overwrited) {
                    *overwrited = mCache.pick()->index;
                }
                mCache.pop();
            }

            auto it = mCache.reserve();
            ret = (it != mCache.end());
            if (ret) {
                (*it).index = index;
                (*it).queue.clear();
                (*it).queue.push(data);
            }
        }
        return ret;
    }

    bool anyFull(QueueIdxT& index) const {
        bool ret = false;
        for (auto const& q : mCache) {
            if (q.queue.full()) {
                index = q.index;
                ret = true;
                break;
            }
        }
        return ret;
    }

    bool anyFull(QueueIdxT& index, std::function<bool(Mapping const&)>&& full) {
        bool ret = false;
        for (auto const& q : mCache) {
            if (full(q)) {
                index = q.index;
                ret = true;
                break;
            }
        }
        return ret;
    }

    // inclusion : queue with specific index will also be dequed.
    bool popUntil(QueueIdxT& index, bool inclusion = false) {
        bool ret = false;
        for (auto const& q : mCache) {
            if (q.index != index) {
                mCache.pop();
            } else {
                if (inclusion) {
                    mCache.pop();
                }
                ret = true;
                break;
            }
        }
        return ret;
    }

    ItemT* find(QueueIdxT const& index) {
        ItemT* qptr = nullptr;
        for (auto& q : mCache) {
            if (q.index == index) {
                qptr = &(q.queue);
                break;
            }
        }
        return qptr;
    }

    Mapping* pick() const { return mCache.pick(); }

    bool pop() { return mCache.pop(); }

    bool full() { return mCache.full(); }

    QueueIdxT minIndex() {
        return mCache.reduce(0UL, [](auto const& idx, auto const& mapping) { return std::min(idx, mapping.index); });
    }

private:
    FixedQueue<Mapping, QueueSize> mCache;
};

NAMESPACE_COMMON_END
NAMESPACE_HIPPO_END

#endif  //!__HIPPOMULTIQUEUE__H__