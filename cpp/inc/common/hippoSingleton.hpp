#ifndef __HIPPOSINGLETON__H__
#define __HIPPOSINGLETON__H__

#include "hippoPublic.hpp"

NAMESPACE_HIPPO_BEGIN
NAMESPACE_COMMON_BEGIN

// A boost style Signleton implementation
// T must be: no-throw default constructible and no-throw destructible
template <typename T>
struct GlobalSingleton
{
public:
    typedef T object_type;

    // If, at any point (in user code), GlobalSingleton<T>::instance()
    //  is called, then the following function is instantiated.
    static object_type &instance()
    {
        // This is the object that we return a reference to.
        // It is guaranteed to be created before main() begins because of
        //  the next line.
        static object_type obj{};

        // The following line does nothing else than force the instantiation
        //  of GlobalSingleton<T>::create_object_, whose constructor is
        //  called before main() begins.
        create_object_.do_nothing();

        return obj;
    }

private:
    struct _ObjectCreator
    {
        // This constructor does nothing more than ensure that instance()
        //  is called before main() begins, thus creating the static
        //  T object before multithreading race issues can come up.
        _ObjectCreator() { GlobalSingleton<T>::instance(); }
        inline void do_nothing() const {}
    };
    static _ObjectCreator create_object_;

    GlobalSingleton();
};

template <typename T>
typename GlobalSingleton<T>::_ObjectCreator GlobalSingleton<T>::create_object_;

// A thread local scoped Signleton implementation
// T must be: no-throw default constructible and no-throw destructible
template <typename T>
struct ThreadScopedSingleton
{
public:
    typedef T object_type;

    static object_type &instance()
    {
        // wrapper must be a `thread_local` object to make sure
        // there is only one instance in per-thread.
        static thread_local LazyInitializationWrapper wrapper_{};

        return wrapper_.instance();
    }

private:
    struct LazyInitializationWrapper
    {
        // for a thread local variable, instances should not
        // be created automatically, so delay the instance
        // creation untill instance() was invoked.
        LazyInitializationWrapper() : instance_ptr_(nullptr) {}

        ~LazyInitializationWrapper()
        {
            if (instance_ptr_)
            {
                delete instance_ptr_;
                instance_ptr_ = nullptr;
            }
        }

        object_type &instance()
        {
            if (nullptr == instance_ptr_)
            {
                instance_ptr_ = new object_type{};
            }
            return *instance_ptr_;
        }

        object_type *instance_ptr_;
    };

    ThreadScopedSingleton();
};

NAMESPACE_COMMON_END
NAMESPACE_HIPPO_END

#endif //!__HIPPOSINGLETON__H__