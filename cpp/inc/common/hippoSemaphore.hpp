#ifndef __HIPPOSEMAPHORE__H__
#define __HIPPOSEMAPHORE__H__

#include "hippoPublic.hpp"
#include <semaphore.h>
#include <cassert>

NAMESPACE_HIPPO_BEGIN
NAMESPACE_COMMON_BEGIN

class Semaphore {
private:
    sem_t m_sema;

    Semaphore(const Semaphore& other) = delete;
    Semaphore& operator=(const Semaphore& other) = delete;

public:
    Semaphore(int initialCount = 0) {
        assert(initialCount >= 0);
        int rc = sem_init(&m_sema, 0, static_cast<unsigned int>(initialCount));
        assert(rc == 0);
        (void)rc;
    }

    ~Semaphore() { sem_destroy(&m_sema); }

    bool wait() {
        // http://stackoverflow.com/questions/2013181/gdb-causes-sem-wait-to-fail-with-eintr-error
        int rc;
        do {
            rc = sem_wait(&m_sema);
        } while (rc == -1 && errno == EINTR);
        return rc == 0;
    }

    bool try_wait() {
        int rc;
        do {
            rc = sem_trywait(&m_sema);
        } while (rc == -1 && errno == EINTR);
        return rc == 0;
    }

    bool timed_wait(std::uint64_t usecs) {
        struct timespec ts;
        constexpr int usecs_in_1_sec = 1000000;
        constexpr int nsecs_in_1_sec = 1000000000;
#ifdef HPC_CONFIG_SEM_MONOTONIC_CLOCK
        clock_gettime(CLOCK_MONOTONIC, &ts);
#else
        clock_gettime(CLOCK_REALTIME, &ts);
#endif
        ts.tv_sec += (time_t)(usecs / usecs_in_1_sec);
        ts.tv_nsec += (long)(usecs % usecs_in_1_sec) * 1000;
        // sem_timedwait bombs if you have more than 1e9 in tv_nsec
        // so we have to clean things up before passing it in
        if (ts.tv_nsec >= nsecs_in_1_sec) {
            ts.tv_nsec -= nsecs_in_1_sec;
            ++ts.tv_sec;
        }

        int rc;
        do {
#ifdef HPC_CONFIG_SEM_MONOTONIC_CLOCK
            rc = sem_clockwait(&m_sema, CLOCK_MONOTONIC, &ts);
#else
            rc = sem_timedwait(&m_sema, &ts);
#endif
        } while (rc == -1 && errno == EINTR);
        return rc == 0;
    }

    void signal() {
        while (sem_post(&m_sema) == -1)
            ;
    }

    void signal(int count) {
        while (count-- > 0) {
            while (sem_post(&m_sema) == -1)
                ;
        }
    }
};

NAMESPACE_COMMON_END
NAMESPACE_HIPPO_END

#endif  //!__HIPPOSEMAPHORE__H__