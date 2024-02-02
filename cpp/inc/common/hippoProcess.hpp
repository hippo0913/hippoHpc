#ifndef __HIPPOPROCESS__H__
#define __HIPPOPROCESS__H__

#include <iostream>
#include <unistd.h>
#include <string.h>
#include <limits.h>
#include "hippoSingleton.hpp"
#include "hippoLog.hpp"

NAMESPACE_HIPPO_BEGIN
NAMESPACE_COMMON_BEGIN

class hippoProcess {
public:
    explicit hippoProcess() {}
    int initialize() {
        InitializeGlog();
        return 0;
    }
    int finalize() {
        finalizeGlog();
        return 0;
    }

    std::string getSelfProcessName() {
        char path[PATH_MAX] = {0};
        if (readlink("/proc/self/exe", path, sizeof(path) - 1) <= 0) {
            return 0;
        }

        std::string processName = path;
        size_t pos = processName.find_last_of("\\/");
        processName = processName.substr(pos + 1);
        return processName;
    }
};

NAMESPACE_COMMON_END
NAMESPACE_HIPPO_END

#define hippoProcessInst (hippo::common::GlobalSingleton<hippo::hippoProcess>::instance())

#endif  //!__HIPPOPROCESS__H__