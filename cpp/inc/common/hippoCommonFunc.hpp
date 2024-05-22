#ifndef __HIPPOCOMMONFUNC__H__
#define __HIPPOCOMMONFUNC__H__

#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include "hippoCommonPublic.hpp"

NAMESPACE_HIPPO_BEGIN
NAMESPACE_COMMON_BEGIN

size_t GetFileSize(const std::string& filepath) {
    std::ifstream in(filepath, std::ios::binary);
    in.seekg(0, std::ios::end);
    size_t size = in.tellg();
    in.close();
    return size;
}

std::string GetFatherDirectory(const std::string& path) {
    std::string::size_type pos = path.find_last_of('/');
    if (pos == std::string::npos) {
        return "";
    }
    return path.substr(0, pos);
}

std::string GetFileNameWithoutExtension(const std::string& path) {
    std::string::size_type slashPos = path.find_last_of('/');
    if (slashPos == std::string::npos) {
        return path;
    }
    std::string::size_type dotPos = path.find_last_of('.');
    if (dotPos == std::string::npos) {
        return path;
    }
    return path.substr(slashPos + 1, dotPos - slashPos - 1);
}

NAMESPACE_COMMON_END NAMESPACE_HIPPO_END

#endif  //!__HIPPOCOMMONFUNC__H__