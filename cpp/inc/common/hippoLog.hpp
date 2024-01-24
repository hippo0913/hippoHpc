#ifndef __HIPPOLOG__H__
#define __HIPPOLOG__H__

#include <string.h>
#ifndef GLOG_NO_ABBREVIATED_SEVERITIES
#define GLOG_NO_ABBREVIATED_SEVERITIES  // 如果不加这个宏定义代码就会报错
#endif
#include "glog/logging.h"

void InitializeGlog() {
    google::InitGoogleLogging("");  //使用glog之前必须先初始化库，仅需执行一次，括号内为程序名
    FLAGS_alsologtostderr = true;   //是否将日志输出到文件和stderr
    FLAGS_colorlogtostderr = true;  //是否启用不同颜色显示

    char *cwd = nullptr;
    cwd = get_current_dir_name();
    std::string logDir = std::string(cwd) + std::string("/log");
    std::string info = logDir + std::string("/INFO_");
    std::string warning = logDir + std::string("/WARNING_");
    std::string error = logDir + std::string("/ERROR_");
    std::string fatal = logDir + std::string("/FATAL_");

    // INFO级别的日志都存放到logs目录下且前缀为INFO_
    google::SetLogDestination(google::GLOG_INFO, info.c_str());
    // WARNING级别的日志都存放到logs目录下且前缀为WARNING_
    google::SetLogDestination(google::GLOG_WARNING, warning.c_str());
    // ERROR级别的日志都存放到logs目录下且前缀为ERROR_
    google::SetLogDestination(google::GLOG_ERROR, error.c_str());
    // FATAL级别的日志都存放到logs目录下且前缀为FATAL_
    google::SetLogDestination(google::GLOG_FATAL, fatal.c_str());
}

void finalizeGlog() {
    google::ShutdownGoogleLogging();  //当要结束glog时必须关闭库，否则会内存溢出
}

#endif  //!__HIPPOLOG__H__