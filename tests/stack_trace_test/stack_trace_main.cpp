#include <csignal>
#include <iostream>

#if Backtrace_enable
#define BOOST_STACKTRACE_USE_BACKTRACE
#endif
#ifdef __APPLE__
#define _GNU_SOURCE
#endif
#include <boost/stacktrace.hpp>

void signalHandler(int sig_num)
{
    std::cerr << boost::stacktrace::stacktrace();
    exit(sig_num);
}

int main()
{
    signal(SIGSEGV, signalHandler);
    signal(SIGABRT, signalHandler);

    int *foo = (int*)-1; // make a bad pointer
    printf("%d\n", *foo);
}