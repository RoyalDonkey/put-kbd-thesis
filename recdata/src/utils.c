#include "utils.h"
#include <stdlib.h>

#ifdef __unix__
#include <time.h>
#include <sys/time.h>
double get_time() {
    struct timeval tval;
    if (gettimeofday(&tval, NULL))
        return 0;
    return (double)tval.tv_sec + tval.tv_usec / 1e6;
}

#elif defined(_WIN32)
#include <Windows.h>
double get_time() {
    LARGE_INTEGER counter, freq;

    if (!QueryPerformanceFrequency(&freq))
        return 0;
    if (!QueryPerformanceCounter(&counter))
        return 0;

    return (double)counter.QuadPart / freq.QuadPart;
}

#else
#error recdata only supports Unix and Windows operating systems
#endif

size_t strnlen(const char *s, size_t maxlen)
{
	size_t i = 0;
	for (; i < maxlen && s[i] != '\0'; i++);
	return i;
}
