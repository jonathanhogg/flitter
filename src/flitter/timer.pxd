
cdef extern from *:
    """
#include "Python.h"

#if PY_MINOR_VERSION >= 13
    static inline double perf_counter() {
        PyTime_t result;
        PyTime_PerfCounter(&result);
        return PyTime_AsSecondsDouble(result);
    }
#else
    static inline double perf_counter() {
        return _PyTime_AsSecondsDouble(_PyTime_GetPerfCounter());
    }
#endif
    """

    cdef double perf_counter() noexcept nogil
