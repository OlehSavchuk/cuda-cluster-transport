#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__CUDACC__)
#define TRANSPORT_HAS_CUDA 1
#include <cuda_runtime.h>
#define HD_INLINE __host__ __device__ __forceinline__
#define D_INLINE __device__ __forceinline__
#define GLOBAL_KERNEL __global__
#else
#define TRANSPORT_HAS_CUDA 0
#define HD_INLINE inline
#define D_INLINE inline
#define GLOBAL_KERNEL inline
#endif

namespace transport {

// Default to double precision for exact relativistic kinematics from keV nuclear
// excitation scales up to multi-GeV heavy-ion energies.
#if defined(TRANSPORT_SINGLE_PRECISION)
using Real = float;
#else
using Real = double;
#endif

#if TRANSPORT_HAS_CUDA

#define CUDA_CHECK(expr)                                                                 \
    do {                                                                                 \
        cudaError_t err__ = (expr);                                                      \
        if (err__ != cudaSuccess) {                                                      \
            std::fprintf(stderr, "CUDA error at %s:%d in %s: %s\n", __FILE__, __LINE__,  \
                         #expr, cudaGetErrorString(err__));                              \
            std::abort();                                                                \
        }                                                                                \
    } while (0)

inline void device_synchronize() {
    CUDA_CHECK(cudaDeviceSynchronize());
}

inline std::string backend_description() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) {
        return "CUDA (no active GPU detected)";
    }
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    char buf[256];
    std::snprintf(buf, sizeof(buf), "CUDA GPU [%s, SM %d.%d, %.1f GB VRAM, %s]",
                  prop.name, prop.major, prop.minor,
                  static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0),
                  (sizeof(Real) == 4) ? "FP32 float" : "FP64 double");
    return std::string(buf);
}

#else  // Host C++ fallback when compiled without nvcc (e.g. macOS local testing)

#define CUDA_CHECK(expr) ((void)(expr))

inline void device_synchronize() {}

inline std::string backend_description() {
    return (sizeof(Real) == 4)
        ? "Host C++17 Fallback [FP32 float] (compile with nvcc on GPU for 12.7 TFLOPS FP32)"
        : "Host C++17 Fallback [FP64 double] (compile with nvcc / -DUSE_CUDA=ON on GPU machine)";
}

// Host atomic helpers matching CUDA semantics for single/multi-threaded CPU execution
template <typename T>
inline T host_atomic_add(T* address, T val) {
#if defined(__clang__) || defined(__GNUC__)
    return __atomic_fetch_add(address, val, __ATOMIC_RELAXED);
#else
    T old = *address;
    *address += val;
    return old;
#endif
}

inline double host_atomic_add_double(double* address, double val) {
#if defined(__clang__) || defined(__GNUC__)
    unsigned long long* addr_ull = reinterpret_cast<unsigned long long*>(address);
    unsigned long long old_ull = __atomic_load_n(addr_ull, __ATOMIC_RELAXED);
    unsigned long long assumed;
    do {
        assumed = old_ull;
        double old_d;
        std::memcpy(&old_d, &assumed, sizeof(double));
        double new_d = old_d + val;
        unsigned long long new_ull;
        std::memcpy(&new_ull, &new_d, sizeof(double));
        old_ull = __sync_val_compare_and_swap(addr_ull, assumed, new_ull);
    } while (assumed != old_ull);
    double res;
    std::memcpy(&res, &old_ull, sizeof(double));
    return res;
#else
    double old = *address;
    *address += val;
    return old;
#endif
}

inline int host_atomic_cas(int* address, int compare, int val) {
#if defined(__clang__) || defined(__GNUC__)
    return __sync_val_compare_and_swap(address, compare, val);
#else
    int old = *address;
    if (old == compare) *address = val;
    return old;
#endif
}

inline void host_atomic_max_double(double* address, double val) {
#if defined(__clang__) || defined(__GNUC__)
    unsigned long long* addr_ull = reinterpret_cast<unsigned long long*>(address);
    unsigned long long old_ull = __atomic_load_n(addr_ull, __ATOMIC_RELAXED);
    while (true) {
        double old_d;
        std::memcpy(&old_d, &old_ull, sizeof(double));
        if (old_d >= val) break;
        unsigned long long new_ull;
        std::memcpy(&new_ull, &val, sizeof(double));
        unsigned long long prev = __sync_val_compare_and_swap(addr_ull, old_ull, new_ull);
        if (prev == old_ull) break;
        old_ull = prev;
    }
#else
    if (val > *address) *address = val;
#endif
}

#endif  // TRANSPORT_HAS_CUDA

// Portable atomic wrappers usable inside __device__ and host fallback code
HD_INLINE int portable_atomic_add_int(int* addr, int val) {
#if defined(__CUDA_ARCH__)
    return atomicAdd(addr, val);
#elif !TRANSPORT_HAS_CUDA
    return host_atomic_add(addr, val);
#else
    *addr += val;
    return *addr - val;
#endif
}

HD_INLINE unsigned long long portable_atomic_add_ull(unsigned long long* addr,
                                                     unsigned long long val) {
#if defined(__CUDA_ARCH__)
    return atomicAdd(addr, val);
#elif !TRANSPORT_HAS_CUDA
    return host_atomic_add(addr, val);
#else
    *addr += val;
    return *addr - val;
#endif
}

HD_INLINE double portable_atomic_add_double(double* addr, double val) {
#if defined(__CUDA_ARCH__)
#if __CUDA_ARCH__ >= 600
    return atomicAdd(addr, val);
#else
    unsigned long long int* address_as_ull = reinterpret_cast<unsigned long long int*>(addr);
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
#endif
#elif !TRANSPORT_HAS_CUDA
    return host_atomic_add_double(addr, val);
#else
    double old = *addr;
    *addr += val;
    return old;
#endif
}

HD_INLINE float portable_atomic_add_float(float* addr, float val) {
#if defined(__CUDA_ARCH__)
    return atomicAdd(addr, val);
#elif !TRANSPORT_HAS_CUDA && (defined(__clang__) || defined(__GNUC__))
    unsigned int* addr_u = reinterpret_cast<unsigned int*>(addr);
    unsigned int old_u = __atomic_load_n(addr_u, __ATOMIC_RELAXED);
    unsigned int assumed;
    do {
        assumed = old_u;
        float old_f;
        std::memcpy(&old_f, &assumed, sizeof(float));
        float new_f = old_f + val;
        unsigned int new_u;
        std::memcpy(&new_u, &new_f, sizeof(float));
        old_u = __sync_val_compare_and_swap(addr_u, assumed, new_u);
    } while (assumed != old_u);
    float res;
    std::memcpy(&res, &old_u, sizeof(float));
    return res;
#else
    float old = *addr;
    *addr += val;
    return old;
#endif
}

HD_INLINE Real portable_atomic_add_real(Real* addr, Real val) {
#if defined(TRANSPORT_SINGLE_PRECISION)
    return portable_atomic_add_float(addr, val);
#else
    return portable_atomic_add_double(addr, val);
#endif
}

HD_INLINE int portable_atomic_cas_int(int* addr, int compare, int val) {
#if defined(__CUDA_ARCH__)
    return atomicCAS(addr, compare, val);
#elif !TRANSPORT_HAS_CUDA
    return host_atomic_cas(addr, compare, val);
#else
    int old = *addr;
    if (old == compare) *addr = val;
    return old;
#endif
}

HD_INLINE void portable_atomic_max_double(double* addr, double val) {
#if defined(__CUDA_ARCH__)
    unsigned long long int* address_as_ull = reinterpret_cast<unsigned long long int*>(addr);
    unsigned long long int old = *address_as_ull;
    while (__longlong_as_double(old) < val) {
        unsigned long long int assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val));
        if (old == assumed) break;
    }
#elif !TRANSPORT_HAS_CUDA
    host_atomic_max_double(addr, val);
#else
    if (val > *addr) *addr = val;
#endif
}

// RAII buffer managing GPU device memory (or host memory in CPU fallback mode)
template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(std::size_t count) { resize(count); }

    ~DeviceBuffer() { release(); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : ptr_(other.ptr_), size_(other.size_), capacity_(other.capacity_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
        other.capacity_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            release();
            ptr_ = other.ptr_;
            size_ = other.size_;
            capacity_ = other.capacity_;
            other.ptr_ = nullptr;
            other.size_ = 0;
            other.capacity_ = 0;
        }
        return *this;
    }

    void resize(std::size_t count) {
        if (count > capacity_) {
            release();
#if TRANSPORT_HAS_CUDA
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_), count * sizeof(T)));
#else
            ptr_ = static_cast<T*>(std::malloc(count * sizeof(T)));
            if (!ptr_ && count > 0) throw std::bad_alloc();
#endif
            capacity_ = count;
        }
        size_ = count;
    }

    void zero() {
        if (size_ == 0 || !ptr_) return;
#if TRANSPORT_HAS_CUDA
        CUDA_CHECK(cudaMemset(ptr_, 0, size_ * sizeof(T)));
#else
        std::memset(ptr_, 0, size_ * sizeof(T));
#endif
    }

    void copy_from_host(const T* src, std::size_t count) {
        resize(count);
        if (count == 0) return;
#if TRANSPORT_HAS_CUDA
        CUDA_CHECK(cudaMemcpy(ptr_, src, count * sizeof(T), cudaMemcpyHostToDevice));
#else
        std::memcpy(ptr_, src, count * sizeof(T));
#endif
    }

    void copy_from_host(const std::vector<T>& src) {
        copy_from_host(src.data(), src.size());
    }

    void copy_to_host(T* dst, std::size_t count) const {
        if (count == 0) return;
#if TRANSPORT_HAS_CUDA
        CUDA_CHECK(cudaMemcpy(dst, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost));
#else
        std::memcpy(dst, ptr_, count * sizeof(T));
#endif
    }

    std::vector<T> to_vector() const {
        std::vector<T> out(size_);
        copy_to_host(out.data(), size_);
        return out;
    }

    void swap(DeviceBuffer& other) noexcept {
        T* tp = ptr_; ptr_ = other.ptr_; other.ptr_ = tp;
        std::size_t ts = size_; size_ = other.size_; other.size_ = ts;
        std::size_t tc = capacity_; capacity_ = other.capacity_; other.capacity_ = tc;
    }

    T* data() { return ptr_; }
    const T* data() const { return ptr_; }
    std::size_t size() const { return size_; }

private:
    void release() {
        if (ptr_) {
#if TRANSPORT_HAS_CUDA
            cudaFree(ptr_);
#else
            std::free(ptr_);
#endif
            ptr_ = nullptr;
            size_ = 0;
            capacity_ = 0;
        }
    }

    T* ptr_ = nullptr;
    std::size_t size_ = 0;
    std::size_t capacity_ = 0;
};

}  // namespace transport
