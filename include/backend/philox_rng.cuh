#pragma once

#include <cmath>
#include <cstdint>
#include "backend/cuda_compat.cuh"

namespace transport {

// Counter-based Philox4x32-10 PRNG (Salmon et al., SC11; identical constants to cuRAND).
// Stateless & thread-schedule invariant: requires zero global memory per thread/particle.
struct PhiloxRng {
    std::uint32_t ctr[4];
    std::uint32_t key[2];
    std::uint32_t out[4];
    int sub_idx;

    HD_INLINE PhiloxRng(std::uint64_t seed,
                        std::uint64_t stream_hi,
                        std::uint64_t stream_lo,
                        std::uint32_t draw_block = 0)
        : sub_idx(4) {
        key[0] = static_cast<std::uint32_t>(seed);
        key[1] = static_cast<std::uint32_t>(seed >> 32);
        ctr[0] = draw_block;
        ctr[1] = static_cast<std::uint32_t>(stream_lo);
        ctr[2] = static_cast<std::uint32_t>(stream_lo >> 32);
        ctr[3] = static_cast<std::uint32_t>(stream_hi ^ (stream_hi >> 32));
    }

    static HD_INLINE void mulhilo32(std::uint32_t a, std::uint32_t b,
                                    std::uint32_t& hi, std::uint32_t& lo) {
        std::uint64_t prod = static_cast<std::uint64_t>(a) * static_cast<std::uint64_t>(b);
        lo = static_cast<std::uint32_t>(prod);
        hi = static_cast<std::uint32_t>(prod >> 32);
    }

    HD_INLINE void refill() {
        constexpr std::uint32_t PHILOX_M0 = 0xD2511F53U;
        constexpr std::uint32_t PHILOX_M1 = 0xCD9E8D57U;
        constexpr std::uint32_t PHILOX_W0 = 0x9E3779B9U;
        constexpr std::uint32_t PHILOX_W1 = 0xBB67AE85U;

        std::uint32_t c0 = ctr[0];
        std::uint32_t c1 = ctr[1];
        std::uint32_t c2 = ctr[2];
        std::uint32_t c3 = ctr[3];
        std::uint32_t k0 = key[0];
        std::uint32_t k1 = key[1];

#if defined(__CUDA_ARCH__)
        #pragma unroll
#endif
        for (int round = 0; round < 10; ++round) {
            std::uint32_t hi0, lo0, hi1, lo1;
            mulhilo32(PHILOX_M0, c0, hi0, lo0);
            mulhilo32(PHILOX_M1, c2, hi1, lo1);
            c0 = hi1 ^ c1 ^ k0;
            c1 = lo1;
            c2 = hi0 ^ c3 ^ k1;
            c3 = lo0;
            k0 += PHILOX_W0;
            k1 += PHILOX_W1;
        }

        out[0] = c0;
        out[1] = c1;
        out[2] = c2;
        out[3] = c3;
        ++ctr[0];
        if (ctr[0] == 0) ++ctr[1];
        sub_idx = 0;
    }

    HD_INLINE std::uint32_t next_u32() {
        if (sub_idx >= 4) refill();
        return out[sub_idx++];
    }

    // Uniform in (0, 1) with 53-bit double precision
    HD_INLINE double uniform() {
        std::uint64_t hi = static_cast<std::uint64_t>(next_u32() >> 5);
        std::uint64_t lo = static_cast<std::uint64_t>(next_u32() >> 6);
        std::uint64_t combined = (hi << 26) | lo;
        // (combined + 0.5) * 2^-53 lies strictly in (0, 1)
        return (static_cast<double>(combined) + 0.5) * (1.0 / 9007199254740992.0);
    }

    // Uniform integer in [0, bound - 1] using Lemire's fast nearly-divisionless method
    HD_INLINE std::uint32_t uniform_int(std::uint32_t bound) {
        if (bound <= 1) return 0;
        std::uint32_t x = next_u32();
        std::uint64_t m = static_cast<std::uint64_t>(x) * static_cast<std::uint64_t>(bound);
        std::uint32_t l = static_cast<std::uint32_t>(m);
        if (l < bound) {
            std::uint32_t t = (0U - bound) % bound;
            while (l < t) {
                x = next_u32();
                m = static_cast<std::uint64_t>(x) * static_cast<std::uint64_t>(bound);
                l = static_cast<std::uint32_t>(m);
            }
        }
        return static_cast<std::uint32_t>(m >> 32);
    }

    // Standard normal N(0, 1) via Box-Muller
    HD_INLINE double normal() {
        double u1 = uniform();
        double u2 = uniform();
        return ::sqrt(-2.0 * ::log(u1)) * ::cos(6.28318530717958647692 * u2);
    }
};

}  // namespace transport
