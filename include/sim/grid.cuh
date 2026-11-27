#pragma once

#include <cmath>
#include <cstdint>
#include "backend/cuda_compat.cuh"
#include "sim/particles.cuh"

namespace transport {

enum class BoundaryMode : int {
    PeriodicBox = 0,     // 3D periodic torus on [xmin, xmax) x [ymin, ymax) x [zmin, zmax)
    OpenExpanding = 1    // Open vacuum boundaries; particles outside active grid free-stream
};

struct GridConfig {
    BoundaryMode boundary = BoundaryMode::PeriodicBox;
    int nx = 10;
    int ny = 10;
    int nz = 10;
    double xmin = 0.0, xmax = 10.0;
    double ymin = 0.0, ymax = 10.0;
    double zmin = 0.0, zmax = 10.0;

    HD_INLINE int num_cells() const { return nx * ny * nz; }
    HD_INLINE double lx() const { return xmax - xmin; }
    HD_INLINE double ly() const { return ymax - ymin; }
    HD_INLINE double lz() const { return zmax - zmin; }
    HD_INLINE double dx() const { return (xmax - xmin) / static_cast<double>(nx); }
    HD_INLINE double dy() const { return (ymax - ymin) / static_cast<double>(ny); }
    HD_INLINE double dz() const { return (zmax - zmin) / static_cast<double>(nz); }
    HD_INLINE double cell_volume_fm3() const { return dx() * dy() * dz(); }
};

struct GridView {
    GridConfig cfg;
    int* cell_counts = nullptr;
    int* cell_start = nullptr;
    int* cell_write_offset = nullptr;
    int* sorted_indices = nullptr;
    int* active_cells = nullptr;
    int num_active_cells = 0;
};

namespace detail {

HD_INLINE Real wrap_periodic(Real pos, Real pmin, Real L) {
    Real rel = std::fmod(pos - pmin, L);
    if (rel < Real(0)) rel += L;
    if (rel >= L) rel = Real(0);
    return pmin + rel;
}

HD_INLINE int compute_cell_id(Real x, Real y, Real z, const GridConfig& cfg) {
    const Real xmin = static_cast<Real>(cfg.xmin), xmax = static_cast<Real>(cfg.xmax);
    const Real ymin = static_cast<Real>(cfg.ymin), ymax = static_cast<Real>(cfg.ymax);
    const Real zmin = static_cast<Real>(cfg.zmin), zmax = static_cast<Real>(cfg.zmax);
    if (cfg.boundary == BoundaryMode::OpenExpanding) {
        if (x < xmin || x >= xmax || y < ymin || y >= ymax || z < zmin || z >= zmax) {
            return -1;
        }
    }
    const Real inv_dx = static_cast<Real>(cfg.nx) / (xmax - xmin);
    const Real inv_dy = static_cast<Real>(cfg.ny) / (ymax - ymin);
    const Real inv_dz = static_cast<Real>(cfg.nz) / (zmax - zmin);
    int ix = static_cast<int>(std::floor((x - xmin) * inv_dx));
    int iy = static_cast<int>(std::floor((y - ymin) * inv_dy));
    int iz = static_cast<int>(std::floor((z - zmin) * inv_dz));

    if (cfg.boundary == BoundaryMode::PeriodicBox) {
        ix = (ix % cfg.nx + cfg.nx) % cfg.nx;
        iy = (iy % cfg.ny + cfg.ny) % cfg.ny;
        iz = (iz % cfg.nz + cfg.nz) % cfg.nz;
    } else {
        if (ix < 0 || ix >= cfg.nx || iy < 0 || iy >= cfg.ny || iz < 0 || iz >= cfg.nz) {
            return -1;
        }
    }
    return ix + cfg.nx * (iy + cfg.ny * iz);
}

#if TRANSPORT_HAS_CUDA

constexpr int SCAN_BLOCK_SIZE = 256;

__global__ void assign_and_count_cells_kernel(ParticleSoAView p, GridView g) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.n) return;
    if (!p.active[i] || p.freezeout_tag[i] != 0) {
        p.cell_id[i] = -1;
        if (!p.active[i]) p.collided[i] = 0;
        return;
    }

    const int cid = compute_cell_id(p.x[i], p.y[i], p.z[i], g.cfg);
    p.cell_id[i] = cid;
    p.collided[i] = 0;
    if (cid >= 0) {
        atomicAdd(&g.cell_counts[cid], 1);
    }
}

__global__ void scan_blocks_kernel(const int* __restrict__ d_in,
                                   int* __restrict__ d_out,
                                   int* __restrict__ d_block_sums,
                                   int n) {
    __shared__ int buf0[SCAN_BLOCK_SIZE];
    __shared__ int buf1[SCAN_BLOCK_SIZE];

    const int tid = threadIdx.x;
    const int gid = blockIdx.x * SCAN_BLOCK_SIZE + tid;

    buf0[tid] = (gid < n) ? d_in[gid] : 0;
    __syncthreads();

    int* src = buf0;
    int* dst = buf1;
    for (int offset = 1; offset < SCAN_BLOCK_SIZE; offset <<= 1) {
        if (tid >= offset) {
            dst[tid] = src[tid] + src[tid - offset];
        } else {
            dst[tid] = src[tid];
        }
        __syncthreads();
        int* tmp = src;
        src = dst;
        dst = tmp;
    }

    if (gid < n) {
        d_out[gid] = (tid == 0) ? 0 : src[tid - 1];
    }
    if (d_block_sums != nullptr && tid == SCAN_BLOCK_SIZE - 1) {
        d_block_sums[blockIdx.x] = src[SCAN_BLOCK_SIZE - 1];
    }
}

__global__ void add_block_offsets_kernel(int* __restrict__ d_out,
                                         const int* __restrict__ d_block_offsets,
                                         int n) {
    const int gid = blockIdx.x * SCAN_BLOCK_SIZE + threadIdx.x;
    if (gid >= n) return;
    d_out[gid] += d_block_offsets[blockIdx.x];
}

__global__ void copy_int_array_kernel(const int* __restrict__ src,
                                      int* __restrict__ dst,
                                      int n) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) dst[gid] = src[gid];
}

__global__ void scatter_particles_kernel(ParticleSoAView p, GridView g) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.n) return;
    const int cid = p.cell_id[i];
    if (cid >= 0) {
        const int slot = atomicAdd(&g.cell_write_offset[cid], 1);
        g.sorted_indices[slot] = i;
    }
}

__global__ void collect_active_cells_kernel(const int* __restrict__ cell_counts,
                                            int n_cells,
                                            int* __restrict__ active_cells,
                                            int* __restrict__ d_num_active) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < n_cells && cell_counts[c] >= 2) {
        const int idx = atomicAdd(d_num_active, 1);
        active_cells[idx] = c;
    }
}

#endif  // TRANSPORT_HAS_CUDA

}  // namespace detail

class SpatialGrid {
public:
    explicit SpatialGrid(const GridConfig& cfg = GridConfig{}) { configure(cfg); }

    void configure(const GridConfig& cfg) {
        cfg_ = cfg;
        const int n_cells = cfg_.num_cells();
        cell_counts_.resize(n_cells);
        cell_start_.resize(n_cells);
        cell_write_offset_.resize(n_cells);
        active_cells_.resize(n_cells > 0 ? n_cells : 1);
        d_num_active_cells_.resize(1);
        num_active_cells_ = 0;

        const int l1 = (n_cells + 255) / 256;
        const int l2 = (l1 + 255) / 256;
        l1_sums_.resize(l1 > 0 ? l1 : 1);
        l1_offsets_.resize(l1 > 0 ? l1 : 1);
        l2_sums_.resize(l2 > 0 ? l2 : 1);
        l2_offsets_.resize(l2 > 0 ? l2 : 1);
    }

    void build(ParticleSystem& particles) {
        const int np = particles.slot_size();
        const int n_cells = cfg_.num_cells();
        if (static_cast<int>(sorted_indices_.size()) < np) {
            sorted_indices_.resize(np > 0 ? np : 1);
        }

        cell_counts_.zero();
        num_active_cells_ = 0;
        if (np == 0 || n_cells == 0) {
            cell_start_.zero();
            return;
        }

        ParticleSoAView pview = particles.view();

#if TRANSPORT_HAS_CUDA
        GridView gview = view();
        constexpr int TPB = 256;
        const int p_blocks = (np + TPB - 1) / TPB;
        detail::assign_and_count_cells_kernel<<<p_blocks, TPB>>>(pview, gview);
        CUDA_CHECK(cudaGetLastError());

        exclusive_scan_device(cell_counts_.data(), cell_start_.data(), n_cells);

        const int c_blocks = (n_cells + TPB - 1) / TPB;
        detail::copy_int_array_kernel<<<c_blocks, TPB>>>(
            cell_start_.data(), cell_write_offset_.data(), n_cells);
        CUDA_CHECK(cudaGetLastError());

        detail::scatter_particles_kernel<<<p_blocks, TPB>>>(pview, gview);
        CUDA_CHECK(cudaGetLastError());

        d_num_active_cells_.zero();
        detail::collect_active_cells_kernel<<<c_blocks, TPB>>>(
            cell_counts_.data(), n_cells, active_cells_.data(), d_num_active_cells_.data());
        CUDA_CHECK(cudaGetLastError());
        d_num_active_cells_.copy_to_host(&num_active_cells_, 1);
#else
        int* counts = cell_counts_.data();
        int* starts = cell_start_.data();
        int* offsets = cell_write_offset_.data();
        int* sorted = sorted_indices_.data();
        int* act_cells = active_cells_.data();

        for (int i = 0; i < np; ++i) {
            if (!pview.active[i] || pview.freezeout_tag[i] != 0) {
                pview.cell_id[i] = -1;
                if (!pview.active[i]) pview.collided[i] = 0;
                continue;
            }
            const int cid = detail::compute_cell_id(pview.x[i], pview.y[i], pview.z[i], cfg_);
            pview.cell_id[i] = cid;
            pview.collided[i] = 0;
            if (cid >= 0) {
                counts[cid] += 1;
            }
        }

        int running = 0;
        for (int c = 0; c < n_cells; ++c) {
            starts[c] = running;
            offsets[c] = running;
            running += counts[c];
            if (counts[c] >= 2) act_cells[num_active_cells_++] = c;
        }

        for (int i = 0; i < np; ++i) {
            const int cid = pview.cell_id[i];
            if (cid >= 0) {
                sorted[offsets[cid]++] = i;
            }
        }
#endif
    }

    GridView view() {
        GridView gv;
        gv.cfg = cfg_;
        gv.cell_counts = cell_counts_.data();
        gv.cell_start = cell_start_.data();
        gv.cell_write_offset = cell_write_offset_.data();
        gv.sorted_indices = sorted_indices_.data();
        gv.active_cells = active_cells_.data();
        gv.num_active_cells = num_active_cells_;
        return gv;
    }

    int num_active_cells() const { return num_active_cells_; }
    const GridConfig& config() const { return cfg_; }

private:
#if TRANSPORT_HAS_CUDA
    void exclusive_scan_device(const int* d_in, int* d_out, int n) {
        using detail::SCAN_BLOCK_SIZE;
        const int l1_blocks = (n + SCAN_BLOCK_SIZE - 1) / SCAN_BLOCK_SIZE;
        if (l1_blocks <= 1) {
            detail::scan_blocks_kernel<<<1, SCAN_BLOCK_SIZE>>>(d_in, d_out, nullptr, n);
            CUDA_CHECK(cudaGetLastError());
            return;
        }

        detail::scan_blocks_kernel<<<l1_blocks, SCAN_BLOCK_SIZE>>>(
            d_in, d_out, l1_sums_.data(), n);
        CUDA_CHECK(cudaGetLastError());

        const int l2_blocks = (l1_blocks + SCAN_BLOCK_SIZE - 1) / SCAN_BLOCK_SIZE;
        if (l2_blocks <= 1) {
            detail::scan_blocks_kernel<<<1, SCAN_BLOCK_SIZE>>>(
                l1_sums_.data(), l1_offsets_.data(), nullptr, l1_blocks);
            CUDA_CHECK(cudaGetLastError());
        } else {
            detail::scan_blocks_kernel<<<l2_blocks, SCAN_BLOCK_SIZE>>>(
                l1_sums_.data(), l1_offsets_.data(), l2_sums_.data(), l1_blocks);
            CUDA_CHECK(cudaGetLastError());
            detail::scan_blocks_kernel<<<1, SCAN_BLOCK_SIZE>>>(
                l2_sums_.data(), l2_offsets_.data(), nullptr, l2_blocks);
            CUDA_CHECK(cudaGetLastError());
            detail::add_block_offsets_kernel<<<l2_blocks, SCAN_BLOCK_SIZE>>>(
                l1_offsets_.data(), l2_offsets_.data(), l1_blocks);
            CUDA_CHECK(cudaGetLastError());
        }

        detail::add_block_offsets_kernel<<<l1_blocks, SCAN_BLOCK_SIZE>>>(
            d_out, l1_offsets_.data(), n);
        CUDA_CHECK(cudaGetLastError());
    }
#endif

    GridConfig cfg_;
    DeviceBuffer<int> cell_counts_;
    DeviceBuffer<int> cell_start_;
    DeviceBuffer<int> cell_write_offset_;
    DeviceBuffer<int> sorted_indices_;
    DeviceBuffer<int> active_cells_;
    DeviceBuffer<int> d_num_active_cells_;
    int num_active_cells_ = 0;
    DeviceBuffer<int> l1_sums_, l1_offsets_;
    DeviceBuffer<int> l2_sums_, l2_offsets_;
};

}  // namespace transport
