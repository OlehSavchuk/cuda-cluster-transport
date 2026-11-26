#pragma once

#include <cmath>
#include <cstring>
#include <vector>

#include "backend/cuda_compat.cuh"
#include "physics/constants.cuh"
#include "sim/grid.cuh"

namespace transport {
namespace coulomb_detail {

// Build 1D orthonormal eigenbasis Q (row m, col i: Q[m * N + i]) and eigenvalues lam[m]
// of the 1D discrete negative second-derivative operator (-d^2/dx^2):
//   - OpenExpanding: cell-centered ghost-Dirichlet DST-I basis on N points
//   - PeriodicBox  : real Fourier (1, cos, sin, Nyquist) basis on N points (lam[0] = 0)
inline void build_1d_laplacian_eigenbasis(int N, double dx, BoundaryMode bmode,
                                          std::vector<double>& Q_out,
                                          std::vector<double>& lam_out) {
    Q_out.assign(static_cast<std::size_t>(N * N), 0.0);
    lam_out.assign(static_cast<std::size_t>(N), 0.0);
    if (N <= 0) return;

    const double inv_dx2 = 1.0 / (dx * dx);
    if (bmode == BoundaryMode::OpenExpanding) {
        const double norm = std::sqrt(2.0 / static_cast<double>(N + 1));
        for (int m = 0; m < N; ++m) {
            const double s = std::sin(0.5 * phys::PI * static_cast<double>(m + 1) / static_cast<double>(N + 1));
            lam_out[m] = 4.0 * inv_dx2 * s * s;
            for (int i = 0; i < N; ++i) {
                Q_out[m * N + i] = norm * std::sin(phys::PI * static_cast<double>((m + 1) * (i + 1)) /
                                                   static_cast<double>(N + 1));
            }
        }
    } else {
        const double inv_sqrtN = 1.0 / std::sqrt(static_cast<double>(N));
        for (int i = 0; i < N; ++i) Q_out[0 * N + i] = inv_sqrtN;
        lam_out[0] = 0.0;
        int m = 1;
        const int k_max = (N - 1) / 2;
        const double norm2 = std::sqrt(2.0 / static_cast<double>(N));
        for (int k = 1; k <= k_max; ++k) {
            const double s = std::sin(phys::PI * static_cast<double>(k) / static_cast<double>(N));
            const double ev = 4.0 * inv_dx2 * s * s;
            lam_out[m] = ev;
            lam_out[m + 1] = ev;
            for (int i = 0; i < N; ++i) {
                const double ang = phys::TWO_PI * static_cast<double>(k * i) / static_cast<double>(N);
                Q_out[m * N + i]       = norm2 * std::cos(ang);
                Q_out[(m + 1) * N + i] = norm2 * std::sin(ang);
            }
            m += 2;
        }
        if (N % 2 == 0 && m < N) {
            lam_out[m] = 4.0 * inv_dx2;
            for (int i = 0; i < N; ++i) {
                Q_out[m * N + i] = ((i % 2 == 0) ? 1.0 : -1.0) * inv_sqrtN;
            }
        }
    }
}

// Exterior multipole electrostatic potential (l = 0, 1, 2) [GeV] around box center r0
// m[0] = Q, m[1..3] = D_x, D_y, D_z, m[4..9] = Q_xx, Q_xy, Q_xz, Q_yy, Q_yz, Q_zz
HD_INLINE double eval_multipole_potential_gev(double rx, double ry, double rz, const double* m) {
    const double r2 = rx * rx + ry * ry + rz * rz;
    if (r2 <= 1.0e-12) return 0.0;
    const double inv_r = 1.0 / ::sqrt(r2);
    const double inv_r2 = inv_r * inv_r;
    const double inv_r3 = inv_r2 * inv_r;
    const double inv_r5 = inv_r3 * inv_r2;
    const double mono = m[0] * inv_r;
    const double dip = (m[1] * rx + m[2] * ry + m[3] * rz) * inv_r3;
    const double quad_rr = m[4] * rx * rx + m[7] * ry * ry + m[9] * rz * rz +
                           2.0 * (m[5] * rx * ry + m[6] * rx * rz + m[8] * ry * rz);
    return phys::E2_COULOMB_GEV_FM * (mono + dip + 0.5 * quad_rr * inv_r5);
}

HD_INLINE void accum_cell_multipoles(const Real* rho_q, int c, const GridConfig& g, double* mp10) {
    const double rq = static_cast<double>(rho_q[c]);
    if (::fabs(rq) <= 1.0e-15) return;
    const int ix = c % g.nx;
    const int iy = (c / g.nx) % g.ny;
    const int iz = c / (g.nx * g.ny);
    const double dV = g.cell_volume_fm3();
    const double x0 = 0.5 * (g.xmin + g.xmax);
    const double y0 = 0.5 * (g.ymin + g.ymax);
    const double z0 = 0.5 * (g.zmin + g.zmax);
    const double rx = (g.xmin + (ix + 0.5) * g.dx()) - x0;
    const double ry = (g.ymin + (iy + 0.5) * g.dy()) - y0;
    const double rz = (g.zmin + (iz + 0.5) * g.dz()) - z0;
    const double r2 = rx * rx + ry * ry + rz * rz;
    const double q = rq * dV;

    portable_atomic_add_double(&mp10[0], q);
    portable_atomic_add_double(&mp10[1], q * rx);
    portable_atomic_add_double(&mp10[2], q * ry);
    portable_atomic_add_double(&mp10[3], q * rz);
    portable_atomic_add_double(&mp10[4], q * (3.0 * rx * rx - r2));
    portable_atomic_add_double(&mp10[5], q * (3.0 * rx * ry));
    portable_atomic_add_double(&mp10[6], q * (3.0 * rx * rz));
    portable_atomic_add_double(&mp10[7], q * (3.0 * ry * ry - r2));
    portable_atomic_add_double(&mp10[8], q * (3.0 * ry * rz));
    portable_atomic_add_double(&mp10[9], q * (3.0 * rz * rz - r2));
}

HD_INLINE void build_poisson_rhs_cell(const Real* rho_q, double* rhs, int c,
                                      const GridConfig& g, const double* mp10) {
    const double pref = 4.0 * phys::PI * phys::E2_COULOMB_GEV_FM;
    double val = pref * static_cast<double>(rho_q[c]);
    if (g.boundary == BoundaryMode::OpenExpanding) {
        const int ix = c % g.nx;
        const int iy = (c / g.nx) % g.ny;
        const int iz = c / (g.nx * g.ny);
        const double dx = g.dx(), dy = g.dy(), dz = g.dz();
        const double rx = (g.xmin + (ix + 0.5) * dx) - 0.5 * (g.xmin + g.xmax);
        const double ry = (g.ymin + (iy + 0.5) * dy) - 0.5 * (g.ymin + g.ymax);
        const double rz = (g.zmin + (iz + 0.5) * dz) - 0.5 * (g.zmin + g.zmax);

        if (ix == 0)        val += eval_multipole_potential_gev(rx - dx, ry, rz, mp10) / (dx * dx);
        if (ix == g.nx - 1) val += eval_multipole_potential_gev(rx + dx, ry, rz, mp10) / (dx * dx);
        if (iy == 0)        val += eval_multipole_potential_gev(rx, ry - dy, rz, mp10) / (dy * dy);
        if (iy == g.ny - 1) val += eval_multipole_potential_gev(rx, ry + dy, rz, mp10) / (dy * dy);
        if (iz == 0)        val += eval_multipole_potential_gev(rx, ry, rz - dz, mp10) / (dz * dz);
        if (iz == g.nz - 1) val += eval_multipole_potential_gev(rx, ry, rz + dz, mp10) / (dz * dz);
    }
    rhs[c] = val;
}

// 1D matrix-tensor contraction along axis in {0, 1, 2}.
// forward == true : dst(..., m, ...) = sum_i Q[m * N + i] * src(..., i, ...)
// forward == false: dst(..., i, ...) = sum_m Q[m * N + i] * src(..., m, ...)
HD_INLINE void transform_axis_1d_cell(const double* src, double* dst, const double* Q,
                                      int cid, int nx, int ny, int nz, int axis, bool forward) {
    const int ix = cid % nx;
    const int iy = (cid / nx) % ny;
    const int iz = cid / (nx * ny);
    double sum = 0.0;
    if (axis == 0) {
        const int base = nx * (iy + ny * iz);
        if (forward) {
            for (int k = 0; k < nx; ++k) sum += Q[ix * nx + k] * src[base + k];
        } else {
            for (int m = 0; m < nx; ++m) sum += Q[m * nx + ix] * src[base + m];
        }
    } else if (axis == 1) {
        if (forward) {
            for (int k = 0; k < ny; ++k) sum += Q[iy * ny + k] * src[ix + nx * (k + ny * iz)];
        } else {
            for (int m = 0; m < ny; ++m) sum += Q[m * ny + iy] * src[ix + nx * (m + ny * iz)];
        }
    } else {
        const int stride_z = nx * ny;
        const int base = ix + nx * iy;
        if (forward) {
            for (int k = 0; k < nz; ++k) sum += Q[iz * nz + k] * src[base + k * stride_z];
        } else {
            for (int m = 0; m < nz; ++m) sum += Q[m * nz + iz] * src[base + m * stride_z];
        }
    }
    dst[cid] = sum;
}

HD_INLINE void spectral_poisson_div_cell(double* spec, Real* phi_raw_out, int cid,
                                         int nx, int ny, int nz,
                                         const double* lx, const double* ly, const double* lz,
                                         bool write_out) {
    (void)nz;
    if (!write_out) {
        const int ix = cid % nx;
        const int iy = (cid / nx) % ny;
        const int iz = cid / (nx * ny);
        const double denom = lx[ix] + ly[iy] + lz[iz];
        spec[cid] = (denom > 1.0e-13) ? (spec[cid] / denom) : 0.0;
    } else {
        phi_raw_out[cid] = static_cast<Real>(spec[cid]);
    }
}

#if TRANSPORT_HAS_CUDA
__global__ void accum_multipoles_kernel(const Real* rho_q, int nc, GridConfig g, double* mp10) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < nc) accum_cell_multipoles(rho_q, c, g, mp10);
}
__global__ void build_rhs_kernel(const Real* rho_q, double* rhs, int nc, GridConfig g, const double* mp10) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < nc) build_poisson_rhs_cell(rho_q, rhs, c, g, mp10);
}
__global__ void transform_axis_kernel(const double* src, double* dst, const double* Q,
                                      int nc, int nx, int ny, int nz, int axis, bool forward) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < nc) transform_axis_1d_cell(src, dst, Q, c, nx, ny, nz, axis, forward);
}
__global__ void spectral_div_or_cast_kernel(double* spec, Real* phi_out, int nc,
                                            int nx, int ny, int nz,
                                            const double* lx, const double* ly, const double* lz,
                                            bool write_out) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < nc) spectral_poisson_div_cell(spec, phi_out, c, nx, ny, nz, lx, ly, lz, write_out);
}
#endif

}  // namespace coulomb_detail

class CoulombPoissonSolver {
public:
    void configure(const GridConfig& gcfg) {
        gcfg_ = gcfg;
        const int nc = gcfg_.num_cells();
        if (nc <= 0) return;
        std::vector<double> hQx, hQy, hQz, hlx, hly, hlz;
        coulomb_detail::build_1d_laplacian_eigenbasis(gcfg_.nx, gcfg_.dx(), gcfg_.boundary, hQx, hlx);
        coulomb_detail::build_1d_laplacian_eigenbasis(gcfg_.ny, gcfg_.dy(), gcfg_.boundary, hQy, hly);
        coulomb_detail::build_1d_laplacian_eigenbasis(gcfg_.nz, gcfg_.dz(), gcfg_.boundary, hQz, hlz);
        Qx_.copy_from_host(hQx); Qy_.copy_from_host(hQy); Qz_.copy_from_host(hQz);
        lx_.copy_from_host(hlx); ly_.copy_from_host(hly); lz_.copy_from_host(hlz);
        buf_a_.resize(nc); buf_b_.resize(nc);
        mp10_.resize(10);
    }

    void solve(const DeviceBuffer<Real>& rho_q, DeviceBuffer<Real>& phi_raw_out) {
        const int nc = gcfg_.num_cells();
        if (nc <= 0) return;
        const int nx = gcfg_.nx, ny = gcfg_.ny, nz = gcfg_.nz;
        mp10_.zero();
#if TRANSPORT_HAS_CUDA
        constexpr int TPB = 256;
        const int blocks = (nc + TPB - 1) / TPB;
        if (gcfg_.boundary == BoundaryMode::OpenExpanding) {
            coulomb_detail::accum_multipoles_kernel<<<blocks, TPB>>>(rho_q.data(), nc, gcfg_, mp10_.data());
        }
        coulomb_detail::build_rhs_kernel<<<blocks, TPB>>>(rho_q.data(), buf_a_.data(), nc, gcfg_, mp10_.data());
        coulomb_detail::transform_axis_kernel<<<blocks, TPB>>>(buf_a_.data(), buf_b_.data(), Qx_.data(), nc, nx, ny, nz, 0, true);
        coulomb_detail::transform_axis_kernel<<<blocks, TPB>>>(buf_b_.data(), buf_a_.data(), Qy_.data(), nc, nx, ny, nz, 1, true);
        coulomb_detail::transform_axis_kernel<<<blocks, TPB>>>(buf_a_.data(), buf_b_.data(), Qz_.data(), nc, nx, ny, nz, 2, true);
        coulomb_detail::spectral_div_or_cast_kernel<<<blocks, TPB>>>(
            buf_b_.data(), nullptr, nc, nx, ny, nz, lx_.data(), ly_.data(), lz_.data(), false);
        coulomb_detail::transform_axis_kernel<<<blocks, TPB>>>(buf_b_.data(), buf_a_.data(), Qz_.data(), nc, nx, ny, nz, 2, false);
        coulomb_detail::transform_axis_kernel<<<blocks, TPB>>>(buf_a_.data(), buf_b_.data(), Qy_.data(), nc, nx, ny, nz, 1, false);
        coulomb_detail::transform_axis_kernel<<<blocks, TPB>>>(buf_b_.data(), buf_a_.data(), Qx_.data(), nc, nx, ny, nz, 0, false);
        coulomb_detail::spectral_div_or_cast_kernel<<<blocks, TPB>>>(
            buf_a_.data(), phi_raw_out.data(), nc, nx, ny, nz, nullptr, nullptr, nullptr, true);
        CUDA_CHECK(cudaGetLastError());
#else
        if (gcfg_.boundary == BoundaryMode::OpenExpanding) {
            for (int c = 0; c < nc; ++c) coulomb_detail::accum_cell_multipoles(rho_q.data(), c, gcfg_, mp10_.data());
        }
        for (int c = 0; c < nc; ++c) coulomb_detail::build_poisson_rhs_cell(rho_q.data(), buf_a_.data(), c, gcfg_, mp10_.data());
        for (int c = 0; c < nc; ++c) coulomb_detail::transform_axis_1d_cell(buf_a_.data(), buf_b_.data(), Qx_.data(), c, nx, ny, nz, 0, true);
        for (int c = 0; c < nc; ++c) coulomb_detail::transform_axis_1d_cell(buf_b_.data(), buf_a_.data(), Qy_.data(), c, nx, ny, nz, 1, true);
        for (int c = 0; c < nc; ++c) coulomb_detail::transform_axis_1d_cell(buf_a_.data(), buf_b_.data(), Qz_.data(), c, nx, ny, nz, 2, true);
        for (int c = 0; c < nc; ++c) {
            coulomb_detail::spectral_poisson_div_cell(buf_b_.data(), nullptr, c, nx, ny, nz, lx_.data(), ly_.data(), lz_.data(), false);
        }
        for (int c = 0; c < nc; ++c) coulomb_detail::transform_axis_1d_cell(buf_b_.data(), buf_a_.data(), Qz_.data(), c, nx, ny, nz, 2, false);
        for (int c = 0; c < nc; ++c) coulomb_detail::transform_axis_1d_cell(buf_a_.data(), buf_b_.data(), Qy_.data(), c, nx, ny, nz, 1, false);
        for (int c = 0; c < nc; ++c) coulomb_detail::transform_axis_1d_cell(buf_b_.data(), buf_a_.data(), Qx_.data(), c, nx, ny, nz, 0, false);
        for (int c = 0; c < nc; ++c) {
            coulomb_detail::spectral_poisson_div_cell(buf_a_.data(), phi_raw_out.data(), c, nx, ny, nz, nullptr, nullptr, nullptr, true);
        }
#endif
    }

private:
    GridConfig gcfg_;
    DeviceBuffer<double> Qx_, Qy_, Qz_, lx_, ly_, lz_, buf_a_, buf_b_, mp10_;
};

}  // namespace transport
