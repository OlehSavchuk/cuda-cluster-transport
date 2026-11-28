#pragma once

#include <cstddef>
#include <vector>
#include "backend/cuda_compat.cuh"
#include "physics/species.cuh"

namespace transport {

struct HostParticle {
    double t = 0.0;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double E = 0.0;
    double px = 0.0;
    double py = 0.0;
    double pz = 0.0;
    double mass = 0.0;
    int species = PROTON;
    int collided = 0;       // 0=none, 1=elastic, 2=inelastic 2->2, 3=formed 2->1, 4=decay 1->2, 5=freezeout
    int freezeout_tag = 0;  // 0=dense transport phase, 1=crossed n_B < n_fr freeze-out cell
};

// Lightweight POD Structure-of-Arrays view passed by value into CUDA __global__ kernels.
// Supports dynamic particle creation (1 -> 2 decays) via atomic d_n_slots and removal
// (2 -> 1 resonance formations) via active[i] = 0.
struct ParticleSoAView {
    int n = 0;                 // Current upper slot bound
    int capacity = 0;          // Allocated slot capacity
    int* d_n_slots = nullptr;  // Device-resident slot counter for atomic 1 -> 2 appends
    int* active = nullptr;     // 1 = live particle, 0 = absorbed slot
    Real* t = nullptr;
    Real* x = nullptr;
    Real* y = nullptr;
    Real* z = nullptr;
    Real* E = nullptr;
    Real* px = nullptr;
    Real* py = nullptr;
    Real* pz = nullptr;
    Real* mass = nullptr;
    int* species = nullptr;
    int* cell_id = nullptr;
    int* collided = nullptr;
    int* freezeout_tag = nullptr;
};

#if TRANSPORT_HAS_CUDA
namespace detail {
__global__ void count_active_slots_kernel(const int* __restrict__ active, int n, int* d_count) {
    int local = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        if (active[i]) ++local;
    }
    if (local > 0) atomicAdd(d_count, local);
}

__global__ void compact_particles_kernel(ParticleSoAView src, ParticleSoAView dst, int* d_write_idx) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= src.n) return;
    if (!src.active[i]) return;

    const int out = atomicAdd(d_write_idx, 1);
    dst.active[out] = 1;
    dst.t[out] = src.t[i];
    dst.x[out] = src.x[i];
    dst.y[out] = src.y[i];
    dst.z[out] = src.z[i];
    dst.E[out] = src.E[i];
    dst.px[out] = src.px[i];
    dst.py[out] = src.py[i];
    dst.pz[out] = src.pz[i];
    dst.mass[out] = src.mass[i];
    dst.species[out] = src.species[i];
    dst.cell_id[out] = src.cell_id[i];
    dst.collided[out] = src.collided[i];
    dst.freezeout_tag[out] = src.freezeout_tag[i];
}
}  // namespace detail
#endif

class ParticleSystem {
public:
    ParticleSystem() { d_n_slots_.resize(1); d_compact_count_.resize(1); }

    explicit ParticleSystem(std::size_t n) : ParticleSystem() { resize_with_headroom(n); }

    void resize_with_headroom(std::size_t n_live) {
        const std::size_t cap = (n_live * 3 / 2 + 16384 > capacity_) ? (n_live * 3 / 2 + 16384) : capacity_;
        allocate_capacity(cap);
        n_slots_ = static_cast<int>(n_live);
        ++mutation_epoch_;
        sync_slot_counter_to_device();
    }

    void ensure_headroom(int extra_needed) {
        sync_slot_counter_from_device();
        if (n_slots_ + extra_needed <= static_cast<int>(capacity_)) return;

#if TRANSPORT_HAS_CUDA
        const int live = count_active();
        const std::size_t needed = static_cast<std::size_t>(live + extra_needed) * 3 / 2 + 16384;
        const std::size_t new_cap = (needed > capacity_ * 3 / 2 + 16384) ? needed : (capacity_ * 3 / 2 + 16384);
        compact_on_device_to_capacity(new_cap);
#else
        auto live = download();
        const std::size_t new_cap = static_cast<std::size_t>(live.size() + extra_needed) * 3 / 2 + 16384;
        allocate_capacity(new_cap);
        upload(live);
#endif
    }

    void bump_mutation_epoch() { ++mutation_epoch_; }

    void append_host(const std::vector<HostParticle>& add) {
        const std::size_t m = add.size();
        ++mutation_epoch_;
        if (m == 0) return;
        ensure_headroom(static_cast<int>(m) + 64);
        sync_slot_counter_from_device();
        const std::size_t off = static_cast<std::size_t>(n_slots_);

        for (auto& v : stage_i_) v.resize(m);
        for (auto& v : stage_r_) v.resize(m);
        std::fill_n(stage_i_[0].data(), m, 1);
        std::fill_n(stage_i_[2].data(), m, -1);

        for (std::size_t i = 0; i < m; ++i) {
            stage_r_[0][i] = static_cast<Real>(add[i].t);
            stage_r_[1][i] = static_cast<Real>(add[i].x);
            stage_r_[2][i] = static_cast<Real>(add[i].y);
            stage_r_[3][i] = static_cast<Real>(add[i].z);
            stage_r_[4][i] = static_cast<Real>(add[i].E);
            stage_r_[5][i] = static_cast<Real>(add[i].px);
            stage_r_[6][i] = static_cast<Real>(add[i].py);
            stage_r_[7][i] = static_cast<Real>(add[i].pz);
            stage_r_[8][i] = static_cast<Real>(add[i].mass);
            stage_i_[1][i] = add[i].species;
            stage_i_[3][i] = add[i].collided;
            stage_i_[4][i] = add[i].freezeout_tag;
        }

        copy_offset_to_device(active_, stage_i_[0], off, m);
        copy_offset_to_device(t_, stage_r_[0], off, m);
        copy_offset_to_device(x_, stage_r_[1], off, m);
        copy_offset_to_device(y_, stage_r_[2], off, m);
        copy_offset_to_device(z_, stage_r_[3], off, m);
        copy_offset_to_device(E_, stage_r_[4], off, m);
        copy_offset_to_device(px_, stage_r_[5], off, m);
        copy_offset_to_device(py_, stage_r_[6], off, m);
        copy_offset_to_device(pz_, stage_r_[7], off, m);
        copy_offset_to_device(mass_, stage_r_[8], off, m);
        copy_offset_to_device(species_, stage_i_[1], off, m);
        copy_offset_to_device(cell_id_, stage_i_[2], off, m);
        copy_offset_to_device(collided_, stage_i_[3], off, m);
        copy_offset_to_device(freezeout_tag_, stage_i_[4], off, m);

        n_slots_ += static_cast<int>(m);
        sync_slot_counter_to_device();
    }

    void sync_slot_counter_from_device() {
        if (d_n_slots_.size() > 0) {
            d_n_slots_.copy_to_host(&n_slots_, 1);
        }
    }

    void sync_slot_counter_to_device() {
        d_n_slots_.copy_from_host(&n_slots_, 1);
    }

    void upload(const std::vector<HostParticle>& host_p) {
        const std::size_t n = host_p.size();
        const std::size_t needed_cap = n * 2 + 8192;
        if (needed_cap > capacity_) {
            allocate_capacity(needed_cap);
        }
        n_slots_ = static_cast<int>(n);
        ++mutation_epoch_;
        sync_slot_counter_to_device();

        if (n == 0) return;

        for (auto& v : stage_i_) v.resize(n);
        for (auto& v : stage_r_) v.resize(n);
        std::fill_n(stage_i_[0].data(), n, 1);
        std::fill_n(stage_i_[2].data(), n, -1);

        for (std::size_t i = 0; i < n; ++i) {
            stage_r_[0][i] = static_cast<Real>(host_p[i].t);
            stage_r_[1][i] = static_cast<Real>(host_p[i].x);
            stage_r_[2][i] = static_cast<Real>(host_p[i].y);
            stage_r_[3][i] = static_cast<Real>(host_p[i].z);
            stage_r_[4][i] = static_cast<Real>(host_p[i].E);
            stage_r_[5][i] = static_cast<Real>(host_p[i].px);
            stage_r_[6][i] = static_cast<Real>(host_p[i].py);
            stage_r_[7][i] = static_cast<Real>(host_p[i].pz);
            stage_r_[8][i] = static_cast<Real>(host_p[i].mass);
            stage_i_[1][i] = host_p[i].species;
            stage_i_[3][i] = host_p[i].collided;
            stage_i_[4][i] = host_p[i].freezeout_tag;
        }

        copy_prefix_to_device(active_, stage_i_[0], n);
        copy_prefix_to_device(t_, stage_r_[0], n);
        copy_prefix_to_device(x_, stage_r_[1], n);
        copy_prefix_to_device(y_, stage_r_[2], n);
        copy_prefix_to_device(z_, stage_r_[3], n);
        copy_prefix_to_device(E_, stage_r_[4], n);
        copy_prefix_to_device(px_, stage_r_[5], n);
        copy_prefix_to_device(py_, stage_r_[6], n);
        copy_prefix_to_device(pz_, stage_r_[7], n);
        copy_prefix_to_device(mass_, stage_r_[8], n);
        copy_prefix_to_device(species_, stage_i_[1], n);
        copy_prefix_to_device(cell_id_, stage_i_[2], n);
        copy_prefix_to_device(collided_, stage_i_[3], n);
        copy_prefix_to_device(freezeout_tag_, stage_i_[4], n);
    }

    // Download ONLY active particles
    std::vector<HostParticle> download() {
        sync_slot_counter_from_device();
        const std::size_t n = static_cast<std::size_t>(n_slots_);
        std::vector<HostParticle> out;
        if (n == 0) return out;

        for (auto& v : stage_i_) v.resize(n);
        for (auto& v : stage_r_) v.resize(n);

        active_.copy_to_host(stage_i_[0].data(), n);
        t_.copy_to_host(stage_r_[0].data(), n);
        x_.copy_to_host(stage_r_[1].data(), n);
        y_.copy_to_host(stage_r_[2].data(), n);
        z_.copy_to_host(stage_r_[3].data(), n);
        E_.copy_to_host(stage_r_[4].data(), n);
        px_.copy_to_host(stage_r_[5].data(), n);
        py_.copy_to_host(stage_r_[6].data(), n);
        pz_.copy_to_host(stage_r_[7].data(), n);
        mass_.copy_to_host(stage_r_[8].data(), n);
        species_.copy_to_host(stage_i_[1].data(), n);
        collided_.copy_to_host(stage_i_[3].data(), n);
        freezeout_tag_.copy_to_host(stage_i_[4].data(), n);

        out.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            if (!stage_i_[0][i]) continue;
            HostParticle hp;
            hp.t = stage_r_[0][i];
            hp.x = stage_r_[1][i];
            hp.y = stage_r_[2][i];
            hp.z = stage_r_[3][i];
            hp.E = stage_r_[4][i];
            hp.px = stage_r_[5][i];
            hp.py = stage_r_[6][i];
            hp.pz = stage_r_[7][i];
            hp.mass = stage_r_[8][i];
            hp.species = stage_i_[1][i];
            hp.collided = stage_i_[3][i];
            hp.freezeout_tag = stage_i_[4][i];
            out.push_back(hp);
        }
        return out;
    }

    int count_active() {
        sync_slot_counter_from_device();
        if (n_slots_ == 0) return 0;
#if TRANSPORT_HAS_CUDA
        d_compact_count_.zero();
        constexpr int TPB = 256;
        int blocks = (n_slots_ + TPB - 1) / TPB;
        if (blocks > 512) blocks = 512;
        detail::count_active_slots_kernel<<<blocks, TPB>>>(active_.data(), n_slots_, d_compact_count_.data());
        CUDA_CHECK(cudaGetLastError());
        int h_cnt = 0;
        d_compact_count_.copy_to_host(&h_cnt, 1);
        return h_cnt;
#else
        int cnt = 0;
        const int* act = active_.data();
        for (int i = 0; i < n_slots_; ++i) {
            if (act[i]) ++cnt;
        }
        return cnt;
#endif
    }

    // Compact inactive holes out of the SoA arrays when hole fraction exceeds threshold
    void compact_if_fragmented() {
        sync_slot_counter_from_device();
        if (n_slots_ < 256) return;
        const int live = count_active();
        if (live >= static_cast<int>(0.85 * n_slots_) &&
            n_slots_ + live + 1024 < static_cast<int>(capacity_)) {
            return;
        }
#if TRANSPORT_HAS_CUDA
        const std::size_t new_cap =
            (static_cast<std::size_t>(live) * 3 / 2 + 16384 > capacity_)
                ? (static_cast<std::size_t>(live) * 3 / 2 + 16384)
                : capacity_;
        compact_on_device_to_capacity(new_cap);
#else
        if (static_cast<std::size_t>(live) * 2 + 8192 > capacity_) {
            auto host_live = download();
            upload(host_live);
            return;
        }
        int dst = 0;
        int* act = active_.data();
        for (int src = 0; src < n_slots_; ++src) {
            if (!act[src]) continue;
            if (dst != src) {
                act[dst] = 1;
                t_.data()[dst] = t_.data()[src];
                x_.data()[dst] = x_.data()[src];
                y_.data()[dst] = y_.data()[src];
                z_.data()[dst] = z_.data()[src];
                E_.data()[dst] = E_.data()[src];
                px_.data()[dst] = px_.data()[src];
                py_.data()[dst] = py_.data()[src];
                pz_.data()[dst] = pz_.data()[src];
                mass_.data()[dst] = mass_.data()[src];
                species_.data()[dst] = species_.data()[src];
                cell_id_.data()[dst] = cell_id_.data()[src];
                collided_.data()[dst] = collided_.data()[src];
                freezeout_tag_.data()[dst] = freezeout_tag_.data()[src];
            }
            ++dst;
        }
        n_slots_ = dst;
        sync_slot_counter_to_device();
#endif
    }

    ParticleSoAView view() {
        sync_slot_counter_from_device();
        ParticleSoAView v;
        v.n = n_slots_;
        v.capacity = static_cast<int>(capacity_);
        v.d_n_slots = d_n_slots_.data();
        v.active = active_.data();
        v.t = t_.data();
        v.x = x_.data();
        v.y = y_.data();
        v.z = z_.data();
        v.E = E_.data();
        v.px = px_.data();
        v.py = py_.data();
        v.pz = pz_.data();
        v.mass = mass_.data();
        v.species = species_.data();
        v.cell_id = cell_id_.data();
        v.collided = collided_.data();
        v.freezeout_tag = freezeout_tag_.data();
        return v;
    }

    int slot_size() {
        sync_slot_counter_from_device();
        return n_slots_;
    }

    std::uint64_t mutation_epoch() const { return mutation_epoch_; }

    std::size_t capacity() const { return capacity_; }

private:
    void allocate_capacity(std::size_t cap) {
        capacity_ = cap;
        active_.resize(cap);
        active_.zero();
        t_.resize(cap);
        x_.resize(cap);
        y_.resize(cap);
        z_.resize(cap);
        E_.resize(cap);
        px_.resize(cap);
        py_.resize(cap);
        pz_.resize(cap);
        mass_.resize(cap);
        species_.resize(cap);
        cell_id_.resize(cap);
        collided_.resize(cap);
        freezeout_tag_.resize(cap);
        freezeout_tag_.zero();
    }

    template <typename T>
    static void copy_prefix_to_device(DeviceBuffer<T>& dst, const std::vector<T>& src, std::size_t n) {
        copy_offset_to_device(dst, src, 0, n);
    }

    template <typename T>
    static void copy_offset_to_device(DeviceBuffer<T>& dst, const std::vector<T>& src, std::size_t off, std::size_t n) {
        if (n == 0) return;
#if TRANSPORT_HAS_CUDA
        CUDA_CHECK(cudaMemcpy(dst.data() + off, src.data(), n * sizeof(T), cudaMemcpyHostToDevice));
#else
        std::memcpy(dst.data() + off, src.data(), n * sizeof(T));
#endif
    }

#if TRANSPORT_HAS_CUDA
    void compact_on_device_to_capacity(std::size_t new_cap) {
        DeviceBuffer<int> n_act(new_cap), n_sp(new_cap), n_cid(new_cap), n_col(new_cap), n_fr(new_cap);
        n_act.zero();
        n_fr.zero();
        DeviceBuffer<Real> n_t(new_cap), n_x(new_cap), n_y(new_cap), n_z(new_cap);
        DeviceBuffer<Real> n_E(new_cap), n_px(new_cap), n_py(new_cap), n_pz(new_cap), n_m(new_cap);

        if (n_slots_ > 0) {
            ParticleSoAView dst{};
            dst.capacity = static_cast<int>(new_cap);
            dst.active = n_act.data();
            dst.t = n_t.data(); dst.x = n_x.data(); dst.y = n_y.data(); dst.z = n_z.data();
            dst.E = n_E.data(); dst.px = n_px.data(); dst.py = n_py.data(); dst.pz = n_pz.data();
            dst.mass = n_m.data(); dst.species = n_sp.data();
            dst.cell_id = n_cid.data(); dst.collided = n_col.data();
            dst.freezeout_tag = n_fr.data();

            d_compact_count_.zero();
            constexpr int TPB = 256;
            const int blocks = (n_slots_ + TPB - 1) / TPB;
            detail::compact_particles_kernel<<<blocks, TPB>>>(view(), dst, d_compact_count_.data());
            CUDA_CHECK(cudaGetLastError());
            d_compact_count_.copy_to_host(&n_slots_, 1);
        } else {
            n_slots_ = 0;
        }

        ++mutation_epoch_;
        sync_slot_counter_to_device();
        capacity_ = new_cap;

        active_ = std::move(n_act);
        t_ = std::move(n_t); x_ = std::move(n_x); y_ = std::move(n_y); z_ = std::move(n_z);
        E_ = std::move(n_E); px_ = std::move(n_px); py_ = std::move(n_py); pz_ = std::move(n_pz);
        mass_ = std::move(n_m); species_ = std::move(n_sp);
        cell_id_ = std::move(n_cid); collided_ = std::move(n_col);
        freezeout_tag_ = std::move(n_fr);
    }
#endif

    int n_slots_ = 0;
    std::size_t capacity_ = 0;
    std::uint64_t mutation_epoch_ = 1;
    DeviceBuffer<int> d_n_slots_;
    DeviceBuffer<int> d_compact_count_;
    DeviceBuffer<int> active_;
    DeviceBuffer<Real> t_, x_, y_, z_;
    DeviceBuffer<Real> E_, px_, py_, pz_, mass_;
    DeviceBuffer<int> species_, cell_id_, collided_, freezeout_tag_;
    std::vector<int> stage_i_[5];
    std::vector<Real> stage_r_[9];
};

}  // namespace transport
