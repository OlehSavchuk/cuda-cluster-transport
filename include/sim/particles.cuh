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
    int collided = 0;
};

// Lightweight POD Structure-of-Arrays view passed by value into CUDA __global__ kernels
// for fully coalesced GPU memory transactions.
struct ParticleSoAView {
    int n = 0;
    double* t = nullptr;
    double* x = nullptr;
    double* y = nullptr;
    double* z = nullptr;
    double* E = nullptr;
    double* px = nullptr;
    double* py = nullptr;
    double* pz = nullptr;
    double* mass = nullptr;
    int* species = nullptr;
    int* cell_id = nullptr;
    int* collided = nullptr;
};

class ParticleSystem {
public:
    ParticleSystem() = default;

    explicit ParticleSystem(std::size_t n) { resize(n); }

    void resize(std::size_t n) {
        n_ = static_cast<int>(n);
        t_.resize(n);
        x_.resize(n);
        y_.resize(n);
        z_.resize(n);
        E_.resize(n);
        px_.resize(n);
        py_.resize(n);
        pz_.resize(n);
        mass_.resize(n);
        species_.resize(n);
        cell_id_.resize(n);
        collided_.resize(n);
    }

    void upload(const std::vector<HostParticle>& host_p) {
        const std::size_t n = host_p.size();
        resize(n);
        std::vector<double> ht(n), hx(n), hy(n), hz(n);
        std::vector<double> hE(n), hpx(n), hpy(n), hpz(n), hmass(n);
        std::vector<int> hsp(n), hcell(n, -1), hcol(n, 0);

        for (std::size_t i = 0; i < n; ++i) {
            ht[i] = host_p[i].t;
            hx[i] = host_p[i].x;
            hy[i] = host_p[i].y;
            hz[i] = host_p[i].z;
            hE[i] = host_p[i].E;
            hpx[i] = host_p[i].px;
            hpy[i] = host_p[i].py;
            hpz[i] = host_p[i].pz;
            hmass[i] = host_p[i].mass;
            hsp[i] = host_p[i].species;
        }

        t_.copy_from_host(ht);
        x_.copy_from_host(hx);
        y_.copy_from_host(hy);
        z_.copy_from_host(hz);
        E_.copy_from_host(hE);
        px_.copy_from_host(hpx);
        py_.copy_from_host(hpy);
        pz_.copy_from_host(hpz);
        mass_.copy_from_host(hmass);
        species_.copy_from_host(hsp);
        cell_id_.copy_from_host(hcell);
        collided_.copy_from_host(hcol);
    }

    std::vector<HostParticle> download() const {
        const std::size_t n = static_cast<std::size_t>(n_);
        std::vector<HostParticle> out(n);
        if (n == 0) return out;

        auto ht = t_.to_vector();
        auto hx = x_.to_vector();
        auto hy = y_.to_vector();
        auto hz = z_.to_vector();
        auto hE = E_.to_vector();
        auto hpx = px_.to_vector();
        auto hpy = py_.to_vector();
        auto hpz = pz_.to_vector();
        auto hmass = mass_.to_vector();
        auto hsp = species_.to_vector();
        auto hcol = collided_.to_vector();

        for (std::size_t i = 0; i < n; ++i) {
            out[i].t = ht[i];
            out[i].x = hx[i];
            out[i].y = hy[i];
            out[i].z = hz[i];
            out[i].E = hE[i];
            out[i].px = hpx[i];
            out[i].py = hpy[i];
            out[i].pz = hpz[i];
            out[i].mass = hmass[i];
            out[i].species = hsp[i];
            out[i].collided = hcol[i];
        }
        return out;
    }

    ParticleSoAView view() {
        ParticleSoAView v;
        v.n = n_;
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
        return v;
    }

    int size() const { return n_; }

private:
    int n_ = 0;
    DeviceBuffer<double> t_, x_, y_, z_;
    DeviceBuffer<double> E_, px_, py_, pz_, mass_;
    DeviceBuffer<int> species_, cell_id_, collided_;
};

}  // namespace transport
