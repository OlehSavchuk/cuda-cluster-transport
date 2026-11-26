#pragma once

#include <cmath>
#include "backend/cuda_compat.cuh"
#include "physics/constants.cuh"

namespace transport {

struct FourVector {
    Real t = 0;  // E or t
    Real x = 0;  // px or x
    Real y = 0;  // py or y
    Real z = 0;  // pz or z

    HD_INLINE constexpr FourVector() = default;
    HD_INLINE constexpr FourVector(double t_, double x_, double y_, double z_)
        : t(static_cast<Real>(t_)), x(static_cast<Real>(x_)),
          y(static_cast<Real>(y_)), z(static_cast<Real>(z_)) {}
};

struct PairKinematics {
    Real m1;
    Real m2;
    Real s;           // Mandelstam s = (p1 + p2)^2 [GeV^2]
    Real sqrt_s;      // sqrt(s) [GeV]
    Real p_cm;        // CM momentum magnitude [GeV/c]
    Real v_moller;    // Relativistic Moller velocity sqrt((p1.p2)^2 - m1^2 m2^2)/(E1 E2)
    Real gamma_rel;   // Lorentz-invariant relative gamma = (p1.p2)/(m1 m2)
};

// Cancellation-free relativistic pair kinematics for arbitrary masses m1, m2 > 0.
// Uses the exact relativistic identity:
//   Q^2 = (p1.p2)^2 - m1^2 m2^2 = |E2*p1 - E1*p2|^2 - |p1 x p2|^2
// avoiding catastrophic cancellation near threshold s -> (m1 + m2)^2 even for heavy clusters.
HD_INLINE PairKinematics compute_pair_kinematics(const FourVector& p1, Real m1,
                                                 const FourVector& p2, Real m2) {
    PairKinematics kin{};
    kin.m1 = m1;
    kin.m2 = m2;

    const Real dx = p2.t * p1.x - p1.t * p2.x;
    const Real dy = p2.t * p1.y - p1.t * p2.y;
    const Real dz = p2.t * p1.z - p1.t * p2.z;
    const Real diff_sq = dx * dx + dy * dy + dz * dz;

    const Real cx = p1.y * p2.z - p1.z * p2.y;
    const Real cy = p1.z * p2.x - p1.x * p2.z;
    const Real cz = p1.x * p2.y - p1.y * p2.x;
    const Real cross_sq = cx * cx + cy * cy + cz * cz;

    const Real q_sq = (diff_sq > cross_sq) ? (diff_sq - cross_sq) : Real(0);
    const Real q = std::sqrt(q_sq);

    const Real m1m2 = m1 * m2;
    const Real p1_dot_p2 = std::sqrt(m1m2 * m1m2 + q_sq);

    kin.s = m1 * m1 + m2 * m2 + Real(2) * p1_dot_p2;
    kin.sqrt_s = std::sqrt(kin.s);
    kin.p_cm = (kin.sqrt_s > Real(0)) ? (q / kin.sqrt_s) : Real(0);
    kin.v_moller = (p1.t * p2.t > Real(0)) ? (q / (p1.t * p2.t)) : Real(0);
    kin.gamma_rel = (m1m2 > Real(0)) ? (p1_dot_p2 / m1m2) : Real(1);
    return kin;
}

// Boost spatial momentum p from lab frame into the rest frame moving with velocity beta = P/E_tot
HD_INLINE void lorentz_boost_to_cm(Real E, Real px, Real py, Real pz,
                                   Real bx, Real by, Real bz, Real gamma,
                                   Real& px_cm, Real& py_cm, Real& pz_cm) {
    const Real b_dot_p = bx * px + by * py + bz * pz;
    const Real factor = (gamma * gamma / (gamma + Real(1))) * b_dot_p - gamma * E;
    px_cm = px + bx * factor;
    py_cm = py + by * factor;
    pz_cm = pz + bz * factor;
}

// Boost 4-momentum (E*, p*) from CM frame back to lab frame with velocity +beta
HD_INLINE FourVector lorentz_boost_from_cm(Real m, Real E_cm,
                                           Real px_cm, Real py_cm, Real pz_cm,
                                           Real bx, Real by, Real bz, Real gamma) {
    const Real b_dot_p = bx * px_cm + by * py_cm + bz * pz_cm;
    const Real factor = (gamma * gamma / (gamma + Real(1))) * b_dot_p + gamma * E_cm;
    FourVector out;
    out.x = px_cm + bx * factor;
    out.y = py_cm + by * factor;
    out.z = pz_cm + bz * factor;
    out.t = std::sqrt(m * m + out.x * out.x + out.y * out.y + out.z * out.z);
    return out;
}

// Rotate a CM momentum vector of magnitude p_cm by polar angle cos_theta and azimuth phi
// relative to the incoming CM unit direction n = (nx, ny, nz).
HD_INLINE void rotate_from_axis(Real nx, Real ny, Real nz,
                                Real p_cm, Real cos_theta, Real phi,
                                Real& ox, Real& oy, Real& oz) {
    const Real sin_theta = std::sqrt(
        (Real(1) - cos_theta * cos_theta > Real(0)) ? (Real(1) - cos_theta * cos_theta) : Real(0));
    const Real cos_phi = std::cos(phi);
    const Real sin_phi = std::sin(phi);

    Real e1x, e1y, e1z;
    if (std::fabs(nz) < Real(0.9)) {
        const Real inv_perp = Real(1) / std::sqrt(nx * nx + ny * ny);
        e1x = -ny * inv_perp;
        e1y =  nx * inv_perp;
        e1z =  Real(0);
    } else {
        const Real inv_perp = Real(1) / std::sqrt(ny * ny + nz * nz);
        e1x =  Real(0);
        e1y = -nz * inv_perp;
        e1z =  ny * inv_perp;
    }
    const Real e2x = ny * e1z - nz * e1y;
    const Real e2y = nz * e1x - nx * e1z;
    const Real e2z = nx * e1y - ny * e1x;

    const Real st_cp = sin_theta * cos_phi;
    const Real st_sp = sin_theta * sin_phi;

    ox = p_cm * (st_cp * e1x + st_sp * e2x + cos_theta * nx);
    oy = p_cm * (st_cp * e1y + st_sp * e2y + cos_theta * ny);
    oz = p_cm * (st_cp * e1z + st_sp * e2z + cos_theta * nz);
}

// Execute a full relativistic 2 -> 2 elastic scattering in the pair CM frame
// given sampled CM scattering angles (cos_theta, phi), returning exact 4-momenta.
HD_INLINE void scatter_elastic_2to2(const FourVector& p1_in, Real m1,
                                    const FourVector& p2_in, Real m2,
                                    const PairKinematics& kin,
                                    Real cos_theta, Real phi,
                                    FourVector& p1_out, FourVector& p2_out) {
    const Real E_tot = p1_in.t + p2_in.t;
    const Real Px_tot = p1_in.x + p2_in.x;
    const Real Py_tot = p1_in.y + p2_in.y;
    const Real Pz_tot = p1_in.z + p2_in.z;

    const Real inv_E = Real(1) / E_tot;
    const Real bx = Px_tot * inv_E;
    const Real by = Py_tot * inv_E;
    const Real bz = Pz_tot * inv_E;
    const Real gamma = E_tot / kin.sqrt_s;

    Real p1x_cm, p1y_cm, p1z_cm;
    lorentz_boost_to_cm(p1_in.t, p1_in.x, p1_in.y, p1_in.z, bx, by, bz, gamma,
                        p1x_cm, p1y_cm, p1z_cm);

    const Real p1_norm = std::sqrt(p1x_cm * p1x_cm + p1y_cm * p1y_cm + p1z_cm * p1z_cm);
    Real nx = Real(0), ny = Real(0), nz = Real(1);
    if (p1_norm > Real(1.0e-12)) {
        const Real inv_norm = Real(1) / p1_norm;
        nx = p1x_cm * inv_norm;
        ny = p1y_cm * inv_norm;
        nz = p1z_cm * inv_norm;
    }

    Real qx_cm, qy_cm, qz_cm;
    rotate_from_axis(nx, ny, nz, kin.p_cm, cos_theta, phi, qx_cm, qy_cm, qz_cm);

    const Real E1_cm = std::sqrt(m1 * m1 + kin.p_cm * kin.p_cm);
    const Real E2_cm = std::sqrt(m2 * m2 + kin.p_cm * kin.p_cm);

    p1_out = lorentz_boost_from_cm(m1, E1_cm,  qx_cm,  qy_cm,  qz_cm, bx, by, bz, gamma);
    p2_out = lorentz_boost_from_cm(m2, E2_cm, -qx_cm, -qy_cm, -qz_cm, bx, by, bz, gamma);
}

// Exact CM breakup momentum p_cm(sqrt_s; ma, mb) for sqrt_s > ma + mb (0 otherwise)
HD_INLINE Real two_body_p_cm(Real sqrt_s, Real ma, Real mb) {
    const Real m_sum = ma + mb;
    if (sqrt_s <= m_sum) return Real(0);
    const Real m_diff = std::fabs(ma - mb);
    const Real lam = (sqrt_s - m_sum) * (sqrt_s + m_sum) *
                     (sqrt_s - m_diff) * (sqrt_s + m_diff);
    return (lam > Real(0)) ? (Real(0.5) * std::sqrt(lam) / sqrt_s) : Real(0);
}

// General 2 -> 2 scattering (elastic or inelastic) from (p1_in, m1) + (p2_in, m2)
// into outgoing masses (m3, m4) at CM polar angle (cos_theta, phi) relative to incoming p1*.
HD_INLINE void scatter_general_2to2(const FourVector& p1_in,
                                    const FourVector& p2_in,
                                    Real sqrt_s,
                                    Real m3, Real m4,
                                    Real p_cm_out,
                                    Real cos_theta, Real phi,
                                    FourVector& p3_out, FourVector& p4_out) {
    const Real E_tot = p1_in.t + p2_in.t;
    const Real Px_tot = p1_in.x + p2_in.x;
    const Real Py_tot = p1_in.y + p2_in.y;
    const Real Pz_tot = p1_in.z + p2_in.z;

    const Real inv_E = Real(1) / E_tot;
    const Real bx = Px_tot * inv_E;
    const Real by = Py_tot * inv_E;
    const Real bz = Pz_tot * inv_E;
    const Real gamma = E_tot / sqrt_s;

    Real p1x_cm, p1y_cm, p1z_cm;
    lorentz_boost_to_cm(p1_in.t, p1_in.x, p1_in.y, p1_in.z, bx, by, bz, gamma,
                        p1x_cm, p1y_cm, p1z_cm);

    const Real p1_norm = std::sqrt(p1x_cm * p1x_cm + p1y_cm * p1y_cm + p1z_cm * p1z_cm);
    Real nx = Real(0), ny = Real(0), nz = Real(1);
    if (p1_norm > Real(1.0e-12)) {
        const Real inv_norm = Real(1) / p1_norm;
        nx = p1x_cm * inv_norm;
        ny = p1y_cm * inv_norm;
        nz = p1z_cm * inv_norm;
    }

    Real qx_cm, qy_cm, qz_cm;
    rotate_from_axis(nx, ny, nz, p_cm_out, cos_theta, phi, qx_cm, qy_cm, qz_cm);

    const Real E3_cm = std::sqrt(m3 * m3 + p_cm_out * p_cm_out);
    const Real E4_cm = std::sqrt(m4 * m4 + p_cm_out * p_cm_out);

    p3_out = lorentz_boost_from_cm(m3, E3_cm,  qx_cm,  qy_cm,  qz_cm, bx, by, bz, gamma);
    p4_out = lorentz_boost_from_cm(m4, E4_cm, -qx_cm, -qy_cm, -qz_cm, bx, by, bz, gamma);
}

// Isotropic 1 -> 2 resonance decay R(p_res, m_res) -> a(ma) + b(mb)
HD_INLINE void decay_isotropic_1to2(const FourVector& p_res, Real m_res,
                                    Real ma, Real mb,
                                    Real cos_theta, Real phi,
                                    FourVector& pa_out, FourVector& pb_out) {
    const Real p_cm = two_body_p_cm(m_res, ma, mb);
    const Real sin_th = std::sqrt(
        (Real(1) - cos_theta * cos_theta > Real(0)) ? (Real(1) - cos_theta * cos_theta) : Real(0));
    const Real qx = p_cm * sin_th * std::cos(phi);
    const Real qy = p_cm * sin_th * std::sin(phi);
    const Real qz = p_cm * cos_theta;

    const Real Ea_cm = std::sqrt(ma * ma + p_cm * p_cm);
    const Real Eb_cm = std::sqrt(mb * mb + p_cm * p_cm);

    const Real inv_E = Real(1) / p_res.t;
    const Real bx = p_res.x * inv_E;
    const Real by = p_res.y * inv_E;
    const Real bz = p_res.z * inv_E;
    const Real gamma = p_res.t / m_res;

    pa_out = lorentz_boost_from_cm(ma, Ea_cm,  qx,  qy,  qz, bx, by, bz, gamma);
    pb_out = lorentz_boost_from_cm(mb, Eb_cm, -qx, -qy, -qz, bx, by, bz, gamma);
}

}  // namespace transport
