#pragma once

#include <cmath>
#include "backend/cuda_compat.cuh"
#include "physics/constants.cuh"

namespace transport {

struct FourVector {
    double t;  // E or t
    double x;  // px or x
    double y;  // py or y
    double z;  // pz or z
};

struct PairKinematics {
    double m1;
    double m2;
    double s;           // Mandelstam s = (p1 + p2)^2 [GeV^2]
    double sqrt_s;      // sqrt(s) [GeV]
    double p_cm;        // CM momentum magnitude [GeV/c]
    double v_moller;    // Relativistic Moller velocity sqrt((p1.p2)^2 - m1^2 m2^2)/(E1 E2)
    double gamma_rel;   // Lorentz-invariant relative gamma = (p1.p2)/(m1 m2)
};

// Cancellation-free relativistic pair kinematics for arbitrary masses m1, m2 > 0.
// Uses the exact relativistic identity:
//   Q^2 = (p1.p2)^2 - m1^2 m2^2 = |E2*p1 - E1*p2|^2 - |p1 x p2|^2
// avoiding catastrophic cancellation near threshold s -> (m1 + m2)^2 even for heavy clusters.
HD_INLINE PairKinematics compute_pair_kinematics(const FourVector& p1, double m1,
                                                 const FourVector& p2, double m2) {
    PairKinematics kin{};
    kin.m1 = m1;
    kin.m2 = m2;

    const double dx = p2.t * p1.x - p1.t * p2.x;
    const double dy = p2.t * p1.y - p1.t * p2.y;
    const double dz = p2.t * p1.z - p1.t * p2.z;
    const double diff_sq = dx * dx + dy * dy + dz * dz;

    const double cx = p1.y * p2.z - p1.z * p2.y;
    const double cy = p1.z * p2.x - p1.x * p2.z;
    const double cz = p1.x * p2.y - p1.y * p2.x;
    const double cross_sq = cx * cx + cy * cy + cz * cz;

    const double q_sq = (diff_sq > cross_sq) ? (diff_sq - cross_sq) : 0.0;
    const double q = ::sqrt(q_sq);

    const double m1m2 = m1 * m2;
    const double p1_dot_p2 = ::sqrt(m1m2 * m1m2 + q_sq);

    kin.s = m1 * m1 + m2 * m2 + 2.0 * p1_dot_p2;
    kin.sqrt_s = ::sqrt(kin.s);
    kin.p_cm = (kin.sqrt_s > 0.0) ? (q / kin.sqrt_s) : 0.0;
    kin.v_moller = (p1.t * p2.t > 0.0) ? (q / (p1.t * p2.t)) : 0.0;
    kin.gamma_rel = (m1m2 > 0.0) ? (p1_dot_p2 / m1m2) : 1.0;
    return kin;
}

// Boost spatial momentum p from lab frame into the rest frame moving with velocity beta = P/E_tot
HD_INLINE void lorentz_boost_to_cm(double E, double px, double py, double pz,
                                   double bx, double by, double bz, double gamma,
                                   double& px_cm, double& py_cm, double& pz_cm) {
    const double b_dot_p = bx * px + by * py + bz * pz;
    const double factor = (gamma * gamma / (gamma + 1.0)) * b_dot_p - gamma * E;
    px_cm = px + bx * factor;
    py_cm = py + by * factor;
    pz_cm = pz + bz * factor;
}

// Boost 4-momentum (E*, p*) from CM frame back to lab frame with velocity +beta
HD_INLINE FourVector lorentz_boost_from_cm(double m, double E_cm,
                                           double px_cm, double py_cm, double pz_cm,
                                           double bx, double by, double bz, double gamma) {
    const double b_dot_p = bx * px_cm + by * py_cm + bz * pz_cm;
    const double factor = (gamma * gamma / (gamma + 1.0)) * b_dot_p + gamma * E_cm;
    FourVector out;
    out.x = px_cm + bx * factor;
    out.y = py_cm + by * factor;
    out.z = pz_cm + bz * factor;
    // Reconstruct on-shell energy to prevent floating-point mass drift over many collisions
    out.t = ::sqrt(m * m + out.x * out.x + out.y * out.y + out.z * out.z);
    return out;
}

// Rotate a CM momentum vector of magnitude p_cm by polar angle cos_theta and azimuth phi
// relative to the incoming CM unit direction n = (nx, ny, nz).
HD_INLINE void rotate_from_axis(double nx, double ny, double nz,
                                double p_cm, double cos_theta, double phi,
                                double& ox, double& oy, double& oz) {
    const double sin_theta = ::sqrt(
        (1.0 - cos_theta * cos_theta > 0.0) ? (1.0 - cos_theta * cos_theta) : 0.0);
    const double cos_phi = ::cos(phi);
    const double sin_phi = ::sin(phi);

    // Construct orthonormal basis (e1, e2, n) robustly without branch singularity at nz = +-1
    double e1x, e1y, e1z;
    if ( ::fabs(nz) < 0.999999 ) {
        const double inv_perp = 1.0 / ::sqrt(nx * nx + ny * ny);
        e1x = -ny * inv_perp;
        e1y =  nx * inv_perp;
        e1z =  0.0;
    } else {
        e1x = 1.0;
        e1y = 0.0;
        e1z = 0.0;
    }
    const double e2x = ny * e1z - nz * e1y;
    const double e2y = nz * e1x - nx * e1z;
    const double e2z = nx * e1y - ny * e1x;

    const double st_cp = sin_theta * cos_phi;
    const double st_sp = sin_theta * sin_phi;

    ox = p_cm * (st_cp * e1x + st_sp * e2x + cos_theta * nx);
    oy = p_cm * (st_cp * e1y + st_sp * e2y + cos_theta * ny);
    oz = p_cm * (st_cp * e1z + st_sp * e2z + cos_theta * nz);
}

// Execute a full relativistic 2 -> 2 elastic scattering in the pair CM frame
// given sampled CM scattering angles (cos_theta, phi), returning exact 4-momenta.
HD_INLINE void scatter_elastic_2to2(const FourVector& p1_in, double m1,
                                    const FourVector& p2_in, double m2,
                                    const PairKinematics& kin,
                                    double cos_theta, double phi,
                                    FourVector& p1_out, FourVector& p2_out) {
    const double E_tot = p1_in.t + p2_in.t;
    const double Px_tot = p1_in.x + p2_in.x;
    const double Py_tot = p1_in.y + p2_in.y;
    const double Pz_tot = p1_in.z + p2_in.z;

    const double inv_E = 1.0 / E_tot;
    const double bx = Px_tot * inv_E;
    const double by = Py_tot * inv_E;
    const double bz = Pz_tot * inv_E;
    const double gamma = E_tot / kin.sqrt_s;

    // Incoming momentum of particle 1 in CM frame defines polar z*-axis
    double p1x_cm, p1y_cm, p1z_cm;
    lorentz_boost_to_cm(p1_in.t, p1_in.x, p1_in.y, p1_in.z, bx, by, bz, gamma,
                        p1x_cm, p1y_cm, p1z_cm);

    const double p1_norm = ::sqrt(p1x_cm * p1x_cm + p1y_cm * p1y_cm + p1z_cm * p1z_cm);
    double nx = 0.0, ny = 0.0, nz = 1.0;
    if (p1_norm > 1.0e-15) {
        const double inv_norm = 1.0 / p1_norm;
        nx = p1x_cm * inv_norm;
        ny = p1y_cm * inv_norm;
        nz = p1z_cm * inv_norm;
    }

    double qx_cm, qy_cm, qz_cm;
    rotate_from_axis(nx, ny, nz, kin.p_cm, cos_theta, phi, qx_cm, qy_cm, qz_cm);

    const double E1_cm = ::sqrt(m1 * m1 + kin.p_cm * kin.p_cm);
    const double E2_cm = ::sqrt(m2 * m2 + kin.p_cm * kin.p_cm);

    p1_out = lorentz_boost_from_cm(m1, E1_cm,  qx_cm,  qy_cm,  qz_cm, bx, by, bz, gamma);
    p2_out = lorentz_boost_from_cm(m2, E2_cm, -qx_cm, -qy_cm, -qz_cm, bx, by, bz, gamma);
}

// Exact CM breakup momentum p_cm(sqrt_s; ma, mb) for sqrt_s > ma + mb (0 otherwise)
HD_INLINE double two_body_p_cm(double sqrt_s, double ma, double mb) {
    const double m_sum = ma + mb;
    if (sqrt_s <= m_sum) return 0.0;
    const double m_diff = ::fabs(ma - mb);
    const double lam = (sqrt_s - m_sum) * (sqrt_s + m_sum) *
                       (sqrt_s - m_diff) * (sqrt_s + m_diff);
    return (lam > 0.0) ? (0.5 * ::sqrt(lam) / sqrt_s) : 0.0;
}

// General 2 -> 2 scattering (elastic or inelastic) from (p1_in, m1) + (p2_in, m2)
// into outgoing masses (m3, m4) at CM polar angle (cos_theta, phi) relative to incoming p1*.
HD_INLINE void scatter_general_2to2(const FourVector& p1_in,
                                    const FourVector& p2_in,
                                    double sqrt_s,
                                    double m3, double m4,
                                    double p_cm_out,
                                    double cos_theta, double phi,
                                    FourVector& p3_out, FourVector& p4_out) {
    const double E_tot = p1_in.t + p2_in.t;
    const double Px_tot = p1_in.x + p2_in.x;
    const double Py_tot = p1_in.y + p2_in.y;
    const double Pz_tot = p1_in.z + p2_in.z;

    const double inv_E = 1.0 / E_tot;
    const double bx = Px_tot * inv_E;
    const double by = Py_tot * inv_E;
    const double bz = Pz_tot * inv_E;
    const double gamma = E_tot / sqrt_s;

    double p1x_cm, p1y_cm, p1z_cm;
    lorentz_boost_to_cm(p1_in.t, p1_in.x, p1_in.y, p1_in.z, bx, by, bz, gamma,
                        p1x_cm, p1y_cm, p1z_cm);

    const double p1_norm = ::sqrt(p1x_cm * p1x_cm + p1y_cm * p1y_cm + p1z_cm * p1z_cm);
    double nx = 0.0, ny = 0.0, nz = 1.0;
    if (p1_norm > 1.0e-15) {
        const double inv_norm = 1.0 / p1_norm;
        nx = p1x_cm * inv_norm;
        ny = p1y_cm * inv_norm;
        nz = p1z_cm * inv_norm;
    }

    double qx_cm, qy_cm, qz_cm;
    rotate_from_axis(nx, ny, nz, p_cm_out, cos_theta, phi, qx_cm, qy_cm, qz_cm);

    const double E3_cm = ::sqrt(m3 * m3 + p_cm_out * p_cm_out);
    const double E4_cm = ::sqrt(m4 * m4 + p_cm_out * p_cm_out);

    p3_out = lorentz_boost_from_cm(m3, E3_cm,  qx_cm,  qy_cm,  qz_cm, bx, by, bz, gamma);
    p4_out = lorentz_boost_from_cm(m4, E4_cm, -qx_cm, -qy_cm, -qz_cm, bx, by, bz, gamma);
}

// Isotropic 1 -> 2 resonance decay R(p_res, m_res) -> a(ma) + b(mb)
HD_INLINE void decay_isotropic_1to2(const FourVector& p_res, double m_res,
                                    double ma, double mb,
                                    double cos_theta, double phi,
                                    FourVector& pa_out, FourVector& pb_out) {
    const double p_cm = two_body_p_cm(m_res, ma, mb);
    const double sin_th = ::sqrt(
        (1.0 - cos_theta * cos_theta > 0.0) ? (1.0 - cos_theta * cos_theta) : 0.0);
    const double qx = p_cm * sin_th * ::cos(phi);
    const double qy = p_cm * sin_th * ::sin(phi);
    const double qz = p_cm * cos_theta;

    const double Ea_cm = ::sqrt(ma * ma + p_cm * p_cm);
    const double Eb_cm = ::sqrt(mb * mb + p_cm * p_cm);

    const double inv_E = 1.0 / p_res.t;
    const double bx = p_res.x * inv_E;
    const double by = p_res.y * inv_E;
    const double bz = p_res.z * inv_E;
    const double gamma = p_res.t / m_res;

    pa_out = lorentz_boost_from_cm(ma, Ea_cm,  qx,  qy,  qz, bx, by, bz, gamma);
    pb_out = lorentz_boost_from_cm(mb, Eb_cm, -qx, -qy, -qz, bx, by, bz, gamma);
}

}  // namespace transport
