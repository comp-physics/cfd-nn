/// @file ibm_geometry.cpp
/// @brief Implementation of IBM geometry signed distance functions

#include "ibm_geometry.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace nncfd {

// ============================================================================
// IBMBody base class — default normal via finite-difference gradient of phi
// ============================================================================

std::tuple<double, double, double> IBMBody::normal(double x, double y, double z) const {
    const double eps = 1e-8;
    double dx = phi(x + eps, y, z) - phi(x - eps, y, z);
    double dy = phi(x, y + eps, z) - phi(x, y - eps, z);
    double dz = phi(x, y, z + eps) - phi(x, y, z - eps);
    double mag = std::sqrt(dx*dx + dy*dy + dz*dz);
    if (mag < 1e-30) return {0.0, 0.0, 0.0};
    return {dx / mag, dy / mag, dz / mag};
}

std::tuple<double, double, double> IBMBody::closest_point(double x, double y, double z) const {
    double d = phi(x, y, z);
    auto [nx, ny, nz] = normal(x, y, z);
    return {x - d * nx, y - d * ny, z - d * nz};
}

// ============================================================================
// CylinderBody
// ============================================================================

CylinderBody::CylinderBody(double cx, double cy, double radius)
    : cx_(cx), cy_(cy), radius_(radius)
{
    if (radius <= 0.0) throw std::runtime_error("CylinderBody: radius must be positive");
}

double CylinderBody::phi(double x, double y, double /*z*/) const {
    double dx = x - cx_;
    double dy = y - cy_;
    return std::sqrt(dx*dx + dy*dy) - radius_;
}

std::tuple<double, double, double> CylinderBody::normal(double x, double y, double /*z*/) const {
    double dx = x - cx_;
    double dy = y - cy_;
    double r = std::sqrt(dx*dx + dy*dy);
    if (r < 1e-30) return {1.0, 0.0, 0.0};
    return {dx / r, dy / r, 0.0};
}

std::tuple<double, double, double> CylinderBody::closest_point(double x, double y, double z) const {
    double dx = x - cx_;
    double dy = y - cy_;
    double r = std::sqrt(dx*dx + dy*dy);
    if (r < 1e-30) return {cx_ + radius_, cy_, z};
    double scale = radius_ / r;
    return {cx_ + dx * scale, cy_ + dy * scale, z};
}

// ============================================================================
// SphereBody
// ============================================================================

SphereBody::SphereBody(double cx, double cy, double cz, double radius)
    : cx_(cx), cy_(cy), cz_(cz), radius_(radius)
{
    if (radius <= 0.0) throw std::runtime_error("SphereBody: radius must be positive");
}

double SphereBody::phi(double x, double y, double z) const {
    double dx = x - cx_;
    double dy = y - cy_;
    double dz = z - cz_;
    return std::sqrt(dx*dx + dy*dy + dz*dz) - radius_;
}

std::tuple<double, double, double> SphereBody::normal(double x, double y, double z) const {
    double dx = x - cx_;
    double dy = y - cy_;
    double dz = z - cz_;
    double r = std::sqrt(dx*dx + dy*dy + dz*dz);
    if (r < 1e-30) return {1.0, 0.0, 0.0};
    return {dx / r, dy / r, dz / r};
}

std::tuple<double, double, double> SphereBody::closest_point(double x, double y, double z) const {
    double dx = x - cx_;
    double dy = y - cy_;
    double dz = z - cz_;
    double r = std::sqrt(dx*dx + dy*dy + dz*dz);
    if (r < 1e-30) return {cx_ + radius_, cy_, cz_};
    double scale = radius_ / r;
    return {cx_ + dx * scale, cy_ + dy * scale, cz_ + dz * scale};
}

// ============================================================================
// NACABody — 4-digit NACA airfoil
// ============================================================================

NACABody::NACABody(double x_le, double y_le, double chord, double aoa, const std::string& digits)
    : x_le_(x_le), y_le_(y_le), chord_(chord), aoa_(aoa), digits_(digits)
{
    if (digits.size() != 4) throw std::runtime_error("NACABody: digits must be 4 characters");
    max_camber_ = (digits[0] - '0') / 100.0;
    camber_pos_ = (digits[1] - '0') / 10.0;
    thickness_ = ((digits[2] - '0') * 10 + (digits[3] - '0')) / 100.0;
}

double NACABody::thickness_at(double xn) const {
    // NACA 4-digit thickness distribution (half-thickness)
    // yt = (t/0.2) * (0.2969*sqrt(x) - 0.1260*x - 0.3516*x^2 + 0.2843*x^3 - 0.1015*x^4)
    // Note: original formula has -0.1036*x^4 for open TE; we use -0.1015 for closed TE
    xn = std::max(0.0, std::min(1.0, xn));
    double sx = std::sqrt(xn);
    return (thickness_ / 0.2) * (0.2969 * sx - 0.1260 * xn - 0.3516 * xn*xn
                                  + 0.2843 * xn*xn*xn - 0.1015 * xn*xn*xn*xn);
}

double NACABody::camber_at(double xn) const {
    if (max_camber_ < 1e-10) return 0.0;
    xn = std::max(0.0, std::min(1.0, xn));
    double p = camber_pos_;
    if (p < 1e-10) return 0.0;
    if (xn < p) {
        return max_camber_ / (p * p) * (2.0 * p * xn - xn * xn);
    } else {
        return max_camber_ / ((1.0 - p) * (1.0 - p)) * (1.0 - 2.0 * p + 2.0 * p * xn - xn * xn);
    }
}

double NACABody::phi(double x, double y, double /*z*/) const {
    // Rotate point into airfoil frame
    double dx = x - x_le_;
    double dy = y - y_le_;
    double ca = std::cos(-aoa_);
    double sa = std::sin(-aoa_);
    double x_body = ca * dx - sa * dy;
    double y_body = sa * dx + ca * dy;

    // Normalize to chord
    double xn = x_body / chord_;
    double yn = y_body / chord_;

    // Clamp to [0,1] for distance computation
    double xn_cl = std::max(0.0, std::min(1.0, xn));

    // Get camber and thickness at this x
    double yc = camber_at(xn_cl);
    double yt = thickness_at(xn_cl);

    // Distance from camber line
    double dy_from_camber = yn - yc;

    // Approximate SDF: distance to upper/lower surface
    // Upper surface: yc + yt, Lower surface: yc - yt
    double d_upper = dy_from_camber - yt;
    double d_lower = -(dy_from_camber + yt);

    // Inside if between upper and lower surfaces AND within chord
    double d_y = std::max(d_upper, d_lower);

    // Distance to leading/trailing edge
    double d_le = xn;       // distance ahead of LE (negative = ahead)
    double d_te = xn - 1.0; // distance past TE (positive = past)

    double d_x = std::max(-d_le, d_te);

    // Combine: inside if d_y < 0 AND d_x < 0
    double d;
    if (d_y < 0.0 && d_x < 0.0) {
        // Inside: SDF is negative max of both
        d = std::max(d_y, d_x);
    } else if (d_y >= 0.0 && d_x >= 0.0) {
        // Outside both: Euclidean distance
        d = std::sqrt(d_y * d_y + d_x * d_x);
    } else {
        // Outside one: take the positive one
        d = std::max(d_y, d_x);
    }

    return d * chord_;  // Scale back to physical coordinates
}

// ============================================================================
// StepBody — forward-facing step
// ============================================================================

StepBody::StepBody(double x_step, double y_step)
    : x_step_(x_step), y_step_(y_step)
{}

double StepBody::phi(double x, double y, double /*z*/) const {
    double dx = x - x_step_;
    double dy = y - y_step_;

    if (dx >= 0.0 && dy <= 0.0) {
        // Inside solid: phi = -min(dx, -dy)
        return -std::min(dx, -dy);
    } else if (dx < 0.0 && dy <= 0.0) {
        // In front of vertical face
        return -dx;
    } else if (dx >= 0.0 && dy > 0.0) {
        // Above top face
        return dy;
    } else {
        // Corner region (dx < 0 && dy > 0)
        return std::sqrt(dx * dx + dy * dy);
    }
}

std::tuple<double, double, double> StepBody::normal(double x, double y, double /*z*/) const {
    double dx = x - x_step_;
    double dy = y - y_step_;

    if (dx >= 0.0 && dy <= 0.0) {
        // Inside solid: normal points toward nearest surface
        if (dx < -dy) {
            return {-1.0, 0.0, 0.0};  // nearest is vertical face
        } else {
            return {0.0, 1.0, 0.0};   // nearest is top face
        }
    } else if (dx < 0.0 && dy <= 0.0) {
        // In front of vertical face
        return {-1.0, 0.0, 0.0};
    } else if (dx >= 0.0 && dy > 0.0) {
        // Above top face
        return {0.0, 1.0, 0.0};
    } else {
        // Corner region
        double r = std::sqrt(dx * dx + dy * dy);
        if (r < 1e-30) return {-1.0, 1.0, 0.0};
        return {dx / r, dy / r, 0.0};
    }
}

std::string StepBody::name() const {
    return "ForwardFacingStep";
}

// ============================================================================
// PeriodicHillBody — Breuer et al. 2009 periodic hills
// ============================================================================

PeriodicHillBody::PeriodicHillBody(double h)
    : h_(h)
{
    if (h <= 0.0) throw std::runtime_error("PeriodicHillBody: h must be positive");
}

double PeriodicHillBody::hill_profile_normalized(double xn) const {
    // Evaluate hill height y/h for x/h in [0, 1.929]
    // 6 piecewise cubic polynomial segments
    // Clamp to zero beyond the hill foot
    if (xn >= 1.929) return 0.0;
    if (xn <= 0.3214) {
        double val = 1.0 + 0.18973 * xn * xn + (-1.66518) * xn * xn * xn;
        return std::min(1.0, val);
    } else if (xn <= 0.5) {
        return 0.8955 + 0.97552 * xn + (-2.84514) * xn * xn + 1.48159 * xn * xn * xn;
    } else if (xn <= 0.7143) {
        return 0.9213 + 0.82068 * xn + (-2.53546) * xn * xn + 1.27499 * xn * xn * xn;
    } else if (xn <= 1.071) {
        return 1.445 + (-1.37956) * xn + 0.54488 * xn * xn + (-0.16231) * xn * xn * xn;
    } else if (xn <= 1.429) {
        return 0.6401 + 0.87444 * xn + (-1.55859) * xn * xn + 0.49216 * xn * xn * xn;
    } else {
        double val = 2.0139 + (-2.01040) * xn + 0.46060 * xn * xn + 0.02097 * xn * xn * xn;
        return std::max(0.0, val);
    }
}

double PeriodicHillBody::hill_height(double x) const {
    // Map to periodic domain [0, 9h)
    double period = 9.0 * h_;
    double xp = std::fmod(x, period);
    if (xp < 0.0) xp += period;

    // Normalize to x/h
    double xn = xp / h_;

    if (xn <= 1.929) {
        return h_ * hill_profile_normalized(xn);
    } else if (xn <= 7.071) {
        return 0.0;  // flat region
    } else {
        // Mirror: hill_height(9h - x) for the descending side
        double xn_mirror = 9.0 - xn;
        return h_ * hill_profile_normalized(xn_mirror);
    }
}

double PeriodicHillBody::phi(double x, double y, double /*z*/) const {
    return y - hill_height(x);
}

std::string PeriodicHillBody::name() const {
    return "PeriodicHills";
}

// ============================================================================
// Factory
// ============================================================================

std::unique_ptr<IBMBody> create_ibm_body(const std::string& type,
    double param1, double param2, double param3,
    double param4, const std::string& extra) {
    if (type == "cylinder") {
        return std::make_unique<CylinderBody>(param1, param2, param3);
    } else if (type == "sphere") {
        return std::make_unique<SphereBody>(param1, param2, param3, param4);
    } else if (type == "naca") {
        return std::make_unique<NACABody>(param1, param2, param3, param4, extra);
    } else if (type == "step") {
        return std::make_unique<StepBody>(param1, param2);
    } else if (type == "periodic_hill" || type == "hills") {
        return std::make_unique<PeriodicHillBody>(param1);
    }
    throw std::runtime_error("Unknown IBM body type: " + type);
}

} // namespace nncfd
