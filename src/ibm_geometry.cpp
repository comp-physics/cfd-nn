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
    }
    throw std::runtime_error("Unknown IBM body type: " + type);
}

} // namespace nncfd
