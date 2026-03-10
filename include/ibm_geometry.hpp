#pragma once

/// @file ibm_geometry.hpp
/// @brief Signed distance functions for immersed boundary method geometries
///
/// Provides analytically-defined bodies (cylinder, sphere, NACA airfoil) via
/// signed distance functions (SDF). Convention: phi < 0 inside, phi > 0 outside,
/// phi = 0 on surface. Normals point outward.

#include <string>
#include <tuple>
#include <memory>

namespace nncfd {

/// Base class for immersed boundary geometry
class IBMBody {
public:
    virtual ~IBMBody() = default;

    /// Signed distance function: negative inside, positive outside
    virtual double phi(double x, double y, double z) const = 0;

    /// Outward surface normal (unit vector)
    /// Default: finite-difference gradient of phi
    virtual std::tuple<double, double, double> normal(double x, double y, double z) const;

    /// Closest point on surface (default: x - phi*normal)
    virtual std::tuple<double, double, double> closest_point(double x, double y, double z) const;

    /// Name for logging
    virtual std::string name() const = 0;
};

/// Infinite cylinder aligned with z-axis
class CylinderBody : public IBMBody {
public:
    /// @param cx      x-center
    /// @param cy      y-center
    /// @param radius  Cylinder radius
    CylinderBody(double cx, double cy, double radius);

    double phi(double x, double y, double z) const override;
    std::tuple<double, double, double> normal(double x, double y, double z) const override;
    std::tuple<double, double, double> closest_point(double x, double y, double z) const override;
    std::string name() const override { return "Cylinder"; }

private:
    double cx_, cy_, radius_;
};

/// Sphere centered at (cx, cy, cz)
class SphereBody : public IBMBody {
public:
    SphereBody(double cx, double cy, double cz, double radius);

    double phi(double x, double y, double z) const override;
    std::tuple<double, double, double> normal(double x, double y, double z) const override;
    std::tuple<double, double, double> closest_point(double x, double y, double z) const override;
    std::string name() const override { return "Sphere"; }

private:
    double cx_, cy_, cz_, radius_;
};

/// 4-digit NACA airfoil (extruded in z)
class NACABody : public IBMBody {
public:
    /// @param x_le   Leading edge x-coordinate
    /// @param y_le   Leading edge y-coordinate
    /// @param chord  Chord length
    /// @param aoa    Angle of attack (radians)
    /// @param digits NACA 4-digit designation (e.g., "0012")
    NACABody(double x_le, double y_le, double chord, double aoa, const std::string& digits);

    double phi(double x, double y, double z) const override;
    std::string name() const override { return "NACA" + digits_; }

private:
    double x_le_, y_le_, chord_, aoa_;
    std::string digits_;
    double max_camber_, camber_pos_, thickness_;

    /// NACA thickness distribution at normalized x in [0,1]
    double thickness_at(double xn) const;
    /// Camber line y at normalized x
    double camber_at(double xn) const;
};

/// Forward-facing step: solid region for x >= x_step, y <= y_step
class StepBody : public IBMBody {
public:
    StepBody(double x_step, double y_step);
    double phi(double x, double y, double z) const override;
    std::tuple<double, double, double> normal(double x, double y, double z) const override;
    std::string name() const override;

private:
    double x_step_;
    double y_step_;
};

/// Periodic hills (Breuer et al. 2009, ERCOFTAC UFR 3-30)
/// Hill profile defined by 6 piecewise cubic polynomials, periodic in x with period 9h.
class PeriodicHillBody : public IBMBody {
public:
    explicit PeriodicHillBody(double h);
    double phi(double x, double y, double z) const override;
    std::string name() const override;
    /// Hill profile height y_hill(x) for arbitrary x (periodic, physical coords)
    double hill_height(double x) const;

private:
    double h_;
    double hill_profile_normalized(double xn) const;
};

/// Factory: create body from type string
std::unique_ptr<IBMBody> create_ibm_body(const std::string& type,
    double param1, double param2, double param3,
    double param4 = 0.0, const std::string& extra = "");

} // namespace nncfd
