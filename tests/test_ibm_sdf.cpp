/// @file test_ibm_sdf.cpp
/// @brief Tests for IBM geometry signed distance functions
///
/// Test coverage:
///   1. CylinderBody: phi, normal, closest_point
///   2. SphereBody: phi, normal, closest_point
///   3. NACABody: phi (inside, outside, near surface)
///   4. Factory function

#include "ibm_geometry.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

using namespace nncfd;

void test_cylinder() {
    CylinderBody cyl(5.0, 0.0, 0.5);  // center=(5,0), radius=0.5

    // On surface: phi = 0
    assert(std::abs(cyl.phi(5.5, 0.0, 0.0)) < 1e-14);
    assert(std::abs(cyl.phi(4.5, 0.0, 0.0)) < 1e-14);
    assert(std::abs(cyl.phi(5.0, 0.5, 0.0)) < 1e-14);

    // Inside: phi < 0
    assert(cyl.phi(5.0, 0.0, 0.0) < 0);
    assert(std::abs(cyl.phi(5.0, 0.0, 0.0) - (-0.5)) < 1e-14);

    // Outside: phi > 0
    assert(cyl.phi(6.0, 0.0, 0.0) > 0);
    assert(std::abs(cyl.phi(6.0, 0.0, 0.0) - 0.5) < 1e-14);

    // Normal on surface points outward
    auto [nx, ny, nz] = cyl.normal(5.5, 0.0, 0.0);
    assert(std::abs(nx - 1.0) < 1e-14);
    assert(std::abs(ny) < 1e-14);
    assert(std::abs(nz) < 1e-14);

    auto [nx2, ny2, nz2] = cyl.normal(5.0, 0.5, 0.0);
    assert(std::abs(nx2) < 1e-14);
    assert(std::abs(ny2 - 1.0) < 1e-14);

    // Closest point
    auto [cx, cy, cz] = cyl.closest_point(6.0, 0.0, 3.0);
    assert(std::abs(cx - 5.5) < 1e-14);
    assert(std::abs(cy) < 1e-14);
    assert(std::abs(cz - 3.0) < 1e-14);  // z preserved

    // z-independence: same phi for different z
    assert(std::abs(cyl.phi(5.5, 0.0, 10.0)) < 1e-14);

    std::cout << "PASS: CylinderBody SDF" << std::endl;
}

void test_sphere() {
    SphereBody sph(0.0, 0.0, 0.0, 1.0);  // center=(0,0,0), radius=1

    // On surface: phi = 0
    assert(std::abs(sph.phi(1.0, 0.0, 0.0)) < 1e-14);
    assert(std::abs(sph.phi(0.0, -1.0, 0.0)) < 1e-14);
    assert(std::abs(sph.phi(0.0, 0.0, 1.0)) < 1e-14);

    // Inside: phi < 0
    assert(sph.phi(0.0, 0.0, 0.0) < 0);
    assert(std::abs(sph.phi(0.0, 0.0, 0.0) - (-1.0)) < 1e-14);

    // Outside: phi > 0
    assert(std::abs(sph.phi(2.0, 0.0, 0.0) - 1.0) < 1e-14);

    // Normal
    auto [nx, ny, nz] = sph.normal(2.0, 0.0, 0.0);
    assert(std::abs(nx - 1.0) < 1e-14);
    assert(std::abs(ny) < 1e-14);
    assert(std::abs(nz) < 1e-14);

    // Closest point from outside
    auto [cx, cy, cz] = sph.closest_point(3.0, 0.0, 0.0);
    assert(std::abs(cx - 1.0) < 1e-14);
    assert(std::abs(cy) < 1e-14);
    assert(std::abs(cz) < 1e-14);

    std::cout << "PASS: SphereBody SDF" << std::endl;
}

void test_naca() {
    NACABody naca(0.0, 0.0, 1.0, 0.0, "0012");  // chord=1, aoa=0

    // Well inside: phi < 0
    double phi_inside = naca.phi(0.3, 0.0, 0.0);
    assert(phi_inside < 0);

    // Well outside: phi > 0
    double phi_outside = naca.phi(0.5, 0.5, 0.0);
    assert(phi_outside > 0);

    // Near leading edge: approximately on surface
    double phi_le = naca.phi(0.0, 0.0, 0.0);
    assert(std::abs(phi_le) < 0.01);

    // Above upper surface at mid-chord: outside
    // NACA 0012 max thickness is 12% of chord, half-thickness ~6% at x=0.3
    double phi_above = naca.phi(0.3, 0.08, 0.0);
    assert(phi_above > 0);

    // Below lower surface at mid-chord: outside
    double phi_below = naca.phi(0.3, -0.08, 0.0);
    assert(phi_below > 0);

    // z-independence (extruded in z)
    assert(std::abs(naca.phi(0.3, 0.0, 0.0) - naca.phi(0.3, 0.0, 5.0)) < 1e-14);

    // Symmetric airfoil: phi(x,y) = phi(x,-y)
    assert(std::abs(naca.phi(0.5, 0.03, 0.0) - naca.phi(0.5, -0.03, 0.0)) < 1e-12);

    std::cout << "PASS: NACABody SDF" << std::endl;
}

void test_factory() {
    auto cyl = create_ibm_body("cylinder", 1.0, 2.0, 0.5);
    assert(cyl->name() == "Cylinder");
    assert(std::abs(cyl->phi(1.5, 2.0, 0.0)) < 1e-14);

    auto sph = create_ibm_body("sphere", 0.0, 0.0, 0.0, 1.0);
    assert(sph->name() == "Sphere");

    auto naca = create_ibm_body("naca", 0.0, 0.0, 1.0, 0.0, "0012");
    assert(naca->name() == "NACA0012");

    std::cout << "PASS: IBM factory" << std::endl;
}

int main() {
    test_cylinder();
    test_sphere();
    test_naca();
    test_factory();

    std::cout << "\nAll IBM SDF tests PASSED" << std::endl;
    return 0;
}
