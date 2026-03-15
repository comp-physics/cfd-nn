/// @file test_ibm_hills_sdf.cpp
/// @brief Tests for PeriodicHillBody signed distance function
///
/// Test coverage:
///   1. hill_height at crest (x=0)
///   2. hill_height in flat region
///   3. hill_height periodicity
///   4. Symmetry about x = 4.5h
///   5. Continuity at segment boundaries
///   6. phi above/below crest
///   7. phi in flat region
///   8. phi periodicity

#include "ibm_geometry.hpp"
#include <cmath>
#include <iostream>
#include <string>

using namespace nncfd;

int main() {
    PeriodicHillBody hill(1.0);
    const double tol = 1e-10;
    const double loose_tol = 0.01;

    // Crest at x=0: hill_height should be h = 1.0
    {
        double val = hill.hill_height(0.0);
        if (std::abs(val - 1.0) > tol) {
            throw std::runtime_error("FAIL: hill_height(0) = " + std::to_string(val)
                                     + ", expected 1.0");
        }
    }

    // Flat region at x=3.0: hill_height = 0
    {
        double val = hill.hill_height(3.0);
        if (std::abs(val) > tol) {
            throw std::runtime_error("FAIL: hill_height(3) = " + std::to_string(val)
                                     + ", expected 0.0");
        }
    }

    // Flat region at x=5.0: hill_height = 0
    {
        double val = hill.hill_height(5.0);
        if (std::abs(val) > tol) {
            throw std::runtime_error("FAIL: hill_height(5) = " + std::to_string(val)
                                     + ", expected 0.0");
        }
    }

    // Periodicity: hill_height(9.0) = hill_height(0.0) = 1.0
    {
        double val = hill.hill_height(9.0);
        if (std::abs(val - 1.0) > tol) {
            throw std::runtime_error("FAIL: hill_height(9) = " + std::to_string(val)
                                     + ", expected 1.0 (periodic crest)");
        }
    }

    // Symmetry: hill(x) == hill(9-x) for x in ascending part
    {
        double test_points[] = {0.0, 0.2, 0.5, 1.0, 1.5, 1.929};
        for (double x : test_points) {
            double h1 = hill.hill_height(x);
            double h2 = hill.hill_height(9.0 - x);
            if (std::abs(h1 - h2) > tol) {
                throw std::runtime_error("FAIL: symmetry at x=" + std::to_string(x)
                                         + ": hill(" + std::to_string(x) + ")="
                                         + std::to_string(h1) + " != hill("
                                         + std::to_string(9.0 - x) + ")="
                                         + std::to_string(h2));
            }
        }
    }

    // Continuity at segment boundaries (loose tolerance)
    {
        double boundaries[] = {0.3214, 0.5, 0.7143, 1.071, 1.429, 1.929};
        double eps = 1e-6;
        for (double xb : boundaries) {
            double h_left = hill.hill_height(xb - eps);
            double h_right = hill.hill_height(xb + eps);
            if (std::abs(h_left - h_right) > loose_tol) {
                throw std::runtime_error("FAIL: continuity at x/h=" + std::to_string(xb)
                                         + ": left=" + std::to_string(h_left)
                                         + ", right=" + std::to_string(h_right));
            }
        }
    }

    // phi above crest > 0: y=1.5 at x=0 (hill=1.0) -> phi = 1.5 - 1.0 = 0.5
    {
        double val = hill.phi(0.0, 1.5, 0.0);
        if (val <= 0.0) {
            throw std::runtime_error("FAIL: phi above crest = " + std::to_string(val)
                                     + ", expected > 0");
        }
    }

    // phi below crest < 0: y=0.5 at x=0 (hill=1.0) -> phi = 0.5 - 1.0 = -0.5
    {
        double val = hill.phi(0.0, 0.5, 0.0);
        if (val >= 0.0) {
            throw std::runtime_error("FAIL: phi below crest = " + std::to_string(val)
                                     + ", expected < 0");
        }
    }

    // phi in flat region > 0: y=0.1 at x=3.0 (hill=0.0) -> phi = 0.1
    {
        double val = hill.phi(3.0, 0.1, 0.0);
        if (val <= 0.0) {
            throw std::runtime_error("FAIL: phi in flat region = " + std::to_string(val)
                                     + ", expected > 0");
        }
    }

    // Periodicity of phi: phi(1, 0.5, 0) == phi(10, 0.5, 0)
    {
        double val1 = hill.phi(1.0, 0.5, 0.0);
        double val2 = hill.phi(10.0, 0.5, 0.0);
        if (std::abs(val1 - val2) > tol) {
            throw std::runtime_error("FAIL: periodicity: phi(1,0.5,0)=" + std::to_string(val1)
                                     + " != phi(10,0.5,0)=" + std::to_string(val2));
        }
    }

    // Name
    {
        std::string n = hill.name();
        if (n != "PeriodicHills") {
            throw std::runtime_error("FAIL: name() = " + n
                                     + ", expected PeriodicHills");
        }
    }

    std::cout << "All PeriodicHillBody SDF tests PASSED" << std::endl;
    return 0;
}
