/// @file test_ibm_step_sdf.cpp
/// @brief Tests for StepBody signed distance function
///
/// Test coverage:
///   1. phi deep inside solid
///   2. phi inside near vertical face
///   3. phi inside near top face
///   4. phi on vertical face (zero)
///   5. phi on horizontal top (zero)
///   6. phi at corner (zero)
///   7. phi in front of step
///   8. phi above step
///   9. phi in corner region (outside)
///  10. name()

#include "ibm_geometry.hpp"
#include <cmath>
#include <iostream>
#include <string>

using namespace nncfd;

int main() {
    StepBody step(5.0, 1.0);
    const double tol = 1e-12;

    // Deep inside solid: x=10 (dx=5), y=0.5 (dy=-0.5) -> phi = -min(5, 0.5) = -0.5
    {
        double val = step.phi(10.0, 0.5, 0.0);
        if (std::abs(val - (-0.5)) > tol) {
            throw std::runtime_error("FAIL: phi deep inside = " + std::to_string(val)
                                     + ", expected -0.5");
        }
    }

    // Inside near vertical face: x=5.1 (dx=0.1), y=0.5 (dy=-0.5) -> phi = -min(0.1, 0.5) = -0.1
    {
        double val = step.phi(5.1, 0.5, 0.0);
        if (std::abs(val - (-0.1)) > tol) {
            throw std::runtime_error("FAIL: phi inside near face = " + std::to_string(val)
                                     + ", expected -0.1");
        }
    }

    // Inside near top: x=10 (dx=5), y=0.9 (dy=-0.1) -> phi = -min(5, 0.1) = -0.1
    {
        double val = step.phi(10.0, 0.9, 0.0);
        if (std::abs(val - (-0.1)) > tol) {
            throw std::runtime_error("FAIL: phi inside near top = " + std::to_string(val)
                                     + ", expected -0.1");
        }
    }

    // On vertical face: x=5.0 (dx=0), y=0.5 (dy=-0.5) -> phi = -min(0, 0.5) = 0
    {
        double val = step.phi(5.0, 0.5, 0.0);
        if (std::abs(val) > tol) {
            throw std::runtime_error("FAIL: phi on vertical face = " + std::to_string(val)
                                     + ", expected 0.0");
        }
    }

    // On horizontal top: x=7.0 (dx=2), y=1.0 (dy=0) -> phi = 0
    {
        double val = step.phi(7.0, 1.0, 0.0);
        if (std::abs(val) > tol) {
            throw std::runtime_error("FAIL: phi on horizontal top = " + std::to_string(val)
                                     + ", expected 0.0");
        }
    }

    // Corner: x=5.0 (dx=0), y=1.0 (dy=0) -> phi = 0
    {
        double val = step.phi(5.0, 1.0, 0.0);
        if (std::abs(val) > tol) {
            throw std::runtime_error("FAIL: phi at corner = " + std::to_string(val)
                                     + ", expected 0.0");
        }
    }

    // In front of step: x=4.0 (dx=-1), y=0.5 (dy=-0.5) -> phi = -dx = 1.0
    {
        double val = step.phi(4.0, 0.5, 0.0);
        if (std::abs(val - 1.0) > tol) {
            throw std::runtime_error("FAIL: phi in front = " + std::to_string(val)
                                     + ", expected 1.0");
        }
    }

    // Above step: x=7.0 (dx=2), y=2.0 (dy=1.0) -> phi = dy = 1.0
    {
        double val = step.phi(7.0, 2.0, 0.0);
        if (std::abs(val - 1.0) > tol) {
            throw std::runtime_error("FAIL: phi above = " + std::to_string(val)
                                     + ", expected 1.0");
        }
    }

    // Corner region outside: x=4.0 (dx=-1), y=2.0 (dy=1.0) -> phi = sqrt(1+1) = sqrt(2)
    {
        double val = step.phi(4.0, 2.0, 0.0);
        double expected = std::sqrt(2.0);
        if (std::abs(val - expected) > tol) {
            throw std::runtime_error("FAIL: phi corner region = " + std::to_string(val)
                                     + ", expected " + std::to_string(expected));
        }
    }

    // Name
    {
        std::string n = step.name();
        if (n != "ForwardFacingStep") {
            throw std::runtime_error("FAIL: name() = " + n
                                     + ", expected ForwardFacingStep");
        }
    }

    std::cout << "All StepBody SDF tests PASSED" << std::endl;
    return 0;
}
