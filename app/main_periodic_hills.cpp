/// Periodic hills flow solver
/// Flow over periodic hills - a common turbulence model benchmark case
/// 
/// Geometry: sinusoidal hill on the bottom wall with periodic BCs in x
/// Reference: Breuer et al. (2009), Mellen et al. (2000)

#include "mesh.hpp"
#include "fields.hpp"
#include "solver.hpp"
#include "config.hpp"
#include "timing.hpp"
#include "turbulence_model.hpp"
#include "turbulence_baseline.hpp"
#include "turbulence_nn_mlp.hpp"
#include "turbulence_nn_tbnn.hpp"

#include <iostream>
#include <iomanip>
#include <cmath>
#include <fstream>

using namespace nncfd;

/// Hill shape function: y_hill(x) for bottom wall
/// Standard periodic hill geometry from Mellen et al.
double hill_height(double x, double L_hill, double h_max) {
    // Normalize x to [0, L_hill]
    double x_norm = std::fmod(x, L_hill);
    if (x_norm < 0) x_norm += L_hill;
    
    // Hill occupies x in [0, 3.857] with L_hill = 9
    // Simplified sinusoidal approximation
    double x_hill = x_norm / L_hill;  // [0, 1]
    
    if (x_hill < 0.0 || x_hill > 0.5) {
        return 0.0;  // Flat bottom
    }
    
    // Sinusoidal hill shape
    double h = h_max * std::sin(M_PI * x_hill / 0.5);
    return std::max(0.0, h);
}

/// Create a mask field for the hill geometry
/// mask(i,j) = 0 if cell is inside solid (hill), 1 if in fluid
void create_hill_mask(const Mesh& mesh, ScalarField& mask, 
                      double L_hill, double h_max) {
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double x = mesh.x(i);
            double y = mesh.y(j);
            double y_hill = mesh.y_min + hill_height(x, L_hill, h_max);
            
            if (y < y_hill) {
                mask(i, j) = 0.0;  // Inside hill (solid)
            } else {
                mask(i, j) = 1.0;  // Fluid
            }
        }
    }
}

/// Modified solver step that accounts for hill geometry
/// This is a simplified approach - a production code would use
/// immersed boundary method or body-fitted coordinates
class PeriodicHillsSolver {
public:
    PeriodicHillsSolver(const Mesh& mesh, const Config& config, 
                        double L_hill, double h_max)
        : mesh_(mesh)
        , config_(config)
        , solver_(mesh, config)
        , mask_(mesh)
        , L_hill_(L_hill)
        , h_max_(h_max)
    {
        create_hill_mask(mesh, mask_, L_hill, h_max);
    }
    
    void initialize() {
        // Initialize with uniform flow
        solver_.initialize_uniform(1.0, 0.0);
        
        // Zero velocity inside hills
        apply_hill_bc();
    }
    
    void set_turbulence_model(std::unique_ptr<TurbulenceModel> model) {
        solver_.set_turbulence_model(std::move(model));
    }
    
    void set_body_force(double fx, double fy) {
        solver_.set_body_force(fx, fy);
    }
    
    void apply_hill_bc() {
        // Apply no-slip on hill surface by setting velocity to zero in masked cells
        // and in cells adjacent to hills
        auto& vel = solver_.velocity();
        
        for (int j = mesh_.j_begin(); j < mesh_.j_end(); ++j) {
            for (int i = mesh_.i_begin(); i < mesh_.i_end(); ++i) {
                if (mask_(i, j) < 0.5) {
                    // Inside hill
                    vel.u(i, j) = 0.0;
                    vel.v(i, j) = 0.0;
                } else {
                    // Check if adjacent to hill
                    bool near_hill = false;
                    if (i > mesh_.i_begin() && mask_(i-1, j) < 0.5) near_hill = true;
                    if (i < mesh_.i_end()-1 && mask_(i+1, j) < 0.5) near_hill = true;
                    if (j > mesh_.j_begin() && mask_(i, j-1) < 0.5) near_hill = true;
                    
                    if (near_hill) {
                        // Reduce velocity near hill (simple immersed boundary approximation)
                        vel.u(i, j) *= 0.5;
                        vel.v(i, j) *= 0.5;
                    }
                }
            }
        }
    }
    
    double step() {
        double residual = solver_.step();
        apply_hill_bc();
        return residual;
    }
    
    std::pair<double, int> solve_steady() {
        double residual = 1.0;
        int iter = 0;
        
        if (config_.verbose) {
            std::cout << std::setw(8) << "Iter"
                      << std::setw(15) << "Residual"
                      << std::setw(15) << "Max |u|"
                      << "\n";
        }
        
        for (iter = 0; iter < config_.max_iter; ++iter) {
            residual = step();
            
            if (config_.verbose && (iter + 1) % config_.output_freq == 0) {
                double max_vel = solver_.velocity().max_magnitude();
                std::cout << std::setw(8) << iter + 1
                          << std::setw(15) << std::scientific << std::setprecision(3) << residual
                          << std::setw(15) << std::fixed << max_vel
                          << "\n";
            }
            
            if (residual < config_.tol) {
                if (config_.verbose) {
                    std::cout << "Converged at iteration " << iter + 1 << "\n";
                }
                break;
            }
        }
        
        return {residual, iter + 1};
    }
    
    std::pair<double, int> solve_steady_with_snapshots(
        const std::string& output_prefix,
        int num_snapshots) 
    {
        // Calculate snapshot frequency
        int snapshot_freq = (num_snapshots > 0) ? 
                            std::max(1, config_.max_iter / num_snapshots) : -1;
        
        if (config_.verbose && !output_prefix.empty()) {
            std::cout << "Will output ";
            if (num_snapshots > 0) {
                std::cout << num_snapshots << " VTK snapshots (every " 
                         << snapshot_freq << " iterations)\n";
            } else {
                std::cout << "final VTK snapshot only\n";
            }
        }
        
        double residual = 1.0;
        int iter = 0;
        int snapshot_count = 0;
        
        if (config_.verbose) {
            std::cout << std::setw(8) << "Iter"
                      << std::setw(15) << "Residual"
                      << std::setw(15) << "Max |u|"
                      << "\n";
        }
        
        for (iter = 0; iter < config_.max_iter; ++iter) {
            residual = step();
            
            // Write VTK snapshots at regular intervals
            if (!output_prefix.empty() && num_snapshots > 0 && 
                snapshot_freq > 0 && (iter + 1) % snapshot_freq == 0) {
                snapshot_count++;
                std::string vtk_file = output_prefix + "_" + 
                                      std::to_string(snapshot_count) + ".vtk";
                try {
                    write_vtk(vtk_file);
                    if (config_.verbose) {
                        std::cout << "Wrote snapshot " << snapshot_count 
                                 << ": " << vtk_file << "\n";
                    }
                } catch (const std::exception& e) {
                    std::cerr << "Warning: Could not write VTK snapshot: " 
                             << e.what() << "\n";
                }
            }
            
            if (config_.verbose && (iter + 1) % config_.output_freq == 0) {
                double max_vel = solver_.velocity().max_magnitude();
                std::cout << std::setw(8) << iter + 1
                          << std::setw(15) << std::scientific << std::setprecision(3) << residual
                          << std::setw(15) << std::fixed << max_vel
                          << "\n";
            }
            
            if (residual < config_.tol) {
                if (config_.verbose) {
                    std::cout << "Converged at iteration " << iter + 1 << "\n";
                }
                break;
            }
        }
        
        // Write final snapshot
        if (!output_prefix.empty()) {
            std::string final_file = output_prefix + "_final.vtk";
            try {
                write_vtk(final_file);
                if (config_.verbose) {
                    std::cout << "Final VTK output: " << final_file << "\n";
                    if (num_snapshots > 0) {
                        std::cout << "Total VTK snapshots: " << snapshot_count + 1 
                                 << " (" << snapshot_count << " during + 1 final)\n";
                    }
                }
            } catch (const std::exception& e) {
                std::cerr << "Warning: Could not write final VTK: " 
                         << e.what() << "\n";
            }
        }
        
        return {residual, iter + 1};
    }
    
    const VectorField& velocity() const { return solver_.velocity(); }
    const ScalarField& pressure() const { return solver_.pressure(); }
    const ScalarField& mask() const { return mask_; }
    
    double bulk_velocity() const {
        // Compute bulk velocity excluding hill region
        double sum = 0.0;
        int count = 0;
        
        for (int j = mesh_.j_begin(); j < mesh_.j_end(); ++j) {
            for (int i = mesh_.i_begin(); i < mesh_.i_end(); ++i) {
                if (mask_(i, j) > 0.5) {
                    sum += solver_.velocity().u(i, j);
                    ++count;
                }
            }
        }
        
        return count > 0 ? sum / count : 0.0;
    }
    
    void write_fields(const std::string& prefix) const {
        solver_.write_fields(prefix);
        mask_.write(prefix + "_mask.dat");
    }
    
    void write_vtk(const std::string& filename) const {
        solver_.write_vtk(filename);
    }
    
private:
    const Mesh& mesh_;
    Config config_;
    RANSSolver solver_;
    ScalarField mask_;
    [[maybe_unused]] double L_hill_;  // Stored for potential debugging/output
    [[maybe_unused]] double h_max_;   // Stored for potential debugging/output
};

/// Write velocity profile at specified x locations
void write_profiles(const std::string& filename, const Mesh& mesh,
                    const VectorField& velocity, const ScalarField& mask,
                    const std::vector<double>& x_locs) {
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Cannot open file: " << filename << "\n";
        return;
    }
    
    file << "# Velocity profiles at different x locations\n";
    file << "# Columns: y, u(x1), u(x2), ...\n";
    
    // Find i indices for each x location
    std::vector<int> i_locs;
    for (double x_target : x_locs) {
        int i_best = mesh.i_begin();
        double min_dist = std::abs(mesh.x(i_best) - x_target);
        for (int i = mesh.i_begin(); i < mesh.i_end(); ++i) {
            double dist = std::abs(mesh.x(i) - x_target);
            if (dist < min_dist) {
                min_dist = dist;
                i_best = i;
            }
        }
        i_locs.push_back(i_best);
    }
    
    // Write header with actual x values
    file << "# x = ";
    for (int i_loc : i_locs) {
        file << mesh.x(i_loc) << " ";
    }
    file << "\n";
    
    // Write profiles
    for (int j = mesh.j_begin(); j < mesh.j_end(); ++j) {
        file << mesh.y(j);
        for (int i_loc : i_locs) {
            if (mask(i_loc, j) > 0.5) {
                file << " " << velocity.u(i_loc, j);
            } else {
                file << " 0.0";  // Inside hill (solid region, zero velocity)
            }
        }
        file << "\n";
    }
}

int main(int argc, char** argv) {
    std::cout << "=== Periodic Hills Flow Solver ===\n\n";
    
    // Configuration
    Config config;
    
    // Periodic hills geometry parameters
    double L_hill = 9.0;    // Hill period
    double h_max = 0.5;     // Maximum hill height
    double H_channel = 3.0; // Channel height
    
    // Mesh
    config.Nx = 64;
    config.Ny = 48;
    config.x_min = 0.0;
    config.x_max = L_hill;
    config.y_min = 0.0;
    config.y_max = H_channel;
    
    config.nu = 0.001;      // Re_H ~ 10,000 based on H and bulk velocity
    config.dp_dx = -0.01;   // Pressure gradient
    
    config.dt = 0.0005;
    config.max_iter = 100000;
    config.tol = 1e-6;
    config.output_freq = 2000;
    config.verbose = true;
    
    config.turb_model = TurbulenceModelType::None;
    
    config.poisson_tol = 1e-7;
    config.poisson_max_iter = 5000;
    config.poisson_omega = 1.7;
    
    // Parse command line
    config.parse_args(argc, argv);
    config.print();
    
    std::cout << "Hill parameters: L = " << L_hill << ", h_max = " << h_max << "\n\n";
    
    // Create mesh
    Mesh mesh;
    if (config.stretch_y) {
        mesh.init_stretched_y(config.Nx, config.Ny,
                              config.x_min, config.x_max,
                              config.y_min, config.y_max,
                              Mesh::tanh_stretching(config.stretch_beta));
    } else {
        mesh.init_uniform(config.Nx, config.Ny,
                          config.x_min, config.x_max,
                          config.y_min, config.y_max);
    }
    
    std::cout << "Mesh: " << mesh.Nx << " x " << mesh.Ny << " cells\n";
    std::cout << "dx = " << mesh.dx << ", dy = " << mesh.dy << "\n\n";
    
    // Create solver
    PeriodicHillsSolver solver(mesh, config, L_hill, h_max);
    
    // Set body force
    solver.set_body_force(-config.dp_dx, 0.0);
    
    // Set turbulence model if requested
    if (config.turb_model != TurbulenceModelType::None) {
        auto turb_model = create_turbulence_model(config.turb_model,
                                                  config.nn_weights_path,
                                                  config.nn_scaling_path);
        if (turb_model) {
            turb_model->set_nu(config.nu);
            
            double delta = H_channel - h_max;  // Effective channel height
            if (auto* ml = dynamic_cast<MixingLengthModel*>(turb_model.get())) {
                ml->set_delta(delta);
            }
            if (auto* nn_mlp = dynamic_cast<TurbulenceNNMLP*>(turb_model.get())) {
                nn_mlp->set_delta(delta);
                nn_mlp->set_nu_t_max(config.nu_t_max);
            }
            if (auto* nn_tbnn = dynamic_cast<TurbulenceNNTBNN*>(turb_model.get())) {
                nn_tbnn->set_delta(delta);
            }
            
            solver.set_turbulence_model(std::move(turb_model));
        }
    }
    
    // Initialize
    solver.initialize();
    
    // Solve to steady state with automatic VTK snapshots
    ScopedTimer total_timer("Total simulation", true);
    
    auto [residual, iterations] = solver.solve_steady_with_snapshots(
        config.output_dir + "periodic_hills",
        config.num_snapshots
    );
    
    total_timer.stop();
    
    // Results
    std::cout << "\n=== Results ===\n";
    std::cout << "Final residual: " << std::scientific << residual << "\n";
    std::cout << "Iterations: " << iterations << "\n";
    std::cout << "Bulk velocity: " << std::fixed << std::setprecision(6) 
              << solver.bulk_velocity() << "\n";
    
    // Write additional output files
    solver.write_fields(config.output_dir + "periodic_hills");
    
    // Write velocity profiles at key locations
    std::vector<double> x_profile_locs = {
        0.5,                    // Near hill crest
        2.0,                    // Separation region
        4.0,                    // Recirculation
        6.0,                    // Reattachment region
        8.0                     // Recovery
    };
    write_profiles(config.output_dir + "velocity_profiles.dat", mesh,
                   solver.velocity(), solver.mask(), x_profile_locs);
    
    // Print timing summary
    TimingStats::instance().print_summary();
    
    return 0;
}


