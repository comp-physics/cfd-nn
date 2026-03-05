/// @file checkpoint.cpp
/// @brief HDF5 checkpoint/restart implementation

#include "checkpoint.hpp"

#ifdef USE_HDF5
#include <hdf5.h>
#endif

#include <iostream>
#include <algorithm>

namespace nncfd {

#ifdef USE_HDF5

namespace {

void write_scalar_dataset(hid_t file, const char* name,
                          const double* data, hsize_t size) {
    hid_t space = H5Screate_simple(1, &size, nullptr);
    hid_t dset = H5Dcreate2(file, name, H5T_NATIVE_DOUBLE, space,
                             H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    H5Dclose(dset);
    H5Sclose(space);
}

void read_scalar_dataset(hid_t file, const char* name,
                         double* data, hsize_t expected_size) {
    hid_t dset = H5Dopen2(file, name, H5P_DEFAULT);
    if (dset < 0) {
        std::cerr << "[Checkpoint] Dataset '" << name << "' not found\n";
        return;
    }
    hid_t space = H5Dget_space(dset);
    hsize_t dims;
    H5Sget_simple_extent_dims(space, &dims, nullptr);
    if (dims != expected_size) {
        std::cerr << "[Checkpoint] Size mismatch for '" << name
                  << "': expected " << expected_size << ", got " << dims << "\n";
    }
    H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    H5Sclose(space);
    H5Dclose(dset);
}

void write_int_attr(hid_t file, const char* name, int value) {
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t attr = H5Acreate2(file, name, H5T_NATIVE_INT, space,
                             H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, H5T_NATIVE_INT, &value);
    H5Aclose(attr);
    H5Sclose(space);
}

void write_double_attr(hid_t file, const char* name, double value) {
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t attr = H5Acreate2(file, name, H5T_NATIVE_DOUBLE, space,
                             H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, H5T_NATIVE_DOUBLE, &value);
    H5Aclose(attr);
    H5Sclose(space);
}

int read_int_attr(hid_t file, const char* name) {
    int value = 0;
    hid_t attr = H5Aopen(file, name, H5P_DEFAULT);
    if (attr >= 0) {
        H5Aread(attr, H5T_NATIVE_INT, &value);
        H5Aclose(attr);
    }
    return value;
}

double read_double_attr(hid_t file, const char* name) {
    double value = 0.0;
    hid_t attr = H5Aopen(file, name, H5P_DEFAULT);
    if (attr >= 0) {
        H5Aread(attr, H5T_NATIVE_DOUBLE, &value);
        H5Aclose(attr);
    }
    return value;
}

} // anonymous namespace

void write_checkpoint(const std::string& filename,
                      const Mesh& mesh,
                      const VectorField& vel,
                      const ScalarField& pressure,
                      int step, double time, double dt) {
    hid_t file = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC,
                            H5P_DEFAULT, H5P_DEFAULT);
    if (file < 0) {
        throw std::runtime_error("Cannot create checkpoint file: " + filename);
    }

    // Write mesh dimensions as attributes
    write_int_attr(file, "Nx", mesh.Nx);
    write_int_attr(file, "Ny", mesh.Ny);
    write_int_attr(file, "Nz", mesh.Nz);
    write_int_attr(file, "Nghost", mesh.Nghost);
    write_int_attr(file, "step", step);
    write_double_attr(file, "time", time);
    write_double_attr(file, "dt", dt);

    // Write velocity components (including ghost cells for exact restart)
    const auto& u_data = vel.u_data();
    const auto& v_data = vel.v_data();
    write_scalar_dataset(file, "u", u_data.data(), u_data.size());
    write_scalar_dataset(file, "v", v_data.data(), v_data.size());

    if (!mesh.is2D()) {
        const auto& w_data = vel.w_data();
        write_scalar_dataset(file, "w", w_data.data(), w_data.size());
    }

    // Write pressure
    const auto& p_data = pressure.data();
    write_scalar_dataset(file, "pressure", p_data.data(), p_data.size());

    H5Fclose(file);
    std::cout << "[Checkpoint] Written to " << filename
              << " (step=" << step << ", time=" << time << ")\n";
}

bool read_checkpoint(const std::string& filename,
                     const Mesh& mesh,
                     VectorField& vel,
                     ScalarField& pressure,
                     int& step, double& time, double& dt) {
    hid_t file = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) {
        std::cerr << "[Checkpoint] Cannot open: " << filename << "\n";
        return false;
    }

    // Verify mesh dimensions
    int saved_Nx = read_int_attr(file, "Nx");
    int saved_Ny = read_int_attr(file, "Ny");
    int saved_Nz = read_int_attr(file, "Nz");

    if (saved_Nx != mesh.Nx || saved_Ny != mesh.Ny || saved_Nz != mesh.Nz) {
        std::cerr << "[Checkpoint] Mesh mismatch: saved " << saved_Nx << "x"
                  << saved_Ny << "x" << saved_Nz
                  << ", current " << mesh.Nx << "x" << mesh.Ny << "x" << mesh.Nz << "\n";
        H5Fclose(file);
        return false;
    }

    step = read_int_attr(file, "step");
    time = read_double_attr(file, "time");
    dt = read_double_attr(file, "dt");

    // Read velocity
    auto& u_data = vel.u_data();
    auto& v_data = vel.v_data();
    read_scalar_dataset(file, "u", u_data.data(), u_data.size());
    read_scalar_dataset(file, "v", v_data.data(), v_data.size());

    if (!mesh.is2D()) {
        auto& w_data = vel.w_data();
        read_scalar_dataset(file, "w", w_data.data(), w_data.size());
    }

    // Read pressure
    auto& p_data = pressure.data();
    read_scalar_dataset(file, "pressure", p_data.data(), p_data.size());

    H5Fclose(file);
    std::cout << "[Checkpoint] Loaded from " << filename
              << " (step=" << step << ", time=" << time << ")\n";
    return true;
}

#else // !USE_HDF5

void write_checkpoint(const std::string& /*filename*/,
                      const Mesh& /*mesh*/,
                      const VectorField& /*vel*/,
                      const ScalarField& /*pressure*/,
                      int /*step*/, double /*time*/, double /*dt*/) {
    std::cerr << "[Checkpoint] HDF5 not available, checkpoint disabled\n";
}

bool read_checkpoint(const std::string& /*filename*/,
                     const Mesh& /*mesh*/,
                     VectorField& /*vel*/,
                     ScalarField& /*pressure*/,
                     int& /*step*/, double& /*time*/, double& /*dt*/) {
    std::cerr << "[Checkpoint] HDF5 not available, restart disabled\n";
    return false;
}

#endif // USE_HDF5

} // namespace nncfd
