/// @file test_gpu_buffer.cpp
/// @brief Unit tests for GPUBuffer RAII wrapper

#include "gpu_buffer.hpp"
#include "test_harness.hpp"
#include <cmath>
#include <numeric>

using namespace nncfd;
using namespace nncfd::test::harness;

//=============================================================================
// Basic construction and sizing tests
//=============================================================================

static void test_construction() {
    // Default construction
    GPUBuffer<double> buf1;
    record("Default construction empty", buf1.empty());
    record("Default construction size 0", buf1.size() == 0);
    record("Default construction not mapped", !buf1.is_mapped());

    // Construction with size
    GPUBuffer<double> buf2(100);
    record("Sized construction not empty", !buf2.empty());
    record("Sized construction correct size", buf2.size() == 100);
    record("Sized construction not mapped", !buf2.is_mapped());

    // Data is accessible and zero-initialized
    bool all_zero = true;
    for (size_t i = 0; i < buf2.size(); ++i) {
        if (buf2[i] != 0.0) all_zero = false;
    }
    record("Sized construction zero-initialized", all_zero);
}

static void test_resize() {
    GPUBuffer<double> buf(50);
    record("Initial size 50", buf.size() == 50);

    buf.resize(100);
    record("Resize to 100", buf.size() == 100);

    buf.resize(25);
    record("Resize to 25", buf.size() == 25);

    buf.resize(0);
    record("Resize to 0 makes empty", buf.empty());
}

static void test_clear() {
    GPUBuffer<double> buf(100);
    buf.clear();
    record("Clear makes empty", buf.empty());
    record("Clear sets size 0", buf.size() == 0);
}

//=============================================================================
// Move semantics tests
//=============================================================================

static void test_move_construction() {
    GPUBuffer<double> buf1(100);
    for (size_t i = 0; i < buf1.size(); ++i) {
        buf1[i] = static_cast<double>(i);
    }

    GPUBuffer<double> buf2(std::move(buf1));

    record("Move ctor: source empty", buf1.empty());
    record("Move ctor: dest has size", buf2.size() == 100);
    record("Move ctor: data preserved", buf2[50] == 50.0);
}

static void test_move_assignment() {
    GPUBuffer<double> buf1(100);
    buf1[0] = 42.0;

    GPUBuffer<double> buf2;
    buf2 = std::move(buf1);

    record("Move assign: source empty", buf1.empty());
    record("Move assign: dest has size", buf2.size() == 100);
    record("Move assign: data preserved", buf2[0] == 42.0);
}

//=============================================================================
// GPU mapping tests (work on both CPU and GPU builds)
//=============================================================================

static void test_mapping() {
    GPUBuffer<double> buf(100);

    record("Initially not mapped", !buf.is_mapped());

    buf.map_to_device();
#ifdef USE_GPU_OFFLOAD
    record("map_to_device sets mapped flag", buf.is_mapped());
#else
    record("map_to_device no-op on CPU (not mapped)", !buf.is_mapped());
#endif

    buf.unmap();
    record("unmap clears mapped flag", !buf.is_mapped());
}

static void test_data_transfer() {
    GPUBuffer<double> buf(100);

    // Initialize data
    for (size_t i = 0; i < buf.size(); ++i) {
        buf[i] = static_cast<double>(i * 2);
    }

    buf.map_to_device_with_data();

    // Modify on host
    buf[0] = 999.0;
    buf.update_to_device();

    // These are no-ops on CPU, but shouldn't crash
    buf.update_from_device();

    buf.unmap();

    // Data should still be accessible after unmap
    record("Data accessible after unmap", buf[0] == 999.0);
}

static void test_empty_buffer_operations() {
    GPUBuffer<double> buf;

    // These should be safe no-ops on empty buffers
    buf.map_to_device();
    buf.update_to_device();
    buf.update_from_device();
    buf.unmap();

    record("Empty buffer operations safe", buf.empty());
}

//=============================================================================
// Iterator and vector compatibility tests
//=============================================================================

static void test_iterators() {
    GPUBuffer<double> buf(10);

    // Fill using iterators
    double val = 0.0;
    for (auto& x : buf) {
        x = val++;
    }

    // Sum using iterators
    double sum = 0.0;
    for (const auto& x : buf) {
        sum += x;
    }

    record("Iterator range-for works", std::abs(sum - 45.0) < 1e-10);
}

static void test_vector_access() {
    GPUBuffer<double> buf(10);
    buf[0] = 1.0;
    buf[9] = 2.0;

    // Access underlying vector
    std::vector<double>& vec = buf.vector();
    vec[5] = 3.0;

    record("Vector access works", buf[5] == 3.0);
}

//=============================================================================
// GPUBufferGroup tests
//=============================================================================

static void test_buffer_group() {
    GPUBuffer<double> buf1(100);
    GPUBuffer<double> buf2(200);
    GPUBuffer<float> buf3(50);

    GPUBufferGroup group;
    record("Group initially empty", group.empty());

    group.add(buf1);
    group.add(buf2);
    group.add(buf3);
    record("Group has buffers after add", !group.empty());

    // Map all - should work on both CPU and GPU builds
    group.map_all();

#ifdef USE_GPU_OFFLOAD
    record("Group map_all maps buffers", buf1.is_mapped() && buf2.is_mapped());
#else
    record("Group map_all no-op on CPU", !buf1.is_mapped() && !buf2.is_mapped());
#endif

    group.unmap_all();
    record("Group unmap_all unmaps all", !buf1.is_mapped() && !buf2.is_mapped());

    group.clear();
    record("Group clear empties group", group.empty());
}

//=============================================================================
// Stress tests
//=============================================================================

static void test_repeated_map_unmap() {
    GPUBuffer<double> buf(1000);

    for (int i = 0; i < 10; ++i) {
        buf.map_to_device();
        buf.update_to_device();
        buf.update_from_device();
        buf.unmap();
    }

    record("Repeated map/unmap cycles safe", true);
}

static void test_resize_while_mapped() {
    GPUBuffer<double> buf(100);
    buf.map_to_device();

    // Resize should unmap first
    buf.resize(200);
    record("Resize unmaps buffer", !buf.is_mapped());
    record("Resize changes size", buf.size() == 200);

    buf.map_to_device();
    buf.clear();
    record("Clear unmaps buffer", !buf.is_mapped());
}

//=============================================================================
// Main
//=============================================================================

int main() {
    return run_sections("GPUBuffer RAII Wrapper Tests", {
        {"Construction", test_construction},
        {"Resize", test_resize},
        {"Clear", test_clear},
        {"Move Construction", test_move_construction},
        {"Move Assignment", test_move_assignment},
        {"GPU Mapping", test_mapping},
        {"Data Transfer", test_data_transfer},
        {"Empty Buffer Operations", test_empty_buffer_operations},
        {"Iterators", test_iterators},
        {"Vector Access", test_vector_access},
        {"Buffer Group", test_buffer_group},
        {"Repeated Map/Unmap", test_repeated_map_unmap},
        {"Resize While Mapped", test_resize_while_mapped}
    });
}
