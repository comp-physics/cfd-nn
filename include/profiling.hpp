#pragma once

/// @file profiling.hpp
/// @brief Comprehensive NVTX profiling infrastructure for kernel-level performance analysis
///
/// This header provides color-coded NVTX ranges for granular GPU profiling.
/// Use with nsys (NVIDIA Nsight Systems) for timeline analysis:
///   nsys profile -t nvtx,cuda ./your_binary
///
/// Categories:
///   - SOLVER (blue): Main solver operations
///   - KERNEL (green): Individual GPU kernels
///   - TURB (yellow): Turbulence model operations
///   - POISSON (cyan): Poisson solver iterations
///   - MEMORY (magenta): Data transfers and copies
///   - BC (orange): Boundary condition application

#include <cstdint>

// ============================================================================
// NVTX Configuration and Includes
// ============================================================================

#ifdef GPU_PROFILE_KERNELS

#if __has_include(<nvtx3/nvToolsExt.h>)
    #include <nvtx3/nvToolsExt.h>
    #define NVTX_AVAILABLE 1
#elif __has_include(<nvToolsExt.h>)
    #include <nvToolsExt.h>
    #define NVTX_AVAILABLE 1
#else
    #define NVTX_AVAILABLE 0
#endif

#if NVTX_AVAILABLE

// ============================================================================
// Color Definitions (ARGB format)
// ============================================================================
namespace nvtx_colors {
    // Main category colors
    constexpr uint32_t SOLVER  = 0xFF3366FF;  // Blue - main solver steps
    constexpr uint32_t KERNEL  = 0xFF33CC33;  // Green - GPU kernels
    constexpr uint32_t TURB    = 0xFFFFCC00;  // Yellow - turbulence
    constexpr uint32_t POISSON = 0xFF00CCCC;  // Cyan - Poisson solver
    constexpr uint32_t MEMORY  = 0xFFCC33CC;  // Magenta - memory operations
    constexpr uint32_t BC      = 0xFFFF6600;  // Orange - boundary conditions

    // Sub-category colors (lighter variants)
    constexpr uint32_t CONVECT  = 0xFF66FF66;  // Light green - convection
    constexpr uint32_t DIFFUSE  = 0xFF99FF99;  // Lighter green - diffusion
    constexpr uint32_t GRADIENT = 0xFF99CC99;  // Sage - gradients
    constexpr uint32_t NN       = 0xFFFFFF66;  // Light yellow - neural network
    constexpr uint32_t CLOSURE  = 0xFFFFCC66;  // Light orange - turbulence closure
    constexpr uint32_t MG_LEVEL = 0xFF66CCCC;  // Light cyan - multigrid levels
    constexpr uint32_t RESIDUAL = 0xFF99CCCC;  // Lighter cyan - residual computation
}

// ============================================================================
// NVTX Colored Range Macros
// ============================================================================

// Helper to create colored event attributes
inline nvtxEventAttributes_t make_nvtx_attr(const char* name, uint32_t color) {
    nvtxEventAttributes_t attr = {0};
    attr.version = NVTX_VERSION;
    attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attr.colorType = NVTX_COLOR_ARGB;
    attr.color = color;
    attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
    attr.message.ascii = name;
    return attr;
}

// Main push with color
#define NVTX_PUSH_COLOR(name, color) do { \
    nvtxEventAttributes_t attr = make_nvtx_attr(name, color); \
    nvtxRangePushEx(&attr); \
} while(0)

#define NVTX_POP_PROFILE() nvtxRangePop()

// Category-specific push macros
#define NVTX_SOLVER(name)   NVTX_PUSH_COLOR(name, nvtx_colors::SOLVER)
#define NVTX_KERNEL(name)   NVTX_PUSH_COLOR(name, nvtx_colors::KERNEL)
#define NVTX_TURB(name)     NVTX_PUSH_COLOR(name, nvtx_colors::TURB)
#define NVTX_POISSON(name)  NVTX_PUSH_COLOR(name, nvtx_colors::POISSON)
#define NVTX_MEMORY(name)   NVTX_PUSH_COLOR(name, nvtx_colors::MEMORY)
#define NVTX_BC(name)       NVTX_PUSH_COLOR(name, nvtx_colors::BC)

// Sub-category push macros
#define NVTX_CONVECT(name)  NVTX_PUSH_COLOR(name, nvtx_colors::CONVECT)
#define NVTX_DIFFUSE(name)  NVTX_PUSH_COLOR(name, nvtx_colors::DIFFUSE)
#define NVTX_GRADIENT(name) NVTX_PUSH_COLOR(name, nvtx_colors::GRADIENT)
#define NVTX_NN(name)       NVTX_PUSH_COLOR(name, nvtx_colors::NN)
#define NVTX_CLOSURE(name)  NVTX_PUSH_COLOR(name, nvtx_colors::CLOSURE)
#define NVTX_MG(name)       NVTX_PUSH_COLOR(name, nvtx_colors::MG_LEVEL)
#define NVTX_RESIDUAL(name) NVTX_PUSH_COLOR(name, nvtx_colors::RESIDUAL)

// ============================================================================
// RAII Scope Guards for Automatic Pop
// ============================================================================

struct NvtxColoredScope {
    explicit NvtxColoredScope(const char* name, uint32_t color) {
        nvtxEventAttributes_t attr = make_nvtx_attr(name, color);
        nvtxRangePushEx(&attr);
    }
    ~NvtxColoredScope() { nvtxRangePop(); }
    NvtxColoredScope(const NvtxColoredScope&) = delete;
    NvtxColoredScope& operator=(const NvtxColoredScope&) = delete;
};

// Unique variable name helper
#define NVTX_CONCAT_(a, b) a##b
#define NVTX_CONCAT(a, b) NVTX_CONCAT_(a, b)
#define NVTX_UNIQUE_VAR NVTX_CONCAT(nvtx_scope_, __LINE__)

// Scoped category macros (RAII - automatically pops when scope exits)
#define NVTX_SCOPE_SOLVER(name)   NvtxColoredScope NVTX_UNIQUE_VAR(name, nvtx_colors::SOLVER)
#define NVTX_SCOPE_KERNEL(name)   NvtxColoredScope NVTX_UNIQUE_VAR(name, nvtx_colors::KERNEL)
#define NVTX_SCOPE_TURB(name)     NvtxColoredScope NVTX_UNIQUE_VAR(name, nvtx_colors::TURB)
#define NVTX_SCOPE_POISSON(name)  NvtxColoredScope NVTX_UNIQUE_VAR(name, nvtx_colors::POISSON)
#define NVTX_SCOPE_MEMORY(name)   NvtxColoredScope NVTX_UNIQUE_VAR(name, nvtx_colors::MEMORY)
#define NVTX_SCOPE_BC(name)       NvtxColoredScope NVTX_UNIQUE_VAR(name, nvtx_colors::BC)
#define NVTX_SCOPE_CONVECT(name)  NvtxColoredScope NVTX_UNIQUE_VAR(name, nvtx_colors::CONVECT)
#define NVTX_SCOPE_DIFFUSE(name)  NvtxColoredScope NVTX_UNIQUE_VAR(name, nvtx_colors::DIFFUSE)
#define NVTX_SCOPE_GRADIENT(name) NvtxColoredScope NVTX_UNIQUE_VAR(name, nvtx_colors::GRADIENT)
#define NVTX_SCOPE_NN(name)       NvtxColoredScope NVTX_UNIQUE_VAR(name, nvtx_colors::NN)
#define NVTX_SCOPE_CLOSURE(name)  NvtxColoredScope NVTX_UNIQUE_VAR(name, nvtx_colors::CLOSURE)
#define NVTX_SCOPE_MG(name)       NvtxColoredScope NVTX_UNIQUE_VAR(name, nvtx_colors::MG_LEVEL)
#define NVTX_SCOPE_RESIDUAL(name) NvtxColoredScope NVTX_UNIQUE_VAR(name, nvtx_colors::RESIDUAL)

// ============================================================================
// Iteration Markers (for tracking iterations in profiler)
// ============================================================================

#define NVTX_MARK(name) nvtxMarkA(name)

inline void nvtx_mark_iteration(const char* prefix, int iter) {
    char buf[64];
    snprintf(buf, sizeof(buf), "%s_iter_%d", prefix, iter);
    nvtxMarkA(buf);
}

#define NVTX_ITERATION(prefix, iter) nvtx_mark_iteration(prefix, iter)

#else // !NVTX_AVAILABLE

// ============================================================================
// Stub Implementations (when NVTX headers not available)
// ============================================================================

#define NVTX_PUSH_COLOR(name, color)
#define NVTX_POP_PROFILE()
#define NVTX_SOLVER(name)
#define NVTX_KERNEL(name)
#define NVTX_TURB(name)
#define NVTX_POISSON(name)
#define NVTX_MEMORY(name)
#define NVTX_BC(name)
#define NVTX_CONVECT(name)
#define NVTX_DIFFUSE(name)
#define NVTX_GRADIENT(name)
#define NVTX_NN(name)
#define NVTX_CLOSURE(name)
#define NVTX_MG(name)
#define NVTX_RESIDUAL(name)

struct NvtxColoredScope {
    explicit NvtxColoredScope(const char*, uint32_t) {}
};

#define NVTX_SCOPE_SOLVER(name)
#define NVTX_SCOPE_KERNEL(name)
#define NVTX_SCOPE_TURB(name)
#define NVTX_SCOPE_POISSON(name)
#define NVTX_SCOPE_MEMORY(name)
#define NVTX_SCOPE_BC(name)
#define NVTX_SCOPE_CONVECT(name)
#define NVTX_SCOPE_DIFFUSE(name)
#define NVTX_SCOPE_GRADIENT(name)
#define NVTX_SCOPE_NN(name)
#define NVTX_SCOPE_CLOSURE(name)
#define NVTX_SCOPE_MG(name)
#define NVTX_SCOPE_RESIDUAL(name)

#define NVTX_MARK(name)
#define NVTX_ITERATION(prefix, iter)

#endif // NVTX_AVAILABLE

#else // !GPU_PROFILE_KERNELS

// ============================================================================
// No-op when profiling disabled
// ============================================================================

#define NVTX_PUSH_COLOR(name, color)
#define NVTX_POP_PROFILE()
#define NVTX_SOLVER(name)
#define NVTX_KERNEL(name)
#define NVTX_TURB(name)
#define NVTX_POISSON(name)
#define NVTX_MEMORY(name)
#define NVTX_BC(name)
#define NVTX_CONVECT(name)
#define NVTX_DIFFUSE(name)
#define NVTX_GRADIENT(name)
#define NVTX_NN(name)
#define NVTX_CLOSURE(name)
#define NVTX_MG(name)
#define NVTX_RESIDUAL(name)

struct NvtxColoredScope {
    explicit NvtxColoredScope(const char*, uint32_t) {}
};

#define NVTX_SCOPE_SOLVER(name)
#define NVTX_SCOPE_KERNEL(name)
#define NVTX_SCOPE_TURB(name)
#define NVTX_SCOPE_POISSON(name)
#define NVTX_SCOPE_MEMORY(name)
#define NVTX_SCOPE_BC(name)
#define NVTX_SCOPE_CONVECT(name)
#define NVTX_SCOPE_DIFFUSE(name)
#define NVTX_SCOPE_GRADIENT(name)
#define NVTX_SCOPE_NN(name)
#define NVTX_SCOPE_CLOSURE(name)
#define NVTX_SCOPE_MG(name)
#define NVTX_SCOPE_RESIDUAL(name)

#define NVTX_MARK(name)
#define NVTX_ITERATION(prefix, iter)

#endif // GPU_PROFILE_KERNELS
