#!/bin/bash
# Quick kernel check for Phase 3 profiles

module reset
module load nvhpc/25.5

cd /storage/home/hcoda1/6/sbryngelson3/cfd-nn/build_phase3_complete

echo "=== Checking k-ω profile for GPU kernels ==="
echo ""
echo "Looking for komega_transport_step_gpu:"
nsys stats --report cuda_gpu_kern_sum phase3_komega_512x512.sqlite 2>/dev/null | grep -i "komega_transport_step_gpu" || echo "❌ NOT FOUND"

echo ""
echo "Looking for compute_boussinesq_closure_gpu:"
nsys stats --report cuda_gpu_kern_sum phase3_komega_512x512.sqlite 2>/dev/null | grep -i "compute_boussinesq_closure_gpu" || echo "❌ NOT FOUND"

echo ""
echo "Looking for compute_gradients_from_mac_gpu:"
nsys stats --report cuda_gpu_kern_sum phase3_komega_512x512.sqlite 2>/dev/null | grep -i "compute_gradients_from_mac_gpu" || echo "❌ NOT FOUND"

echo ""
echo "=== Top 20 GPU kernels in k-ω profile ==="
nsys stats --report cuda_gpu_kern_sum phase3_komega_512x512.sqlite 2>/dev/null | head -25

echo ""
echo "=== Checking SST profile for GPU kernels ==="
echo ""
echo "Looking for compute_sst_closure_gpu:"
nsys stats --report cuda_gpu_kern_sum phase3_sst_512x512.sqlite 2>/dev/null | grep -i "compute_sst_closure_gpu" || echo "❌ NOT FOUND"

echo ""
echo "=== Top 20 GPU kernels in SST profile ==="
nsys stats --report cuda_gpu_kern_sum phase3_sst_512x512.sqlite 2>/dev/null | head -25


