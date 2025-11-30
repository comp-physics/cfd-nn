#!/bin/bash
#SBATCH --job-name=cpu_vs_gpu
#SBATCH --output=benchmark_%j.out
#SBATCH --error=benchmark_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:H200:1
#SBATCH --time=00:30:00
#SBATCH --account=gts-sbryngelson3
#SBATCH --qos=embers

cd /storage/home/hcoda1/6/sbryngelson3/cfd-nn
module reset 2>/dev/null
module load nvhpc

echo "============================================================"
echo " CPU vs GPU BENCHMARK - With Adaptive Time Stepping"
echo "============================================================"
echo ""

# Build CPU version
echo "Building CPU-only version..."
rm -rf build_cpu
mkdir -p build_cpu && cd build_cpu
cmake .. -DUSE_GPU_OFFLOAD=OFF -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1
make -j > /dev/null 2>&1
cd ..
echo "CPU build complete."

# Build GPU version
echo "Building GPU-accelerated version..."
rm -rf build_gpu
mkdir -p build_gpu && cd build_gpu
cmake .. -DUSE_GPU_OFFLOAD=ON -DCMAKE_BUILD_TYPE=Release > /dev/null 2>&1
make -j > /dev/null 2>&1
cd ..
echo "GPU build complete."
echo ""

# Print header
printf "%-12s | %-10s | %-12s | %-12s | %-10s\n" "Grid" "Cells" "CPU Time" "GPU Time" "Speedup"
printf "%-12s-+-%-10s-+-%-12s-+-%-12s-+-%-10s\n" "------------" "----------" "------------" "------------" "----------"

# Test all grid sizes with both CPU and GPU
for size in "64 128" "128 256" "256 512" "512 1024" "1024 2048"; do
    read nx ny <<< "$size"
    cells=$((nx * ny))
    
    # Run CPU version and extract timing (format: "nn_tbnn_inference_cpu    0.155    5    30.942")
    cpu_output=$(./build_cpu/channel --Nx $nx --Ny $ny --nu 0.01 --max_iter 100 --model nn_tbnn --nn_preset example_tbnn 2>&1)
    cpu_time=$(echo "$cpu_output" | grep "nn_tbnn_inference" | awk '{print $2}' | head -1)
    
    # Run GPU version and extract timing
    gpu_output=$(./build_gpu/channel --Nx $nx --Ny $ny --nu 0.01 --max_iter 100 --model nn_tbnn --nn_preset example_tbnn 2>&1)
    gpu_time=$(echo "$gpu_output" | grep "nn_tbnn_inference" | awk '{print $2}' | head -1)
    
    # Calculate speedup (times are in seconds)
    if [[ -n "$cpu_time" && -n "$gpu_time" && "$gpu_time" != "0" ]]; then
        # Convert to ms for display
        cpu_ms=$(echo "scale=3; $cpu_time * 1000" | bc)
        gpu_ms=$(echo "scale=3; $gpu_time * 1000" | bc)
        speedup=$(echo "scale=2; $cpu_time / $gpu_time" | bc)
        printf "%-12s | %-10s | %10.1f ms | %10.1f ms | %10.2fx\n" "${nx}x${ny}" "$cells" "$cpu_ms" "$gpu_ms" "$speedup"
    else
        printf "%-12s | %-10s | %12s | %12s | %10s\n" "${nx}x${ny}" "$cells" "${cpu_time:-N/A}" "${gpu_time:-N/A}" "N/A"
    fi
done

# Cleanup
rm -rf build_cpu build_gpu

echo "Benchmark complete!"

