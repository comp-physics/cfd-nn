#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
# Modify the analysis to use GPU output directory
exec(open('convergence_analysis.py').read().replace(
    'output_dir = script_dir / "output"',
    'output_dir = script_dir / "output_gpu"'
).replace(
    'output_dir / "convergence_analysis.png"',
    'output_dir / "convergence_analysis_gpu_partial.png"'
))
