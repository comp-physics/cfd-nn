#!/usr/bin/env python3
"""
Compare a posteriori QoI results against reference data.
Reads production sweep output and computes errors vs literature values.

Reference sources:
  - Cylinder Re=100: Tritton (1959), Williamson (1996), Park et al. (1998)
  - Sphere Re=200: Johnson & Patel (1999), JFM 378:19-70
  - Hills Re_H=10595: Breuer et al. (2009), Comp. & Fluids 38:433-457
  - Duct Re_b=3500: Pinelli et al. (2010), JFM 644:107-122

Usage:
    python3 scripts/paper/compare_qoi_reference.py [results_dir]
    Default results_dir: results/paper/production_sweep
"""

import os
import sys
import glob
import re
import numpy as np

# ============================================================================
# Reference values
# ============================================================================

REF = {
    'cylinder_re100': {
        'Cd': 1.35,        # Tritton (1959), Williamson (1996)
        'St': 0.165,       # Park et al. (1998), Williamson (1996)
        'source': 'Tritton (1959), Williamson (1996), Park et al. (1998)',
    },
    'sphere_re200': {
        'Cd': 0.77,        # Johnson & Patel (1999)
        'sep_angle': 117.0, # Johnson & Patel (1999), degrees from stagnation
        'source': 'Johnson & Patel (1999)',
    },
    'hills_re10595': {
        'x_sep': 0.22,     # Breuer et al. (2009), x/H
        'x_re': 4.72,      # Breuer et al. (2009), x/H
        'source': 'Breuer et al. (2009)',
    },
    'duct_reb3500': {
        'v_perp_max': 0.020,  # Pinelli et al. (2010), fraction of U_b
        'v_perp_rms': 0.010,  # Pinelli et al. (2010), fraction of U_b
        'source': 'Pinelli et al. (2010)',
    },
}


# ============================================================================
# Parsing functions
# ============================================================================

def parse_qoi_summary(path):
    """Parse qoi_summary.dat → dict of {key: float}."""
    result = {}
    if not os.path.exists(path):
        return result
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    result[parts[0]] = float(parts[1])
                except ValueError:
                    pass
    return result


def parse_output_log(path):
    """Parse output.log for separation/reattachment (hills), residual, timing."""
    result = {}
    if not os.path.exists(path):
        return result
    with open(path) as f:
        text = f.read()

    # Separation point (hills)
    m = re.search(r'Separation:\s+x/H\s*=\s*([\d.eE+-]+)', text)
    if m:
        result['x_sep'] = float(m.group(1))

    # Reattachment point (hills)
    m = re.search(r'Reattachment:\s+x/H\s*=\s*([\d.eE+-]+)', text)
    if m:
        result['x_re'] = float(m.group(1))

    # Final residual
    residuals = re.findall(r'res=([\d.eE+-]+)', text)
    if residuals:
        result['final_res'] = float(residuals[-1])

    # Timing
    m = re.search(r'^solver_step\s+([\d.]+)\s+(\d+)\s+([\d.]+)', text, re.MULTILINE)
    if m:
        result['wall_total'] = float(m.group(1))
        result['steps'] = int(m.group(2))
        result['ms_per_step'] = float(m.group(3))

    # Divergence
    if 'STOPPING' in text or 'diverged' in text:
        result['diverged'] = True

    return result


def parse_forces_dat(path):
    """Parse forces.dat → arrays of step, time, Fx, Fy, Cd, Cl."""
    if not os.path.exists(path):
        return None
    data = []
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 6:
                try:
                    data.append([float(x) for x in parts[:6]])
                except ValueError:
                    continue
    if not data:
        return None
    arr = np.array(data)
    return {
        'step': arr[:, 0],
        'time': arr[:, 1],
        'Fx': arr[:, 2],
        'Fy': arr[:, 3],
        'Cd': arr[:, 4],
        'Cl': arr[:, 5],
    }


def compute_strouhal_from_forces(forces, diameter=1.0, U_inf=1.0):
    """Compute Strouhal from Cl zero-crossings in second half of time series."""
    if forces is None or len(forces['Cl']) < 10:
        return None
    N = len(forces['Cl'])
    start = N // 2
    cl = forces['Cl'][start:]
    t = forces['time'][start:]

    crossings = []
    for i in range(1, len(cl)):
        if cl[i - 1] < 0.0 and cl[i] >= 0.0:
            frac = -cl[i - 1] / (cl[i] - cl[i - 1])
            t_cross = t[i - 1] + frac * (t[i] - t[i - 1])
            crossings.append(t_cross)

    if len(crossings) < 2:
        return None

    periods = [crossings[i + 1] - crossings[i] for i in range(len(crossings) - 1)]
    avg_period = np.mean(periods)
    if avg_period <= 0:
        return None
    freq = 1.0 / avg_period
    return freq * diameter / U_inf


def parse_duct_cross_section(path):
    """Parse duct_cross_section.dat → secondary flow magnitude."""
    if not os.path.exists(path):
        return None
    data = []
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 5:
                try:
                    data.append([float(x) for x in parts[:5]])
                except ValueError:
                    continue
    if not data:
        return None
    arr = np.array(data)
    # Columns: y, z, u, v, w
    v = arr[:, 3]
    w = arr[:, 4]
    v_perp = np.sqrt(v ** 2 + w ** 2)
    return {
        'v_perp_max': float(np.max(v_perp)),
        'v_perp_rms': float(np.sqrt(np.mean(v_perp ** 2))),
    }


# ============================================================================
# Error computation
# ============================================================================

def relative_error(measured, reference):
    """Compute relative error |measured - reference| / |reference|."""
    if reference == 0:
        return abs(measured) if measured != 0 else 0.0
    return abs(measured - reference) / abs(reference)


def compare_case(case, model_dir, ref):
    """Compare one model's QoI against reference for one case."""
    result = {'model': os.path.basename(model_dir), 'case': case}

    qoi = parse_qoi_summary(os.path.join(model_dir, 'qoi', 'qoi_summary.dat'))
    log = parse_output_log(os.path.join(model_dir, 'output.log'))
    result.update(log)

    if result.get('diverged'):
        result['status'] = 'DIVERGED'
        return result

    result['status'] = 'OK'

    if case == 'cylinder_re100':
        # Cd from qoi_summary
        if 'Cd_mean' in qoi and qoi['Cd_mean'] != 0:
            result['Cd'] = qoi['Cd_mean']
            result['Cd_err'] = relative_error(qoi['Cd_mean'], ref['Cd'])

        # St from qoi_summary or recompute from forces
        if 'St' in qoi and qoi['St'] > 0:
            result['St'] = qoi['St']
            result['St_err'] = relative_error(qoi['St'], ref['St'])
        else:
            forces = parse_forces_dat(os.path.join(model_dir, 'forces.dat'))
            st = compute_strouhal_from_forces(forces)
            if st is not None:
                result['St'] = st
                result['St_err'] = relative_error(st, ref['St'])

    elif case == 'sphere_re200':
        if 'Cd_mean' in qoi and qoi['Cd_mean'] != 0:
            result['Cd'] = qoi['Cd_mean']
            result['Cd_err'] = relative_error(qoi['Cd_mean'], ref['Cd'])
        if 'sep_angle' in qoi and qoi['sep_angle'] > 0:
            result['sep_angle'] = qoi['sep_angle']
            result['sep_err'] = relative_error(qoi['sep_angle'], ref['sep_angle'])

    elif case == 'hills_re10595':
        if 'x_sep' in log:
            result['x_sep'] = log['x_sep']
            result['x_sep_err'] = relative_error(log['x_sep'], ref['x_sep'])
        if 'x_re' in log:
            result['x_re'] = log['x_re']
            result['x_re_err'] = relative_error(log['x_re'], ref['x_re'])

    elif case == 'duct_reb3500':
        cs = parse_duct_cross_section(
            os.path.join(model_dir, 'qoi', 'duct_cross_section.dat'))
        if cs:
            result['v_perp_max'] = cs['v_perp_max']
            result['v_perp_rms'] = cs['v_perp_rms']
            result['v_perp_max_err'] = relative_error(cs['v_perp_max'], ref['v_perp_max'])

    return result


# ============================================================================
# Main
# ============================================================================

def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else 'results/paper/production_sweep'

    if not os.path.isdir(results_dir):
        # Try validation_grid as fallback
        results_dir = 'results/paper/validation_grid'

    if not os.path.isdir(results_dir):
        print(f"ERROR: results directory not found: {results_dir}")
        sys.exit(1)

    print(f"Reading results from: {results_dir}")
    print(f"{'=' * 80}")

    for case, ref in REF.items():
        case_dir = os.path.join(results_dir, case)
        if not os.path.isdir(case_dir):
            print(f"\n=== {case} === SKIP (no data)")
            continue

        print(f"\n=== {case} ===")
        print(f"Reference: {ref['source']}")

        # Print header based on case
        if case == 'cylinder_re100':
            print(f"  Ref: Cd={ref['Cd']}, St={ref['St']}")
            print(f"  {'Model':<16s} {'Cd':>8s} {'Cd_err':>8s} {'St':>8s} {'St_err':>8s} {'res':>12s} {'status':>8s}")
        elif case == 'sphere_re200':
            print(f"  Ref: Cd={ref['Cd']}, sep={ref['sep_angle']}°")
            print(f"  {'Model':<16s} {'Cd':>8s} {'Cd_err':>8s} {'sep°':>8s} {'sep_err':>8s} {'res':>12s} {'status':>8s}")
        elif case == 'hills_re10595':
            print(f"  Ref: x_sep/H={ref['x_sep']}, x_re/H={ref['x_re']}")
            print(f"  {'Model':<16s} {'x_sep':>8s} {'sep_err':>8s} {'x_re':>8s} {'re_err':>8s} {'res':>12s} {'status':>8s}")
        elif case == 'duct_reb3500':
            print(f"  Ref: v_perp_max={ref['v_perp_max']} U_b")
            print(f"  {'Model':<16s} {'v_max':>8s} {'v_rms':>8s} {'err':>8s} {'res':>12s} {'status':>8s}")

        model_dirs = sorted(glob.glob(os.path.join(case_dir, '*')))
        for md in model_dirs:
            if not os.path.isdir(md):
                continue
            r = compare_case(case, md, ref)

            if case == 'cylinder_re100':
                print(f"  {r['model']:<16s} "
                      f"{r.get('Cd', '---'):>8.4f} " if isinstance(r.get('Cd'), float) else f"  {r['model']:<16s} {'---':>8s} ",
                      end='')
                cd_err = r.get('Cd_err')
                print(f"{cd_err:>7.1%} " if cd_err is not None else f"{'---':>8s} ", end='')
                st = r.get('St')
                print(f"{st:>8.4f} " if st is not None else f"{'---':>8s} ", end='')
                st_err = r.get('St_err')
                print(f"{st_err:>7.1%} " if st_err is not None else f"{'---':>8s} ", end='')
                print(f"{r.get('final_res', '---'):>12s} " if isinstance(r.get('final_res'), str) else
                      f"{r.get('final_res', 0):>12.3e} ", end='')
                print(f"{r['status']:>8s}")

            elif case == 'sphere_re200':
                cd = r.get('Cd')
                cd_s = f"{cd:>8.4f}" if isinstance(cd, float) else f"{'---':>8s}"
                cd_err = r.get('Cd_err')
                cde_s = f"{cd_err:>7.1%}" if cd_err is not None else f"{'---':>8s}"
                sep = r.get('sep_angle')
                sep_s = f"{sep:>8.1f}" if isinstance(sep, float) else f"{'---':>8s}"
                sep_err = r.get('sep_err')
                sepe_s = f"{sep_err:>7.1%}" if sep_err is not None else f"{'---':>8s}"
                res = r.get('final_res')
                res_s = f"{res:>12.3e}" if isinstance(res, float) else f"{'---':>12s}"
                print(f"  {r['model']:<16s} {cd_s} {cde_s} {sep_s} {sepe_s} {res_s} {r['status']:>8s}")

            elif case == 'hills_re10595':
                xs = r.get('x_sep')
                xs_s = f"{xs:>8.3f}" if isinstance(xs, float) else f"{'---':>8s}"
                xse = r.get('x_sep_err')
                xse_s = f"{xse:>7.1%}" if xse is not None else f"{'---':>8s}"
                xr = r.get('x_re')
                xr_s = f"{xr:>8.3f}" if isinstance(xr, float) else f"{'---':>8s}"
                xre = r.get('x_re_err')
                xre_s = f"{xre:>7.1%}" if xre is not None else f"{'---':>8s}"
                res = r.get('final_res')
                res_s = f"{res:>12.3e}" if isinstance(res, float) else f"{'---':>12s}"
                print(f"  {r['model']:<16s} {xs_s} {xse_s} {xr_s} {xre_s} {res_s} {r['status']:>8s}")

            elif case == 'duct_reb3500':
                vm = r.get('v_perp_max')
                vm_s = f"{vm:>8.4f}" if isinstance(vm, float) else f"{'---':>8s}"
                vr = r.get('v_perp_rms')
                vr_s = f"{vr:>8.4f}" if isinstance(vr, float) else f"{'---':>8s}"
                vme = r.get('v_perp_max_err')
                vme_s = f"{vme:>7.1%}" if vme is not None else f"{'---':>8s}"
                res = r.get('final_res')
                res_s = f"{res:>12.3e}" if isinstance(res, float) else f"{'---':>12s}"
                print(f"  {r['model']:<16s} {vm_s} {vr_s} {vme_s} {res_s} {r['status']:>8s}")

    print(f"\n{'=' * 80}")
    print("Done.")


if __name__ == '__main__':
    main()
