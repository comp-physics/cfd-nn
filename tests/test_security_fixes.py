#!/usr/bin/env python3
"""
Test suite for security fixes:
1. Normalization statistics validation
2. Zip Slip protection
"""

import sys
import os
import tempfile
import zipfile
import pathlib
import numpy as np

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from safe_extract import safe_extract
from fix_normalization_stats import check_stats_validity


def test_normalization_validation():
    """Test that normalization stats validation works."""
    print("\n" + "="*70)
    print("TEST: Normalization Statistics Validation")
    print("="*70)
    
    # Test valid stats
    valid_means = np.array([0.1, 0.2, 0.3])
    valid_stds = np.array([0.5, 0.6, 0.7])
    issues = check_stats_validity(valid_means, valid_stds)
    assert len(issues) == 0, f"Valid stats flagged as invalid: {issues}"
    print("✓ Valid stats pass validation")
    
    # Test inf in stds
    invalid_stds = np.array([0.5, np.inf, 0.7])
    issues = check_stats_validity(valid_means, invalid_stds)
    assert len(issues) > 0, "Inf in stds not detected"
    print("✓ Inf in stds detected")
    
    # Test NaN in means
    invalid_means = np.array([0.1, np.nan, 0.3])
    issues = check_stats_validity(invalid_means, valid_stds)
    assert len(issues) > 0, "NaN in means not detected"
    print("✓ NaN in means detected")
    
    # Test extreme values
    extreme_means = np.array([1e15, 0.2, 0.3])
    issues = check_stats_validity(extreme_means, valid_stds)
    assert len(issues) > 0, "Extreme values not detected"
    print("✓ Extreme values detected")
    
    # Test zero/negative stds
    zero_stds = np.array([0.5, 0.0, -0.1])
    issues = check_stats_validity(valid_means, zero_stds)
    assert len(issues) > 0, "Zero/negative stds not detected"
    print("✓ Zero/negative stds detected")
    
    print("\n✓ All normalization validation tests passed!")
    return True


def test_zip_slip_protection():
    """Test that Zip Slip protection works."""
    print("\n" + "="*70)
    print("TEST: Zip Slip Protection")
    print("="*70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        
        # Test 1: Safe ZIP (should succeed)
        safe_zip = tmpdir / "safe.zip"
        with zipfile.ZipFile(safe_zip, 'w') as z:
            z.writestr("data/file1.txt", "content1")
            z.writestr("data/subdir/file2.txt", "content2")
        
        out_dir = tmpdir / "output_safe"
        try:
            safe_extract(safe_zip, out_dir, verbose=False)
            assert (out_dir / "data" / "file1.txt").exists()
            assert (out_dir / "data" / "subdir" / "file2.txt").exists()
            print("✓ Safe ZIP extracted successfully")
        except Exception as e:
            print(f"✗ Safe ZIP failed: {e}")
            return False
        
        # Test 2: ZIP with parent traversal (should fail)
        traversal_zip = tmpdir / "traversal.zip"
        with zipfile.ZipFile(traversal_zip, 'w') as z:
            z.writestr("data/file1.txt", "safe")
            z.writestr("../evil.txt", "malicious")
        
        out_dir = tmpdir / "output_traversal"
        try:
            safe_extract(traversal_zip, out_dir, verbose=False)
            print("✗ Parent traversal ZIP was not blocked!")
            return False
        except RuntimeError as e:
            if "Parent traversal" in str(e):
                print("✓ Parent traversal blocked")
            else:
                print(f"✗ Wrong error: {e}")
                return False
        
        # Test 3: ZIP with absolute path (should fail)
        absolute_zip = tmpdir / "absolute.zip"
        with zipfile.ZipFile(absolute_zip, 'w') as z:
            z.writestr("data/file1.txt", "safe")
            z.writestr("/tmp/evil.txt", "malicious")
        
        out_dir = tmpdir / "output_absolute"
        try:
            safe_extract(absolute_zip, out_dir, verbose=False)
            print("✗ Absolute path ZIP was not blocked!")
            return False
        except RuntimeError as e:
            if "Absolute path" in str(e):
                print("✓ Absolute path blocked")
            else:
                print(f"✗ Wrong error: {e}")
                return False
        
        # Test 4: ZIP with complex traversal (should fail)
        complex_zip = tmpdir / "complex.zip"
        with zipfile.ZipFile(complex_zip, 'w') as z:
            z.writestr("data/file1.txt", "safe")
            z.writestr("data/../../evil.txt", "malicious")
        
        out_dir = tmpdir / "output_complex"
        try:
            safe_extract(complex_zip, out_dir, verbose=False)
            print("✗ Complex traversal ZIP was not blocked!")
            return False
        except RuntimeError as e:
            if "Parent traversal" in str(e) or "Escapes" in str(e):
                print("✓ Complex traversal blocked")
            else:
                print(f"✗ Wrong error: {e}")
                return False
    
    print("\n✓ All Zip Slip protection tests passed!")
    return True


def test_fixed_model():
    """Test that the fixed model has valid normalization stats."""
    print("\n" + "="*70)
    print("TEST: Fixed Model Normalization")
    print("="*70)
    
    model_dir = pathlib.Path(__file__).parent.parent / "data" / "models" / "tbnn_channel_caseholdout"
    
    if not model_dir.exists():
        print("⚠ Model directory not found, skipping test")
        return True
    
    means_file = model_dir / "input_means.txt"
    stds_file = model_dir / "input_stds.txt"
    
    if not means_file.exists() or not stds_file.exists():
        print("⚠ Normalization files not found, skipping test")
        return True
    
    means = np.loadtxt(means_file)
    stds = np.loadtxt(stds_file)
    
    print(f"Means: {means}")
    print(f"Stds:  {stds}")
    
    issues = check_stats_validity(means, stds)
    
    if issues:
        print(f"✗ Fixed model still has issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    print("✓ Fixed model has valid normalization stats")
    return True


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("SECURITY FIXES TEST SUITE")
    print("="*70)
    
    all_passed = True
    
    try:
        if not test_normalization_validation():
            all_passed = False
    except Exception as e:
        print(f"\n✗ Normalization validation test failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    try:
        if not test_zip_slip_protection():
            all_passed = False
    except Exception as e:
        print(f"\n✗ Zip Slip protection test failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    try:
        if not test_fixed_model():
            all_passed = False
    except Exception as e:
        print(f"\n✗ Fixed model test failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    print("\n" + "="*70)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("="*70)
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("="*70)
        return 1


if __name__ == '__main__':
    sys.exit(main())

