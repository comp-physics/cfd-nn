#!/usr/bin/env python3
"""
Safe ZIP extraction utility with Zip Slip protection.

This module provides a secure alternative to zipfile.extractall() that
validates all archive entries to prevent path traversal attacks.

Usage:
    python safe_extract.py <zip_file> <output_dir>
"""

import sys
import pathlib
import zipfile


def safe_extract(zip_path, out_dir, verbose=True):
    """
    Safely extract a ZIP archive with Zip Slip protection.
    
    This function validates all archive entries to ensure they:
    1. Do not contain absolute paths
    2. Do not contain parent directory traversal (..)
    3. Resolve to paths within the output directory
    
    Args:
        zip_path: Path to ZIP file
        out_dir: Output directory for extraction
        verbose: Print extraction progress
    
    Raises:
        RuntimeError: If any entry has an unsafe path
        FileNotFoundError: If ZIP file doesn't exist
    """
    zip_path = pathlib.Path(zip_path)
    out_dir = pathlib.Path(out_dir).resolve()
    
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")
    
    # Create output directory if needed
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"Extracting: {zip_path}")
        print(f"Output dir: {out_dir}")
    
    with zipfile.ZipFile(zip_path, "r") as z:
        # Validate all entries before extracting
        unsafe_entries = []
        
        for info in z.infolist():
            name = info.filename
            
            # Check for absolute paths
            if name.startswith(("/", "\\")):
                unsafe_entries.append(f"Absolute path: {name}")
                continue
            
            # Check for parent directory traversal
            parts = pathlib.PurePosixPath(name).parts
            if ".." in parts:
                unsafe_entries.append(f"Parent traversal: {name}")
                continue
            
            # Verify resolved path is within output directory
            dest = (out_dir / name).resolve()
            try:
                dest.relative_to(out_dir)
            except ValueError:
                unsafe_entries.append(f"Escapes output dir: {name}")
                continue
        
        # Abort if any unsafe entries found
        if unsafe_entries:
            error_msg = "Unsafe ZIP archive detected:\n" + "\n".join(f"  - {e}" for e in unsafe_entries)
            raise RuntimeError(error_msg)
        
        # All entries validated - safe to extract
        if verbose:
            print(f"Validated {len(z.infolist())} entries - all safe")
            print("Extracting...")
        
        z.extractall(out_dir)
        
        if verbose:
            print(f"✓ Extracted to: {out_dir}")


def main():
    """Command-line interface."""
    if len(sys.argv) != 3:
        print("Usage: python safe_extract.py <zip_file> <output_dir>")
        print("\nSafely extracts a ZIP archive with Zip Slip protection.")
        print("Validates all entries to prevent path traversal attacks.")
        return 1
    
    zip_path = sys.argv[1]
    out_dir = sys.argv[2]
    
    try:
        safe_extract(zip_path, out_dir, verbose=True)
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())

