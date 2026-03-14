#!/usr/bin/env python3
"""
Convert LAMMPS dump files to XYZ format for visualization.

This script converts LAMMPS custom dump files (with id, type, q, x, y, z columns)
to standard XYZ format that can be opened in visualization tools like OVITO, VMD,
ASE, etc.

Usage:
    # Convert all dumps in a directory to a single trajectory file
    python convert_dumps_to_xyz.py outputs_etl_full/dumps -o outputs_etl_full/trajectory.xyz

    # Convert a single dump file
    python convert_dumps_to_xyz.py outputs_etl_full/dumps/frame_000001_000001.000ps.dump

    # Convert to separate XYZ files per frame
    python convert_dumps_to_xyz.py outputs_etl_full/dumps --separate

Type mapping (from pair_coeff * * ffield C H O):
    1 -> C (Carbon)
    2 -> H (Hydrogen)
    3 -> O (Oxygen)
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


TYPE_TO_ELEMENT: Dict[int, str] = {
    1: "C",
    2: "H",
    3: "O",
}


def parse_lammps_dump(filepath: str) -> Optional[Tuple[int, int, List[Tuple[str, float, float, float]]]]:
    """
    Parse a LAMMPS custom dump file.
    
    Returns:
        Tuple of (timestep, natoms, list of (element, x, y, z)) or None if parsing fails
    """
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {filepath}: {e}", file=sys.stderr)
        return None
    
    timestep = 0
    natoms = 0
    atoms = []
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if line == "ITEM: TIMESTEP":
            i += 1
            if i < len(lines):
                timestep = int(lines[i].strip())
            i += 1
            continue
        
        if line == "ITEM: NUMBER OF ATOMS":
            i += 1
            if i < len(lines):
                natoms = int(lines[i].strip())
            i += 1
            continue
        
        if line.startswith("ITEM: BOX BOUNDS"):
            i += 4  # Skip box bounds (3 lines)
            continue
        
        if line.startswith("ITEM: ATOMS"):
            # Parse column order from header: "ITEM: ATOMS id type q x y z"
            cols = line.split()[2:]  # Skip "ITEM:" and "ATOMS"
            try:
                type_idx = cols.index("type")
                x_idx = cols.index("x")
                y_idx = cols.index("y")
                z_idx = cols.index("z")
            except ValueError:
                print(f"Warning: Could not find required columns in {filepath}", file=sys.stderr)
                return None
            
            i += 1
            while i < len(lines) and len(atoms) < natoms:
                atom_line = lines[i].strip()
                if not atom_line or atom_line.startswith("ITEM:"):
                    break
                parts = atom_line.split()
                if len(parts) > max(type_idx, x_idx, y_idx, z_idx):
                    atom_type = int(parts[type_idx])
                    x = float(parts[x_idx])
                    y = float(parts[y_idx])
                    z = float(parts[z_idx])
                    element = TYPE_TO_ELEMENT.get(atom_type, f"X{atom_type}")
                    atoms.append((element, x, y, z))
                i += 1
            continue
        
        i += 1
    
    if len(atoms) != natoms:
        print(f"Warning: Expected {natoms} atoms but found {len(atoms)} in {filepath}", file=sys.stderr)
    
    return (timestep, len(atoms), atoms)


def write_xyz_frame(f, timestep: int, atoms: List[Tuple[str, float, float, float]], comment: str = "") -> None:
    """Write a single XYZ frame to a file handle."""
    f.write(f"{len(atoms)}\n")
    f.write(f"{comment if comment else f'timestep={timestep}'}\n")
    for element, x, y, z in atoms:
        f.write(f"{element} {x:.6f} {y:.6f} {z:.6f}\n")


def extract_time_from_filename(filename: str) -> Optional[float]:
    """Extract time in ps from filename like 'frame_000001_000001.000ps.dump'."""
    match = re.search(r'(\d+\.\d+)ps', filename)
    if match:
        return float(match.group(1))
    return None


def _sort_key_by_time(path: Path) -> Tuple[float, str]:
    """Sort key: (time_ps, path_str) so frames are in chronological order."""
    t = extract_time_from_filename(path.name)
    return (t if t is not None else float('inf'), path.name)


def convert_directory(
    dump_dir: str,
    output_file: Optional[str] = None,
    separate: bool = False,
) -> int:
    """
    Convert all dump files in a directory to XYZ.
    
    Args:
        dump_dir: Path to directory containing .dump files
        output_file: Output XYZ file path (for concatenated output)
        separate: If True, write separate XYZ files per frame
        
    Returns:
        Number of frames converted
    """
    dump_path = Path(dump_dir)
    if not dump_path.is_dir():
        print(f"Error: {dump_dir} is not a directory", file=sys.stderr)
        return 0
    
    dump_files = sorted(
        [f for f in dump_path.glob("*.dump") if not f.name.startswith("._")],
        key=_sort_key_by_time,
    )
    if not dump_files:
        print(f"No .dump files found in {dump_dir}", file=sys.stderr)
        return 0
    
    print(f"Found {len(dump_files)} dump files in {dump_dir}")
    
    if separate:
        xyz_dir = dump_path / "xyz"
        xyz_dir.mkdir(exist_ok=True)
        
        converted = 0
        for dump_file in dump_files:
            result = parse_lammps_dump(str(dump_file))
            if result:
                timestep, natoms, atoms = result
                time_ps = extract_time_from_filename(dump_file.name)
                comment = f"t={time_ps:.3f}ps timestep={timestep}" if time_ps else f"timestep={timestep}"
                
                xyz_file = xyz_dir / dump_file.name.replace(".dump", ".xyz")
                with open(xyz_file, 'w') as f:
                    write_xyz_frame(f, timestep, atoms, comment)
                converted += 1
        
        print(f"Converted {converted} frames to {xyz_dir}/")
        return converted
    
    else:
        out_path = output_file if output_file else str(dump_path.parent / "trajectory.xyz")
        
        converted = 0
        with open(out_path, 'w') as f:
            for dump_file in dump_files:
                result = parse_lammps_dump(str(dump_file))
                if result:
                    timestep, natoms, atoms = result
                    time_ps = extract_time_from_filename(dump_file.name)
                    comment = f"t={time_ps:.3f}ps timestep={timestep}" if time_ps else f"timestep={timestep}"
                    write_xyz_frame(f, timestep, atoms, comment)
                    converted += 1
        
        print(f"Converted {converted} frames to {out_path}")
        return converted


def convert_single_file(dump_file: str, output_file: Optional[str] = None) -> bool:
    """
    Convert a single dump file to XYZ.
    
    Args:
        dump_file: Path to .dump file
        output_file: Output XYZ file path (default: same name with .xyz extension)
        
    Returns:
        True if successful
    """
    result = parse_lammps_dump(dump_file)
    if not result:
        return False
    
    timestep, natoms, atoms = result
    
    if output_file:
        out_path = output_file
    else:
        out_path = dump_file.replace(".dump", ".xyz")
    
    time_ps = extract_time_from_filename(os.path.basename(dump_file))
    comment = f"t={time_ps:.3f}ps timestep={timestep}" if time_ps else f"timestep={timestep}"
    
    with open(out_path, 'w') as f:
        write_xyz_frame(f, timestep, atoms, comment)
    
    print(f"Converted {dump_file} -> {out_path} ({natoms} atoms)")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert LAMMPS dump files to XYZ format for visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Convert all dumps in a directory to a single trajectory
    python convert_dumps_to_xyz.py outputs_etl_full/dumps -o outputs_etl_full/trajectory.xyz
    
    # Convert with separate files per frame
    python convert_dumps_to_xyz.py outputs_etl_full/dumps --separate
    
    # Convert a single dump file
    python convert_dumps_to_xyz.py frame_000001.dump
""",
    )
    
    parser.add_argument("input", help="Input dump file or directory containing dump files")
    parser.add_argument("-o", "--output", help="Output XYZ file path (for single file or concatenated output)")
    parser.add_argument("--separate", action="store_true", help="Write separate XYZ files per frame")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if input_path.is_dir():
        count = convert_directory(str(input_path), args.output, args.separate)
        if count == 0:
            sys.exit(1)
    elif input_path.is_file():
        if not convert_single_file(str(input_path), args.output):
            sys.exit(1)
    else:
        print(f"Error: {args.input} does not exist", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
