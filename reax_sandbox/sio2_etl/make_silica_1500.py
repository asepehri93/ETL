#!/usr/bin/env python3
"""
Create silica_1500.data from silica.data by replicating the simulation box
in the x direction and trimming to exactly 1500 atoms (same format as the
576-atom tutorial file).
"""
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SILICA_DATA = os.path.join(SCRIPT_DIR, "silica.data")
OUT_DATA = os.path.join(SCRIPT_DIR, "silica_1500.data")
TARGET_NATOMS = 1500


def main():
    with open(SILICA_DATA) as f:
        lines = f.readlines()

    # Parse header: find box, masses, atoms section
    xlo = xhi = ylo = yhi = zlo = zhi = None
    masses = []
    atoms_start = None
    atoms_end = None
    for i, line in enumerate(lines):
        if "xlo xhi" in line:
            parts = line.split()
            xlo, xhi = float(parts[0]), float(parts[1])
        elif "ylo yhi" in line:
            parts = line.split()
            ylo, yhi = float(parts[0]), float(parts[1])
        elif "zlo zhi" in line:
            parts = line.split()
            zlo, zhi = float(parts[0]), float(parts[1])
        elif "Atoms # full" in line:
            atoms_start = i + 2  # skip section title and blank line
            break
    for i in range(atoms_start, len(lines)):
        if lines[i].strip().startswith("Velocities") or (lines[i].strip() == "" and i > atoms_start + 10):
            atoms_end = i
            break
    else:
        atoms_end = len(lines)
    # Masses: lines between "Masses" and "Atoms"
    for i, line in enumerate(lines):
        if "Masses" in line:
            j = i + 2
            while j < len(lines) and lines[j].strip():
                p = lines[j].split()
                if len(p) >= 2 and p[0].isdigit():
                    masses.append((int(p[0]), float(p[1])))
                j += 1
            break
    if not masses:
        masses = [(1, 28.0855), (2, 15.9994)]

    Lx = xhi - xlo
    Ly = yhi - ylo
    Lz = zhi - zlo

    # Parse atoms: id mol type q x y z ix iy iz (10 columns)
    atoms = []
    for i in range(atoms_start, atoms_end):
        parts = lines[i].split()
        if len(parts) < 10:
            continue
        aid = int(parts[0])
        mol = int(parts[1])
        atype = int(parts[2])
        q = float(parts[3])
        x, y, z = float(parts[4]), float(parts[5]), float(parts[6])
        ix, iy, iz = int(parts[7]), int(parts[8]), int(parts[9])
        atoms.append((aid, mol, atype, q, x, y, z, ix, iy, iz))

    n_orig = len(atoms)
    assert n_orig == 576, f"Expected 576 atoms, got {n_orig}"

    # Replicate 3x in x so we have enough atoms to select 1500
    replicated = []
    for copy in range(3):
        shift_x = copy * Lx
        for aid, mol, atype, q, x, y, z, ix, iy, iz in atoms:
            new_id = copy * n_orig + aid
            replicated.append((new_id, mol, atype, q, x + shift_x, y, z, 0, 0, 0))

    # Sort by x and take first TARGET_NATOMS to get a contiguous subbox
    replicated.sort(key=lambda r: r[4])
    selected = replicated[:TARGET_NATOMS]
    x_max = max(r[4] for r in selected)
    new_xhi = x_max + 0.5  # margin

    # Renumber atom ids 1..1500
    out_atoms = []
    for i, (_, mol, atype, q, x, y, z, ix, iy, iz) in enumerate(selected, start=1):
        out_atoms.append((i, mol, atype, q, x, y, z, ix, iy, iz))

    # Write new data file (no Velocities; LAMMPS will create on run)
    with open(OUT_DATA, "w") as f:
        f.write("LAMMPS data file from make_silica_1500.py (replicated from silica.data)\n\n")
        f.write(f"{TARGET_NATOMS} atoms\n")
        f.write("2 atom types\n\n")
        f.write(f"{xlo} {new_xhi} xlo xhi\n")
        f.write(f"{ylo} {yhi} ylo yhi\n")
        f.write(f"{zlo} {zhi} zlo zhi\n\n")
        f.write("Atom Type Labels\n\n")
        f.write("1 Si\n")
        f.write("2 O\n\n")
        f.write("Masses\n\n")
        for mid, m in masses:
            f.write(f"{mid} {m}\n")
        f.write("\nAtoms # full\n\n")
        for row in out_atoms:
            aid, mol, atype, q, x, y, z, ix, iy, iz = row
            f.write(f"{aid} {mol} {atype} {q} {x} {y} {z} {ix} {iy} {iz}\n")

    print(f"Wrote {OUT_DATA} ({TARGET_NATOMS} atoms, box x [{xlo}, {new_xhi}])")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
