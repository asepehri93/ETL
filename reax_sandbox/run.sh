#!/usr/bin/env bash
# Run LAMMPS ReaxFF CHO sandbox. Requires LAMMPS with REAX and ffield.reax.cho in this dir or LAMMPS path.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default LAMMPS executable (override with LMP env var)
LMP="${LMP:-lmp}"

mkdir -p dumps restarts

echo "Running gas-phase CH4+O2 oxidation (5 ps + 50 ps)..."
$LMP -in in.reax_cho_gasmix.lmp

echo "Done. Check dumps/ and thermo output."
