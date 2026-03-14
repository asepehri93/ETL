#!/usr/bin/env bash
# Fixed-dt baseline for ETL methodology validation
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
LMP="${LMP:-lmp}"
mkdir -p dumps restarts
echo "Running fixed dt (0.1 fs) reference..."
$LMP -in in.fixed_dt
echo "Done. Output: dump_fixed.lammpstrj, final_fixed.data"
