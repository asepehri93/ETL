#!/usr/bin/env bash
# Adaptive dt (Δℓ-bound) run for ETL methodology validation
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
LMP="${LMP:-lmp}"
mkdir -p dumps restarts
echo "Running adaptive dt (displacement-bound)..."
$LMP -in in.adaptive_dt
echo "Done. Output: dump_adaptive.lammpstrj, final_adaptive.data"
