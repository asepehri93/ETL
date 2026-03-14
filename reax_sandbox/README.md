# ReaxFF CHO gas-phase sandbox (CH₄ + O₂)

Run LAMMPS ReaxFF C/H/O oxidation **without Python**: gas-phase mixture, fixed or adaptive timestep.

## What you need

- **LAMMPS** with REAX package (e.g. `pair_style reax/c`, `fix qeq/reax`).  
  If you see `Unknown pair style reax/c`, use `pair_style reaxff` and change the fix to the style your build expects (see below).
- **Force field**: `ffield.reax.cho` (or `ffield.reax`) in this directory.  
  Often found in LAMMPS `examples/reax` or conda-forge `lammps` extras.

## Quick run (gas-phase oxidation)

```bash
mkdir -p dumps restarts
lmp -in in.reax_cho_gasmix.lmp
```

Or, if your executable has another name:

```bash
LMP=/path/to/lmp ./run.sh
```

## ETL methodology: fixed vs adaptive

- **Fixed reference**: `in.fixed_dt` — 0.1 fs, NPT 300 K.  
  `./run_fixed.sh` or `lmp -in in.fixed_dt`
- **Adaptive (Δℓ-bound)**: `in.adaptive_dt` — dt ≤ sqrt(ε m / F_rms), smoothed.  
  `./run_adaptive.sh` or `lmp -in in.adaptive_dt`
- **RDF check**: `lmp -in in.rdf_check` → `rdf.out`

Same initial configuration (from `in.init_reax`: box + random CH₄/O₂, minimized). No `data.ch4_o2` required.

## Files

| File | Purpose |
|------|--------|
| `in.reax_cho_gasmix.lmp` | Gas-phase run: 1500 K NVT, 5 ps + 50 ps |
| `in.init_reax` | Shared init: box, molecules, create_atoms, minimize, reax/c, qeq |
| `in.fixed_dt` | Fixed dt baseline |
| `in.adaptive_dt` | Adaptive dt from displacement bound |
| `in.rdf_check` | RDF output for validation |
| `mol.ch4`, `mol.o2` | Molecule templates (types 1=C, 2=H, 3=O) |

## If your build only has `reaxff` (no `reax/c`)

In `in.reax_cho_gasmix.lmp` and `in.init_reax` replace:

- `pair_style reax/c NULL` → `pair_style reaxff NULL`
- `fix qeq all qeq/reax 1 0.0 10.0 1e-6 reax/c` → `fix qeq all qeq/reax 1 10.0 1.0e-6 200`

and keep `pair_coeff * * ffield.reax.cho C H O`.

## Sanity checks

- Thermo: temperature near setpoint; no blow-ups; pressure fluctuates.
- Over 10–50 ps you should see onset of chemistry (radicals, H₂O, CO, CO₂).
- If you get QEq warnings or lost atoms, try `timestep 0.05` in the input.

## Venv (for later ETL / Python)

From the **ETL project root** (parent of `reax_sandbox`):

```bash
cd "/Volumes/Extreme Pro/ETL"
python3 -m venv .venv
source .venv/bin/activate
# pip install -r requirements.txt   # when you add one
```

LAMMPS runs from the shell; the venv is for future Python drivers or analysis.
