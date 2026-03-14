# ETL Python driver (ReaxFF-CHO)

This folder contains the **Python-driven ETL workflow** implementing the Equal Thermodynamic-Length methodology for adaptive timestep and QEq tolerance control.

## Stages

- **Stage A**: baseline (fixed timestep + fixed QEq tolerance + fixed thermostat coupling)
- **Stage B**: ETL timestep only (adaptive Δt + fixed QEq tolerance)
- **Stage C**: ETL timestep + adaptive QEq tolerance (budget-based with sentinel calibration)
- **Stage D**: Full ETL (adaptive Δt + adaptive QEq tolerance + adaptive thermostat coupling)

## Core ETL Algorithm

The timestep formula is based on maintaining constant "information distance" per integration chunk:

```
Sbar = (1 / 3N) * sum_i (||F_i||^2 / m_i)   # per-DOF force power
dt = Delta_l * sqrt(kB * T / Sbar)           # ETL timestep
```

QEq tolerance is selected via a learned error model so that the QEq contribution
to the information length stays within `alpha_qeq * Delta_l^2`:

```
cap = alpha_qeq * Delta_l^2 * kB * T / dt^2
log10(tol) = (log10(cap) - A) / B            # from sentinel-learned model
```

## Prerequisites

### LAMMPS

You need a LAMMPS build with:
- `pair_style reaxff`
- `fix qeq/reaxff`
- **Python module** (`from lammps import lammps`)

If `import lammps` fails, install/configure one of:
- A conda-forge LAMMPS package that ships Python bindings, or
- A source build with the PYTHON package enabled, and ensure the `lammps` module is on `PYTHONPATH`.

### Force field

Place your ReaxFF CHO force field at:
- `/Volumes/Extreme Pro/ETL/ffield.txt`

## Files

| File | Purpose |
|------|---------|
| `etl_controller.py` | Core `ETLController` with Sbar metric, budget-based QEq, sentinel calibration, chunked stepping |
| `gen_data_gasmix.py` | Generate random CH4+O2 gas mixture data file |
| `run_with_restart.py` | **Recommended**: restart-based workflow ensuring identical initial states |
| `run_baseline.py` | Stage A runner (legacy, uses LAMMPS input file) |
| `run_etl_dt.py` | Stage B runner (legacy) |
| `run_etl_dt_qeq.py` | Stage C runner (legacy) |
| `run_etl_full.py` | Stage D runner (legacy) |
| `analyze_etl_log.py` | Summarize dt/tol/Sbar/thermo distributions from `etl_log.csv` |
| `bin_thermo.py` | Bin thermo output (from `log.lammps`) by physical time |

## Recommended Workflow (restart-based)

Using `run_with_restart.py` ensures all comparison runs start from the exact same microscopic state:

```bash
cd "/Volumes/Extreme Pro/ETL/reax_sandbox/etl_python"

# 1. Generate a data file (if needed)
python gen_data_gasmix.py --n-ch4 10 --n-o2 20 --box 40 -o data.gasmix_cho.lmp

# 2. Create a restart file (warm-up equilibration)
python run_with_restart.py --write-restart

# 3. Run comparison cases (all from the same restart)
python run_with_restart.py --baseline --t-ps 50
python run_with_restart.py --etl-dt --t-ps 50
python run_with_restart.py --etl-qeq --t-ps 50
python run_with_restart.py --etl-full --t-ps 50
```

Output directories:
- `outputs_baseline/`
- `outputs_etl_dt/`
- `outputs_etl_qeq/`
- `outputs_etl_full/`

Each contains:
- `log.lammps` (LAMMPS log)
- `screen.txt` (LAMMPS screen output)
- `etl_log.csv` (dt/tol/Sbar/thermo over time)
- `dumps/` (time-based `write_dump` frames)

## Legacy Workflow (LAMMPS input-based)

If you prefer to use the LAMMPS input files directly:

```bash
cd "/Volumes/Extreme Pro/ETL/reax_sandbox/etl_python"
python run_baseline.py --t-ps 50
python run_etl_dt.py --t-ps 50 --dt-target0-fs 0.25
python run_etl_dt_qeq.py --t-ps 50 --dt-target0-fs 0.25
python run_etl_full.py --t-ps 50 --dt-target0-fs 0.25
```

Note: The legacy workflow does not guarantee identical initial states.

## Validation workflow

### 1) dt/tol/thermo statistics

```bash
python analyze_etl_log.py outputs_baseline/etl_log.csv
python analyze_etl_log.py outputs_etl_qeq/etl_log.csv
```

### 2) Wall time comparison

The `run_with_restart.py` script prints wall times and speedups automatically.

### 3) Thermo fidelity (time-binned)

```bash
python bin_thermo.py outputs_baseline/log.lammps --bin-ps 0.5 --out baseline_thermo.csv
python bin_thermo.py outputs_etl_qeq/log.lammps --bin-ps 0.5 --out etl_qeq_thermo.csv
```

Compare binned means/stds for `Temp`, `Press`, `Etot`, etc.

### 4) RDF / structure

Post-process the time-based dumps for equal-physical-time structure comparison.

## Key Parameters (ETLParams)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `T_target` | 1500.0 | Target temperature for kB*T scaling (K) |
| `dt_target0_fs` | None | If set, auto-calibrate Delta_l so first dt ≈ this value |
| `dt_min_fs` | 0.02 | Minimum timestep (fs) |
| `dt_max_fs` | 0.30 | Maximum timestep (fs) |
| `dt_growth` | 1.5 | Multiplicative growth limiter: dt <= prev * dt_growth |
| `dt_shrink` | 2.0 | Multiplicative shrink limiter: dt >= prev / dt_shrink |
| `chunk_steps` | 10 | Steps per ETL controller iteration |
| `alpha_qeq` | 0.30 | Fraction of ETL budget for QEq error |
| `tol_min` | 1e-6 | Minimum QEq tolerance |
| `tol_max` | 5e-4 | Maximum QEq tolerance |
| `tol_hysteresis` | 2.0 | Ratio threshold for redefining QEq fix |
| `cal_every` | 100 | Recalibrate error model every N chunks |

## Expected Performance

On 50 ps CH4+O2 oxidation at 1500 K with proper parameters, expect:
- **~3x speedup** with ETL+QEq vs fixed baseline
- Temperature mean relative error < 3%
- Total energy error < 1%
- RDF L2 difference small
