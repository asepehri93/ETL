# SiO2 ReaxFF (separate from CHO)

Baseline and (later) ETL runs for amorphous silica, kept separate from the CHO gas-phase setup.

## Force field

- **Source:** [LAMMPS Tutorial 5 – Reactive silicon dioxide](https://lammpstutorials.github.io/sphinx/build/html/tutorial5/reactive-silicon-dioxide.html)
- **File:** `ffield.reax.CHOFe` from [lammpstutorials-inputs/tutorial5](https://github.com/lammpstutorials/lammpstutorials-inputs/blob/main/tutorial5/ffield.reax.CHOFe)
- **Usage:** `pair_coeff * * ffield.reax.CHOFe Si O` (same as tutorial; ffield contains C,H,O,Fe,Cl,Si,Al,X; we use Si and O only)

## Data

- `silica.data` — 576 atoms (192 Si, 384 O), amorphous SiO2 from the same tutorial.
- **Larger system:** For 1500-atom runs, create `silica_1500.data` (same LAMMPS data format, Si/O types) and place it in this directory. Then run `python run_suite_sio2.py --write-restart --large` to generate `restart_sio2_1500.rst`, and `python run_suite_sio2.py --large-suite` to run the reduced suite (hat + constant_T, baseline_moderate + etl_full only; other 5 cases are placeholders for later).

## Baseline (no ETL)

- **Script:** `run_baseline_sio2.py`
- **Schedule:** Hat (cold 30% → ramp up 20% → ramp down 20% → cold 30%)
- **Tested:** 2 ps with T_end=3000 K: no lost atoms, cold phases ~300 K, peak ~2600–2700 K (slight lag in short ramp). Ready for full 20 ps once you confirm.

## Run

```bash
source /path/to/.venv/bin/activate
cd reax_sandbox/sio2_etl

# 2 ps test (e.g. T_end=3000 K)
python run_baseline_sio2.py --t-ps 2 --T-end 3000 --out-dir outputs_sio2_3k_test

# Full 20 ps (when ready)
python run_baseline_sio2.py --t-ps 20 --T-end 3000 --out-dir outputs_sio2_baseline
```

## Outputs

- `outputs_sio2_3k_test/` — 2 ps run at T_end=3000 K (baseline_log.csv, dumps/, log.lammps).
- Next: add ETL capabilities (same pattern as CHO) once baseline is approved for full suite.
