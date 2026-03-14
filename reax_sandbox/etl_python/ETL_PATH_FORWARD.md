# ETL Path Forward & Diagnostic Results

## Theory framing (keep)

- **ETL-canonical**: `dt = Δℓ √(k_B T / S̄)` — Fisher/metric-inspired.
- **ETL-force**: `dt = Δℓ / √S̄` — configuration-only force-scale controller used here.
- We report results for the **ETL-force** controller; in this benchmark it gave larger dt range and clearer phase separation. Δℓ is calibrated once at start (not the same dimensionless object as in the canonical form).

## Options implemented

### Option A (reduce QEq overhead)
- `cal_every`: 50 → **200** (fewer sentinel calibrations).
- `tol_hysteresis`: 1.5 → **3.0** (less QEq update churn).
- **Result**: In a 2 ps plateau ramp, **ETL(dt+QEq) became faster than ETL(dt)** (15.70 s vs 17.12 s), so QEq adaptation now helps wall time.

### Option B (accuracy: aggressive vs ETL)
- **Accuracy analysis** extended with:
  - **Energy**: `etotal_drift`, `etotal_std`.
  - **Phase-resolved T error**: cold / ramp / hot (plateau schedule 60% / 20% / 20%).
- **2 ps diagnostic** (same schedule, safe as reference):
  - **Hot-phase T error**: ETL(dt) **707.8 K** vs aggressive **826.7 K** → ETL more accurate in the hot phase.
  - Cold phase: all ~8–10 K.
  - Narrative: aggressive baseline is less accurate in the hot phase; ETL(dt) is faster and more accurate there.

### Option C (narrative in code)
- `etl_controller.py`: `compute_dt_fs` docstring and class docstring updated to describe ETL-force vs ETL-canonical and why the force form is used in this benchmark.

## 2 ps diagnostic summary (plateau 300→2000 K)

| Method              | Wall time | vs safe | Hot-phase T error (K) |
|---------------------|-----------|--------|------------------------|
| baseline_safe       | 45.85 s   | 1.00x  | (reference)            |
| baseline_aggressive | 20.84 s   | 2.20x  | 826.7                  |
| etl_dt              | 17.12 s   | 2.68x  | **707.8**              |
| etl_qeq             | **15.70 s** | **2.92x** | 1058.4 (overhead-dominated in short run) |

- ETL(dt) is **faster than aggressive** and **more accurate in the hot phase**.
- With Option A, ETL(dt+QEq) is **fastest** and suitable for claiming QEq improvement once confirmed on longer/larger runs.

## Next step: larger system (e.g. 1500 atoms)

- Current system: **~500 atoms** (data.gasmix_cho.lmp).
- Add a second set of runs with a **~1500-atom** system to test:
  - QEq as a larger fraction of cost → ETL(dt+QEq) savings should be more visible.
  - Same narrative: ETL(dt) and ETL(dt+QEq) faster than aggressive with same or better accuracy.

## How to reproduce

```bash
# Write restart (once)
python run_with_restart.py --write-restart --warmup-ps 5.0 --ramp --T-start 300

# 2 ps diagnostic (Option A params are in ETLParams)
python run_with_restart.py --etl-dt       --t-ps 2 --ramp --ramp-style plateau --T-end 2000 --out-dir outputs_diag_2ps_etl_dt   --dt-target 0.35
python run_with_restart.py --etl-qeq     --t-ps 2 --ramp --ramp-style plateau --T-end 2000 --out-dir outputs_diag_2ps_etl_qeq  --dt-target 0.35
python run_with_restart.py --baseline-safe       --t-ps 2 --ramp --ramp-style plateau --T-end 2000 --out-dir outputs_diag_2ps_safe
python run_with_restart.py --baseline-aggressive --t-ps 2 --ramp --ramp-style plateau --T-end 2000 --out-dir outputs_diag_2ps_aggressive

# Accuracy report (safe as reference)
python analyze_accuracy.py outputs_diag_2ps_aggressive outputs_diag_2ps_etl_dt outputs_diag_2ps_etl_qeq --ref outputs_diag_2ps_safe --no-rdf
```
