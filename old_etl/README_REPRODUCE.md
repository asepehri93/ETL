# CHO ETL reproduction (old code, isolated)

This folder contains a **standalone** reproduction of the old CHO (CH₄ + O₂) ETL+QEq run that achieved ~3–4× speedup over a fixed-dt baseline. It is kept **isolated** from the current SiO₂/ETL code so we can compare and adapt.

## Quick run

From repo root or from `old_etl`:

```bash
cd old_etl
python run_cho_reproduce.py --t-ps 5
```

- **Fixed baseline**: 0.1 fs, 5 ps → 50,000 steps.
- **ETL+QEq**: same 5 ps with adaptive dt (target 0.30 fs, max 0.30 fs) and adaptive QEq (tol_max=5e-4), chunk=20.

Output: `log_fixed_repro.json`, `log_etl_qeq_repro.json`, and printed speedup (expect **~3–3.5×**).

For a quick sanity check use `--t-ps 2`. To match the original long run use `--t-ps 50` (will take much longer).

## What the old run did (and why it got ~3–4×)

| Aspect | Old CHO (this reproduction) | Current SiO₂ suite |
|--------|----------------------------|---------------------|
| **Baseline dt** | 0.1 fs (very conservative) | 0.25 fs (“safe”) |
| **ETL dt_target0** | 0.30 fs (3× baseline) | 0.35 fs |
| **ETL dt_max** | 0.30 fs | 1.0 fs |
| **ETL dt_min** | 0.03 fs | 0.25 fs (tied to safe) |
| **Schedule** | Constant T = 1500 K | Hat: 300→4500→300 K |
| **QEq tol_max** | 5×10⁻⁴ (looser) | 10⁻² (tighter for fidelity) |
| **Chunk** | 20 | 20 |
| **Formula** | dt = Δℓ √(kBT/S̄), Δℓ from dt_target0 | Same (canonical ETL) |

So the old setup had **large headroom**: baseline 0.1 fs vs ETL allowed up to 0.30 fs (3×), constant high T (no ramp), and looser QEq. That combination yielded ~3–4× wall-time speedup.

## Why current SiO₂ gives ~1.2–1.4× instead of ~4×

1. **Baseline is already “safe” (0.25 fs)**  
   We compare against 0.25 fs, not 0.1 fs. So the same ETL formula has less room to “beat” the baseline (dt can’t go 3× above 0.25 in the same way it could above 0.1).

2. **Hat schedule (300→4500→300 K)**  
   During heating and cooling, S̄ is high so ETL shrinks dt. That limits how much we can gain versus a constant-T run.

3. **dt_min = 0.25 fs**  
   We set dt_min to match the safe baseline, so in hard phases we don’t go below 0.25 fs but we also don’t get the same “huge” step reduction in easy phases as in the old run (where dt could go down to 0.03 fs if needed).

4. **QEq fidelity (tol_max = 10⁻²)**  
   We tightened tol_max for charge/pressure stability on SiO₂, which reduces the extra gain that looser QEq gave on CHO.

5. **Ramp and early exit**  
   Ramp runs finish in ~8.35–8.40 ps (early exit); that improves wall time but doesn’t change the fact that the **per-ps** gain is still limited by the above.

So: the **methodology is the same** (canonical ETL dt + adaptive QEq); the **conditions** (baseline dt, schedule, dt_min, tol_max) are more conservative on SiO₂, which is why we see ~1.2–1.4× there and ~3–4× here.

## How we could adapt the current approach to get larger gains (after review)

- **Softer baseline comparison**: e.g. compare ETL to a 0.1 fs (or 0.15 fs) “reference” run for reporting speedup, while still using 0.25 fs as the “safe” production setting. Then ETL would have more headroom to show a higher multiplier.
- **Constant-T or milder schedule**: a constant-T or milder ramp would let ETL use larger dt more of the time (as in CHO).
- **Relax QEq cap where safe**: on systems where charge/pressure are less sensitive, allowing tol_max slightly looser than 10⁻² could recover some of the QEq savings (with validation).
- **Larger systems**: ETL’s benefit (fewer steps at similar fidelity) can scale with system size and cost of force/QEq; testing on larger SiO₂ (e.g. 1500 atoms) may show a higher effective speedup.
- **Report “steps saved” vs fixed 0.25 fs**: we already have step_savings_pct; emphasizing that in addition to wall-time speedup keeps the story consistent with the CHO result (fewer steps for the same physics).

No code changes are made in the main ETL/SiO₂ tree by this reproduction; it is for validation and planning only.
