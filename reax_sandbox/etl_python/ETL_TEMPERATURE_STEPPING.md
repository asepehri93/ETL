# ETL-Consistent Adaptive Temperature Stepping

This document derives the theoretical foundation for adaptive temperature stepping
within the Equal Thermodynamic Length (ETL) framework.

## Background: ETL for Timestep Adaptation

The core ETL principle for timestep is:

```
dt = Δl · √(kB·T / S̄)
```

where:
- `Δl` is the target information length per step
- `S̄ = (1/3N) Σᵢ ||Fᵢ||²/mᵢ` is the average force power (local curvature surrogate)
- `kB·T` sets the thermal energy scale

This ensures each MD step covers approximately equal "information distance" in
configuration space, spending more time (smaller dt) when the landscape is rugged
(high S̄) and less time when it's smooth.

## Extension to Temperature Stepping

When ramping temperature from T₀ to T_final, we want to choose the temperature
increment δT adaptively so that:
1. The system has time to equilibrate at each temperature
2. We don't waste time at temperatures where the system is "easy"
3. We spend more time at temperatures where important transitions occur

### Formulation 1: Susceptibility-Based (Heat Capacity Proxy)

The "thermodynamic cost" of changing temperature relates to the system's heat capacity:

```
Cᵥ = ∂⟨E⟩/∂T = ⟨(E - ⟨E⟩)²⟩ / (kB·T²)
```

High heat capacity means energy fluctuations are large—the system is "sensitive"
to temperature changes, possibly near a phase transition or onset of chemical reactions.

**Adaptive rule:**
```
δT = Δl_T · T² / σ_E²
```

where:
- `Δl_T` is the target temperature-step "budget"
- `σ_E² = ⟨(E - ⟨E⟩)²⟩` is the energy variance (estimated online from recent chunks)

This chooses smaller temperature steps when energy fluctuations are large (system
is "active") and larger steps when fluctuations are small (system is "quiet").

**Online estimation:**
```python
# Maintain running statistics over last N chunks
E_mean = ewma(etotal, alpha=0.1)
E_var = ewma((etotal - E_mean)**2, alpha=0.1)
sigma_E = sqrt(E_var)

# Choose temperature step
dT = Delta_l_T * T**2 / max(sigma_E**2, epsilon)
dT = clamp(dT, dT_min, dT_max)
```

### Formulation 2: Sbar-Informed Heuristic

We already compute S̄ for timestep adaptation. Since S̄ correlates with local
landscape complexity, we can use it as a proxy for "interesting dynamics":

**Adaptive rule:**
```
δT = Δl_T · √(kB·T / S̄)
```

This is dimensionally consistent and mirrors the dt formula. High S̄ (rough landscape)
→ smaller temperature steps. Low S̄ (smooth landscape) → larger temperature steps.

**Comparison with susceptibility approach:**
- S̄ is force-based (instantaneous)
- σ_E is energy-based (requires averaging)
- S̄ responds faster but may be noisier
- σ_E integrates over time, providing a smoother signal

### Hybrid Approach (Recommended)

Use S̄ as the primary signal but smooth it with an EWMA and add a susceptibility
correction when energy variance spikes:

```python
# Smoothed Sbar
Sbar_smooth = ewma(Sbar, alpha=0.2)

# Energy susceptibility factor (dimensionless, >= 1)
chi = 1.0 + sigma_E**2 / (kB * T**2 * baseline_var)

# Adaptive temperature step
dT_raw = Delta_l_T * sqrt(kB * T / Sbar_smooth) / chi
dT = clamp(dT_raw, dT_min, dT_max)
```

## Implementation Constants

Suggested defaults for the 300K → 3000K ramp:
- `Delta_l_T = 50.0` — target "temperature-length" budget in K·sqrt(kcal/mol)
- `dT_min = 5.0` K — minimum temperature step
- `dT_max = 200.0` K — maximum temperature step
- `E_var_baseline = 100.0` — (kcal/mol)² baseline variance for χ normalization

## Validation Strategy

1. Run fixed-ramp (Phase 2a) to establish baseline T(t), E(t), S̄(t) curves
2. Run adaptive-T (Phase 2b) with same total ΔT range
3. Compare:
   - Total simulation time (should be similar or less for adaptive)
   - Temperature tracking (adaptive should reach same final T)
   - Energy trajectory (should be similar to fixed-ramp)
   - Time spent in each temperature phase (adaptive should allocate more time to "active" phases)
