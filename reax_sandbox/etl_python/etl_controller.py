import csv
import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

# LAMMPS extract_atom dtype constants (match lammps.constants)
LAMMPS_INT = 0
LAMMPS_DOUBLE = 2   # per-atom 1D double (e.g. charge "q")
LAMMPS_DOUBLE_2D = 3

KB_KCAL_PER_MOL_K = 0.0019872041


DEFAULT_TYPE_MASS_G_PER_MOL: Dict[int, float] = {
    1: 12.011,
    2: 1.008,
    3: 15.999,
}


@dataclass
class ETLParams:
    # Time integration
    dt_min_fs: float = 0.25       # Minimum dt (match moderate baseline; never slower than reference)
    dt_max_fs: float = 1.00       # Allow large dt in calm phases; ETL self-regulates via Sbar
    dt_target0_fs: Optional[float] = None
    # Delta_l sets the "information length" scale per step; larger values → larger dt
    # Auto-calibrated from dt_target0_fs if provided
    Delta_l: float = 3.5
    dt_smoothing_alpha: float = 0.0  # No smoothing (like old code)
    dt_growth: float = 1.5           # Match old code
    dt_shrink: float = 2.0           # Match old code

    # Temperature for kB*T scaling
    T_target: float = 1500.0

    # QEq control
    qeq_nevery: int = 1
    qeq_cutlo: float = 0.0
    qeq_cuthi: float = 10.0
    tol_min: float = 1.0e-6       # Tight tolerance for demanding phases
    tol_max: float = 1.0e-2       # Cap QEq tolerance for charge/pressure fidelity (was 5e-2)
    tol_fixed: float = 1.0e-6
    tol_hysteresis: float = 1.5   # Allow QEq tol to adapt when ratio >= 1.5
    qeq_maxiter: int = 200
    alpha_qeq: float = 0.65       # QEq budget fraction (tune for charge stability vs savings)
    cal_every: int = 100          # Recalibrate QEq error model (reduced overhead vs 50)

    # Thermostat control (Langevin damping)
    lang_temp: float = 1500.0
    lang_damp_min_fs: float = 50.0
    lang_damp_max_fs: float = 500.0
    lang_damp_fixed_fs: float = 100.0
    lang_damp_ratio: float = 1000.0
    lang_seed: int = 90421
    lang_hysteresis: float = 1.1

    # Barostat control (NPT Pdamp - optional)
    baro_pdamp_min_fs: float = 500.0
    baro_pdamp_max_fs: float = 5000.0
    baro_pdamp_fixed_fs: float = 1000.0
    baro_pdamp_ratio: float = 5000.0
    baro_hysteresis: float = 2.0

    # Output
    snap_every_ps: float = 0.1
    out_dir: str = "outputs"
    log_charge_stats: bool = False  # Log per-type mean charge and overall std (for debugging charge dynamics)
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None  # Called each chunk with log row (for monitoring)

    # Controller runtime
    chunk_steps: int = 20  # Match old code (reduce Python overhead)
    step_savings_reference_dt_fs: float = 0.25  # Reference dt for step_savings_pct (e.g. 0.1 for vs reference baseline)

    # dt hysteresis: only update dt if relative change exceeds this threshold
    dt_hysteresis: float = 0.0  # 0 = disabled; 0.05 = 5% minimum change

    # Adaptive temperature stepping (Phase 2b)
    adapt_T_enabled: bool = False  # Enable adaptive temperature stepping
    T_start: float = 300.0         # Starting temperature for adaptive ramp
    T_end: float = 3000.0          # Target final temperature
    Delta_l_T: float = 50.0        # Temperature-step "budget" in K·sqrt(kcal/mol)
    dT_min: float = 5.0            # Minimum temperature step (K)
    dT_max: float = 200.0          # Maximum temperature step (K)
    dT_smoothing_alpha: float = 0.3  # EWMA smoothing for dT
    E_var_baseline: float = 100.0  # Baseline energy variance for susceptibility (kcal/mol)²
    use_susceptibility: bool = True  # Use energy variance in dT calculation


# Type alias for temperature schedule: callable (time_ps) -> T_K
TempSchedule = Callable[[float], float]


class EWMATracker:
    """Exponentially weighted moving average tracker for online statistics."""
    
    def __init__(self, alpha: float = 0.1, initial: Optional[float] = None):
        self.alpha = alpha
        self.value = initial
        self.var = 0.0
        self._initialized = initial is not None
    
    def update(self, x: float) -> float:
        """Update with new value and return current EWMA."""
        if not self._initialized or self.value is None:
            self.value = x
            self.var = 0.0
            self._initialized = True
        else:
            delta = x - self.value
            self.value = self.alpha * x + (1 - self.alpha) * self.value
            self.var = (1 - self.alpha) * (self.var + self.alpha * delta ** 2)
        return self.value
    
    @property
    def std(self) -> float:
        return math.sqrt(self.var) if self.var > 0 else 0.0


def make_linear_ramp(T_start: float, T_end: float, t_ps: float) -> TempSchedule:
    """Create a linear temperature ramp from T_start to T_end over t_ps."""
    def schedule(time_ps: float) -> float:
        if t_ps <= 0:
            return T_end
        frac = min(1.0, max(0.0, time_ps / t_ps))
        return T_start + frac * (T_end - T_start)
    return schedule


def make_constant_temp(T: float) -> TempSchedule:
    """Create a constant temperature schedule."""
    def schedule(time_ps: float) -> float:
        return T
    return schedule


def make_triangle_ramp(T_low: float, T_high: float, t_ps: float) -> TempSchedule:
    """
    Create a triangle temperature ramp: T_low -> T_high -> T_low over t_ps.
    
    First half (0 to t_ps/2): linear ramp from T_low to T_high
    Second half (t_ps/2 to t_ps): linear ramp from T_high back to T_low
    """
    def schedule(time_ps: float) -> float:
        if t_ps <= 0:
            return T_low
        half_time = t_ps / 2.0
        if time_ps <= half_time:
            frac = time_ps / half_time
            return T_low + frac * (T_high - T_low)
        else:
            frac = (time_ps - half_time) / half_time
            frac = min(1.0, frac)
            return T_high - frac * (T_high - T_low)
    return schedule


def make_plateau_ramp_plateau(T_low: float, T_high: float, t_ps: float,
                               low_frac: float = 0.33, high_frac: float = 0.33) -> TempSchedule:
    """
    Create a plateau-ramp-plateau temperature schedule.
    
    Structure: T_low plateau -> ramp to T_high -> T_high plateau
    
    Args:
        T_low: Low temperature (K) for initial plateau
        T_high: High temperature (K) for final plateau
        t_ps: Total simulation time in picoseconds
        low_frac: Fraction of time at T_low plateau (default 0.33 = 33%)
        high_frac: Fraction of time at T_high plateau (default 0.33 = 33%)
    
    Example with t_ps=15, low_frac=0.33, high_frac=0.33:
        - 0-5 ps: T_low plateau (300K)
        - 5-10 ps: linear ramp T_low -> T_high
        - 10-15 ps: T_high plateau (2000K)
    """
    ramp_frac = 1.0 - low_frac - high_frac
    if ramp_frac < 0:
        raise ValueError("low_frac + high_frac must be <= 1.0")
    
    def schedule(time_ps: float) -> float:
        if t_ps <= 0:
            return T_low
        
        frac = time_ps / t_ps
        frac = min(1.0, max(0.0, frac))
        
        if frac < low_frac:
            return T_low
        elif frac < low_frac + ramp_frac:
            ramp_progress = (frac - low_frac) / ramp_frac if ramp_frac > 0 else 1.0
            return T_low + ramp_progress * (T_high - T_low)
        else:
            return T_high
    
    return schedule


def make_hat_schedule(T_low: float, T_high: float, t_ps: float,
                      cold1_frac: float = 0.30, up_frac: float = 0.20,
                      down_frac: float = 0.20, cold2_frac: float = 0.30) -> TempSchedule:
    """
    Hat-shaped temperature schedule: cold → ramp up → ramp down → cold.

    Args:
        T_low:  Low / baseline temperature (K)
        T_high: Peak temperature (K)
        t_ps:   Total simulation time (ps)
        cold1_frac: Fraction at T_low before ramp-up
        up_frac:    Fraction spent ramping up
        down_frac:  Fraction spent ramping down
        cold2_frac: Fraction at T_low after ramp-down
    """
    total = cold1_frac + up_frac + down_frac + cold2_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Fractions must sum to 1.0, got {total}")

    def schedule(time_ps: float) -> float:
        if t_ps <= 0:
            return T_low
        frac = min(1.0, max(0.0, time_ps / t_ps))
        if frac < cold1_frac:
            return T_low
        elif frac < cold1_frac + up_frac:
            p = (frac - cold1_frac) / up_frac if up_frac > 0 else 1.0
            return T_low + p * (T_high - T_low)
        elif frac < cold1_frac + up_frac + down_frac:
            p = (frac - cold1_frac - up_frac) / down_frac if down_frac > 0 else 1.0
            return T_high - p * (T_high - T_low)
        else:
            return T_low
    return schedule


class AdaptiveTempController:
    """
    Adaptive temperature stepping controller.
    
    Uses Sbar (force power) and optionally energy variance (susceptibility)
    to determine temperature increments that maintain equal "thermodynamic length"
    in temperature space.
    
    Primary formula (Sbar-informed):
        dT = Delta_l_T * sqrt(kB*T / Sbar_smooth) / chi
    
    where chi is a susceptibility factor based on energy variance.
    """
    
    def __init__(
        self,
        T_start: float,
        T_end: float,
        Delta_l_T: float = 50.0,
        dT_min: float = 5.0,
        dT_max: float = 200.0,
        dT_smoothing_alpha: float = 0.3,
        E_var_baseline: float = 100.0,
        use_susceptibility: bool = True,
    ):
        self.T_start = T_start
        self.T_end = T_end
        self.Delta_l_T = Delta_l_T
        self.dT_min = dT_min
        self.dT_max = dT_max
        self.dT_smoothing_alpha = dT_smoothing_alpha
        self.E_var_baseline = E_var_baseline
        self.use_susceptibility = use_susceptibility
        
        self.direction = 1.0 if T_end >= T_start else -1.0
        self.current_T = T_start
        self._prev_dT = (dT_min + dT_max) / 2
        
        # EWMA trackers for smoothing
        self._Sbar_tracker = EWMATracker(alpha=0.2)
        self._E_tracker = EWMATracker(alpha=0.1)
    
    def is_complete(self) -> bool:
        """Check if we've reached the target temperature."""
        if self.direction > 0:
            return self.current_T >= self.T_end
        else:
            return self.current_T <= self.T_end
    
    def compute_dT(self, Sbar: float, etotal: float) -> float:
        """
        Compute the next temperature step based on current Sbar and energy.
        
        Args:
            Sbar: Current force power (local curvature)
            etotal: Current total energy
        
        Returns:
            Temperature increment (positive for heating, negative for cooling)
        """
        # Update trackers
        Sbar_smooth = self._Sbar_tracker.update(Sbar)
        E_mean = self._E_tracker.update(etotal)
        
        # Sbar-based temperature step
        T = max(self.current_T, 1.0)  # Avoid division by zero
        if Sbar_smooth > 0:
            dT_raw = self.Delta_l_T * math.sqrt(KB_KCAL_PER_MOL_K * T / Sbar_smooth)
        else:
            dT_raw = self.dT_max
        
        # Susceptibility correction (dimensionless factor >= 1)
        chi = 1.0
        if self.use_susceptibility and self._E_tracker.var > 0:
            kBT2 = (KB_KCAL_PER_MOL_K * T) ** 2
            chi = 1.0 + self._E_tracker.var / (kBT2 + self.E_var_baseline)
        
        dT = dT_raw / chi
        
        # Clamp to bounds
        dT = max(self.dT_min, min(self.dT_max, dT))
        
        # Smooth with previous
        dT = self.dT_smoothing_alpha * dT + (1 - self.dT_smoothing_alpha) * self._prev_dT
        
        # Final clamp
        dT = max(self.dT_min, min(self.dT_max, dT))
        self._prev_dT = dT
        
        # Apply direction and ensure we don't overshoot
        dT_signed = self.direction * dT
        if self.direction > 0:
            max_dT = self.T_end - self.current_T
            dT_signed = min(dT_signed, max_dT)
        else:
            max_dT = self.current_T - self.T_end
            dT_signed = max(dT_signed, -max_dT)
        
        return dT_signed
    
    def step(self, Sbar: float, etotal: float) -> float:
        """
        Compute and apply a temperature step.
        
        Returns:
            New target temperature
        """
        if self.is_complete():
            return self.T_end
        
        dT = self.compute_dT(Sbar, etotal)
        self.current_T += dT
        
        # Ensure we stay within bounds
        if self.direction > 0:
            self.current_T = min(self.current_T, self.T_end)
        else:
            self.current_T = max(self.current_T, self.T_end)
        
        return self.current_T
    
    def get_current_T(self) -> float:
        return self.current_T


class ETLController:
    """
    Python-side controller for ReaxFF CHO runs implementing Equal Thermodynamic-Length (ETL).

    Core formula:
        Sbar = (1 / 3N) * sum_i (||F_i||^2 / m_i)
        dt = Delta_l * sqrt(kB * T / Sbar)

    This implementation uses the ETL-force (configuration-only) form dt = Delta_l/sqrt(Sbar)
    rather than the ETL-canonical form dt = Delta_l*sqrt(kB*T/Sbar). Both are ETL-inspired;
    the force form was chosen for this benchmark because it gave larger dt dynamic range and
    clearer phase separation (calm vs reactive). Delta_l is calibrated once at start.

    QEq tolerance is selected via a learned error model so that the QEq contribution
    to the information length stays within alpha_qeq * Delta_l^2.

    Stages supported:
    - Baseline: fixed dt + fixed QEq tol + fixed thermostat/barostat coupling
    - ETL(dt): adaptive dt + fixed QEq tol
    - ETL(dt+QEq): adaptive dt + adaptive QEq tol (budget-based with sentinel calibration)
    - ETL(full): adaptive dt + adaptive QEq tol + adaptive thermostat/barostat coupling
    """

    def __init__(
        self,
        lmp,
        params: ETLParams,
        type_mass_g_per_mol: Optional[Dict[int, float]] = None,
        adapt_dt: bool = True,
        adapt_qeq: bool = True,
        adapt_langevin: bool = False,
        adapt_barostat: bool = False,
        adapt_T: bool = False,
        adapt_ramp: bool = False,
        adapt_qeq_nevery: bool = False,
        fixed_dt_fs: float = 0.1,
        T_schedule: Optional[TempSchedule] = None,
    ):
        self.lmp = lmp
        self.p = params
        self.adapt_dt = adapt_dt
        self.adapt_qeq = adapt_qeq
        self.adapt_langevin = adapt_langevin
        self.adapt_barostat = adapt_barostat
        self.adapt_T = adapt_T
        self.adapt_ramp = adapt_ramp
        self.adapt_qeq_nevery = adapt_qeq_nevery
        self.fixed_dt_fs = float(fixed_dt_fs)
        
        # Temperature schedule: if None, use constant T_target
        # If adapt_T is enabled, create an AdaptiveTempController instead
        if adapt_T:
            self._adaptive_T_controller = AdaptiveTempController(
                T_start=params.T_start,
                T_end=params.T_end,
                Delta_l_T=params.Delta_l_T,
                dT_min=params.dT_min,
                dT_max=params.dT_max,
                dT_smoothing_alpha=params.dT_smoothing_alpha,
                E_var_baseline=params.E_var_baseline,
                use_susceptibility=params.use_susceptibility,
            )
            self.T_schedule = None  # Will use adaptive controller instead
        else:
            self._adaptive_T_controller = None
            self.T_schedule = T_schedule if T_schedule else make_constant_temp(params.T_target)

        # Adaptive ramp: ETL-driven traversal speed along the T schedule.
        # _ramp_progress is a "virtual clock" in [0, t_ps] that advances
        # faster when Sbar is low (easy) and slower when Sbar is high (hard).
        self._ramp_progress_ps: float = 0.0
        self._ramp_Sbar_ref: Optional[float] = None  # calibrated at startup
        
        self._current_lang_temp: float = params.lang_temp  # Track current Langevin target T
        self._lang_seed_counter: int = 0  # Increment on each fix recreation for unique RNG seeds

        self.type_mass = dict(DEFAULT_TYPE_MASS_G_PER_MOL)
        if type_mass_g_per_mol:
            self.type_mass.update(type_mass_g_per_mol)

        os.makedirs(self.p.out_dir, exist_ok=True)
        self.snap_dir = os.path.join(self.p.out_dir, "dumps")
        os.makedirs(self.snap_dir, exist_ok=True)

        self.log_path = os.path.join(self.p.out_dir, "etl_log.csv")
        self._log_file = open(self.log_path, "w", newline="")
        base_fieldnames = [
            "step_index",
            "time_fs",
            "time_ps",
            "dt_fs",
            "tol",
            "lang_damp_fs",
            "baro_pdamp_fs",
            "T_target",
            "Sbar",
            "temp",
            "press",
            "pe",
            "ke",
            "etotal",
            "schedule_ps",
        ]
        if self.adapt_ramp:
            base_fieldnames.append("ramp_progress_ps")
        if self.adapt_dt:
            base_fieldnames.append("step_savings_pct")
        if self.p.log_charge_stats:
            for k in sorted(self.type_mass.keys()):
                base_fieldnames.append(f"q_t{k}_mean")
            base_fieldnames.append("q_std")
        self._log_writer = csv.DictWriter(
            self._log_file,
            fieldnames=base_fieldnames,
        )
        self._log_writer.writeheader()

        self._snap_idx = 0
        self._next_snap_fs = self.p.snap_every_ps * 1000.0

        # Initialize dt_prev: use dt_min if adaptive (allows growth from there), else fixed_dt
        self._dt_prev = self.p.dt_min_fs if self.adapt_dt else self.fixed_dt_fs
        self._current_tol: Optional[float] = None
        self._current_lang_damp: Optional[float] = None
        self._current_baro_pdamp: Optional[float] = None

        self._model_A: Optional[float] = None
        self._model_B: Optional[float] = None
        self._Sbar_tight: Optional[float] = None  # Sbar from converged forces (for dt when adapt_qeq)
        self._masses_cache: Optional[np.ndarray] = None  # per-atom masses, refreshed when types change
        self._current_qeq_nevery: int = params.qeq_nevery  # 1 or 2 when adapt_qeq_nevery
        self._calm_chunk_count: int = 0  # consecutive calm chunks for nevery hysteresis
        self._chunk_counter = 0
        self._forces_valid = False
        self._fix_changed = False  # Track if a fix was redefined and needs pre=yes

        init_tol = self.p.tol_fixed if not self.adapt_qeq else self.p.tol_min
        self._define_qeq(init_tol)
        self._current_tol = init_tol

        # Initialize Langevin damping (for logging, even when not adaptive)
        init_damp = self.p.lang_damp_fixed_fs
        self._current_lang_damp = init_damp
        if self.adapt_langevin:
            self._define_langevin(init_damp)

        # Ensure forces are current before auto-calibration
        self._run0()
        self._forces_valid = True
        # Cache masses for Sbar (types fixed for this run)
        types_init = self._extract_types()
        self._masses_cache = self._get_masses_array(types_init)

        if self.adapt_dt and self.p.dt_target0_fs is not None:
            Sbar = self._compute_Sbar_from_current()
            if Sbar > 0.0:
                # Calibrate Delta_l so that dt = dt_target0 at current (T_init, Sbar)
                # dt = Delta_l * sqrt(kBT / Sbar) => Delta_l = dt_target0 * sqrt(Sbar / kBT)
                T_init = max(self.p.lang_temp, 1.0)
                kBT = KB_KCAL_PER_MOL_K * T_init
                dl = float(self.p.dt_target0_fs) * math.sqrt(Sbar / kBT)
                self.p.Delta_l = max(1e-6, dl)
                print(f"[ETL] Auto-calibrated Delta_l = {self.p.Delta_l:.4f} "
                      f"(dt_target={self.p.dt_target0_fs} fs, Sbar={Sbar:.2f}, T={T_init:.0f} K)")
                dt_init = self.p.Delta_l * math.sqrt(kBT / Sbar)
                self._dt_prev = self._clamp(dt_init, self.p.dt_min_fs, self.p.dt_max_fs)

        # Adaptive ramp reference Sbar: set to an estimate of the *hot-phase*
        # Sbar so that calm phases (Sbar < ref) get speed_factor > 1 (faster
        # schedule traversal) while hot phases match real time (speed_factor = 1
        # due to the floor).  Using ~3.5x the cold-start Sbar is a reasonable
        # heuristic for ReaxFF systems ramped to > 4000 K.
        if self.adapt_ramp and self._ramp_Sbar_ref is None:
            Sbar_now = self._compute_Sbar_from_current() if self._forces_valid else 150.0
            self._ramp_Sbar_ref = max(Sbar_now * 3.5, 100.0)
            print(f"[ETL] Ramp Sbar_ref = {self._ramp_Sbar_ref:.1f} "
                  f"(3.5 × cold Sbar = {Sbar_now:.1f})")

        if self.adapt_qeq:
            self._calibrate_model()

    def close(self):
        if getattr(self, "_log_file", None):
            self._log_file.flush()
            self._log_file.close()
            self._log_file = None

    def _define_qeq(self, tol: float, nevery: Optional[int] = None) -> None:
        tol = float(tol)
        if nevery is not None:
            self._current_qeq_nevery = nevery
        nevery_val = self._current_qeq_nevery
        try:
            self.lmp.command("unfix q")
        except Exception:
            pass

        cmd = (
            f"fix q all qeq/reaxff {nevery_val} "
            f"{self.p.qeq_cutlo} {self.p.qeq_cuthi} {tol} reaxff "
            f"maxiter {self.p.qeq_maxiter}"
        )
        self.lmp.command(cmd)
        self._fix_changed = True

    def _define_langevin(self, damp_fs: float, temp: Optional[float] = None) -> None:
        """Define/redefine Langevin thermostat with unique RNG seed each time.

        Each call uses an incremented seed to avoid resetting the Langevin
        RNG to the same state every chunk.  Reusing a fixed seed would produce
        correlated random forces and artificial energy drift — worst for small
        dt where the per-step random-force amplitude is largest.
        """
        damp_fs = float(damp_fs)
        T = temp if temp is not None else self._current_lang_temp
        try:
            self.lmp.command("unfix lang")
        except Exception:
            pass

        self._lang_seed_counter += 1
        seed = self.p.lang_seed + self._lang_seed_counter
        cmd = (
            f"fix lang all langevin {T} {T} "
            f"{damp_fs} {seed}"
        )
        self.lmp.command(cmd)
        self._current_lang_temp = T
        self._current_lang_damp = damp_fs
        self._fix_changed = True
    
    def _update_langevin_temp(self, temp: float, damp_fs: Optional[float] = None) -> None:
        """Update Langevin thermostat temperature (redefines the fix)."""
        damp = damp_fs if damp_fs is not None else self._current_lang_damp
        if damp is None:
            damp = self.p.lang_damp_fixed_fs
        self._define_langevin(damp, temp)

    def _run0(self) -> None:
        self.lmp.command("run 0 post no")

    def _get_natoms(self) -> int:
        return self.lmp.get_natoms()

    def _extract_forces(self) -> np.ndarray:
        """Extract per-atom forces as (N, 3) numpy array (vectorized)."""
        n = self._get_natoms()
        f_ptr = self.lmp.extract_atom("f", LAMMPS_DOUBLE_2D)
        F = np.array([(f_ptr[i][0], f_ptr[i][1], f_ptr[i][2]) for i in range(n)], dtype=np.float64)
        return F

    def _extract_types(self) -> np.ndarray:
        """Extract per-atom types as (N,) numpy array."""
        n = self._get_natoms()
        t_ptr = self.lmp.extract_atom("type", LAMMPS_INT)
        return np.fromiter((t_ptr[i] for i in range(n)), dtype=np.int32, count=n)

    def _get_masses_array(self, types: np.ndarray) -> np.ndarray:
        masses = np.zeros(len(types), dtype=np.float64)
        for i, t in enumerate(types):
            m = self.type_mass.get(int(t))
            if not m:
                raise KeyError(f"Missing mass for atom type {t}. Provide type_mass_g_per_mol.")
            masses[i] = m
        return masses

    def _compute_Sbar_from_F(self, F: np.ndarray) -> float:
        """Compute Sbar = (1 / 3N) * sum_i (||F_i||^2 / m_i) from a force array."""
        n = len(F)
        if self._masses_cache is not None and len(self._masses_cache) == n:
            masses = self._masses_cache
        else:
            types = self._extract_types()
            masses = self._get_masses_array(types)
            self._masses_cache = masses
        if n == 0:
            return 0.0
        F_sq = np.sum(F**2, axis=1)
        return float(np.sum(F_sq / np.maximum(masses, 1e-12)) / (3.0 * n))

    def _compute_Sbar_from_current(self) -> float:
        """Compute Sbar from currently stored LAMMPS forces (no run 0)."""
        return self._compute_Sbar_from_F(self._extract_forces())

    def compute_Sbar(self) -> float:
        """
        Compute Sbar = (1 / 3N) * sum_i (||F_i||^2 / m_i).

        Uses stored forces if valid (from last run), otherwise evaluates via run 0.
        """
        if not self._forces_valid:
            self._run0()
            self._forces_valid = True
        return self._compute_Sbar_from_current()

    def _clamp(self, x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    def compute_dt_fs(self, Sbar: float, T_current: Optional[float] = None) -> float:
        """
        Compute adaptive timestep from Sbar (force power) and temperature.
        
        Uses the ETL-canonical (Fisher information) form:
            dt = Delta_l * sqrt(k_B*T / Sbar)
        so that each step covers equal thermodynamic length. The sqrt(k_B*T)
        factor partially compensates Sbar growth at high T, giving gentler
        contraction in heating simulations. Delta_l is calibrated once at start
        so that dt equals dt_target0_fs in the initial state.
        
        - Larger dt when Sbar is low or T is high (calm / hot equilibration)
        - Smaller dt when Sbar is high (demanding, reactive periods)
        """
        if not self.adapt_dt:
            return self.fixed_dt_fs

        T = T_current if T_current is not None else self.p.T_target
        T = max(float(T), 1.0)  # avoid div by zero

        if Sbar <= 0.0:
            dt_raw = self.p.dt_max_fs
        else:
            kBT = KB_KCAL_PER_MOL_K * T
            dt_raw = self.p.Delta_l * math.sqrt(kBT / Sbar)

        dt = self._clamp(dt_raw, self.p.dt_min_fs, self.p.dt_max_fs)

        # EWMA smoothing (only if alpha > 0)
        a = self._clamp(self.p.dt_smoothing_alpha, 0.0, 1.0)
        if a > 0.0:
            dt = a * dt + (1.0 - a) * self._dt_prev

        # Growth/shrink limits
        dt = min(dt, self._dt_prev * self.p.dt_growth)
        dt = max(dt, self._dt_prev / self.p.dt_shrink)

        dt = self._clamp(dt, self.p.dt_min_fs, self.p.dt_max_fs)
        
        # Apply dt hysteresis: only change if relative difference exceeds threshold
        if self.p.dt_hysteresis > 0.0 and self._dt_prev > 0.0:
            rel_change = abs(dt - self._dt_prev) / self._dt_prev
            if rel_change < self.p.dt_hysteresis:
                dt = self._dt_prev
        
        return dt

    def _calibrate_model(self) -> None:
        """
        Sentinel calibration: evaluate forces at loose and tight QEq tolerance
        to learn the mapping tol -> error.

        Two-point log-linear fit (matches proven notebook):
          Point 1: (log10(tol_loose), log10(err_bar))      — measured
          Point 2: (log10(tol_tight), log10(err_bar*1e-3))  — synthetic floor

        If no signal (err_bar <= 1e-10), sets model to None and uses tol_max.
        """
        if not self.adapt_qeq:
            return

        tight = self.p.tol_min
        loose = self.p.tol_max

        self._define_qeq(loose, nevery=1)
        self._run0()
        F_loose = self._extract_forces()

        self._define_qeq(tight, nevery=1)
        self._run0()
        F_tight = self._extract_forces()
        self._Sbar_tight = self._compute_Sbar_from_F(F_tight)

        # Next chunk will pick tol and call _define_qeq if needed; skip extra restore run 0
        self._forces_valid = True

        dF = F_loose - F_tight
        err_bar_loose = self._compute_Sbar_from_F(dF)

        EPS = 1e-30
        err_bar_loose = max(err_bar_loose, EPS)
        err_bar_tight = max(err_bar_loose * 1e-3, EPS)

        if err_bar_loose <= 1e-30:
            self._model_A = None
            self._model_B = None
            return

        x1 = math.log10(loose)
        y1 = math.log10(err_bar_loose)
        x2 = math.log10(tight)
        y2 = math.log10(err_bar_tight)

        if abs(x1 - x2) < 1e-12:
            self._model_A = None
            self._model_B = None
            return

        B = (y1 - y2) / (x1 - x2)
        A = y1 - B * x1

        self._model_A = A
        self._model_B = B

    def _pick_tol_for_dt(self, dt_fs: float, T_current: Optional[float] = None) -> float:
        """
        Given the current dt and temperature, compute the allowable QEq tolerance from the budget.

        cap = alpha_qeq * Delta_l^2 * kB * T / dt^2
        log10(tol) = (log10(cap) - A) / B
        """
        if not self.adapt_qeq:
            return float(self.p.tol_fixed)

        if self._model_A is None or self._model_B is None:
            return float(self.p.tol_max)

        T = T_current if T_current is not None else self.p.T_target
        kBT = KB_KCAL_PER_MOL_K * T
        cap = self.p.alpha_qeq * (self.p.Delta_l ** 2) * kBT / (dt_fs ** 2)

        if cap <= 0:
            return float(self.p.tol_min)

        log_cap = math.log10(max(cap, 1e-30))
        B = self._model_B if abs(self._model_B) > 1e-12 else 1.0
        log_tol = (log_cap - self._model_A) / B

        tol = 10.0 ** log_tol
        return self._clamp(tol, self.p.tol_min, self.p.tol_max)

    def maybe_update_qeq(self, tol_new: float, nevery: Optional[int] = None) -> None:
        if self._current_tol is None:
            nevery_val = nevery if nevery is not None else self._current_qeq_nevery
            self._define_qeq(tol_new, nevery=nevery_val)
            self._current_tol = float(tol_new)
            return

        tol_new = float(tol_new)
        tol_old = float(self._current_tol)

        ratio = max(tol_old, tol_new) / max(min(tol_old, tol_new), 1.0e-30)
        do_tol = ratio >= self.p.tol_hysteresis
        do_nevery = nevery is not None and nevery != self._current_qeq_nevery
        if do_tol or do_nevery:
            nevery_val = nevery if nevery is not None else self._current_qeq_nevery
            self._define_qeq(tol_new, nevery=nevery_val)
            self._current_tol = tol_new

    def compute_lang_damp(self, dt_fs: float) -> float:
        if not self.adapt_langevin:
            return float(self.p.lang_damp_fixed_fs)

        damp_raw = self.p.lang_damp_ratio * dt_fs
        return self._clamp(damp_raw, self.p.lang_damp_min_fs, self.p.lang_damp_max_fs)

    def maybe_update_langevin(self, damp_new: float) -> None:
        if not self.adapt_langevin:
            return

        if self._current_lang_damp is None:
            self._define_langevin(damp_new)
            self._current_lang_damp = float(damp_new)
            return

        damp_new = float(damp_new)
        damp_old = float(self._current_lang_damp)

        ratio = max(damp_old, damp_new) / max(min(damp_old, damp_new), 1.0e-30)
        if ratio >= self.p.lang_hysteresis:
            self._define_langevin(damp_new)
            self._current_lang_damp = damp_new

    def compute_baro_pdamp(self, dt_fs: float) -> float:
        if not self.adapt_barostat:
            return float(self.p.baro_pdamp_fixed_fs)

        pdamp_raw = self.p.baro_pdamp_ratio * dt_fs
        return self._clamp(pdamp_raw, self.p.baro_pdamp_min_fs, self.p.baro_pdamp_max_fs)

    def _get_thermo(self, name: str) -> float:
        try:
            return float(self.lmp.get_thermo(name))
        except Exception:
            return float("nan")

    def _get_charge_stats(self) -> Dict[str, float]:
        """Extract per-atom charge and return per-type means and overall std. Returns empty dict on failure."""
        if not self.p.log_charge_stats:
            return {}
        try:
            n = self._get_natoms()
            # LAMMPS Python: extract_atom("q", LAMMPS_DOUBLE) for per-atom scalar double (dtype 2, not 1)
            q_ptr = self.lmp.extract_atom("q", LAMMPS_DOUBLE)
            if q_ptr is None:
                return {}
            q = np.array([q_ptr[i] for i in range(n)], dtype=np.float64)
            types = self._extract_types()
            out = {}
            for t in sorted(self.type_mass.keys()):
                mask = types == t
                if np.any(mask):
                    out[f"q_t{t}_mean"] = float(np.mean(q[mask]))
                else:
                    out[f"q_t{t}_mean"] = float("nan")
            out["q_std"] = float(np.std(q)) if len(q) > 0 else float("nan")
            return out
        except Exception:
            return {}

    def _write_dump(self, time_fs: float) -> None:
        t_ps = time_fs / 1000.0
        fname = os.path.join(self.snap_dir, f"frame_{self._snap_idx:06d}_{t_ps:010.3f}ps.dump")
        self.lmp.command(f"write_dump all custom {fname} id type q x y z")
        self._snap_idx += 1

    def run(self, t_ps: float) -> None:
        """
        Run the simulation for t_ps picoseconds using the configured ETL settings.
        
        If a T_schedule is set, the thermostat target temperature is updated each chunk.
        If adapt_T is enabled, temperature steps are determined adaptively based on Sbar
        and energy variance.
        
        The instantaneous target T is used for kBT scaling in dt and QEq tolerance formulas.
        """
        target_fs = float(t_ps) * 1000.0
        time_fs = 0.0
        step_index = 0
        
        # Track previous thermostat temperature to detect changes
        prev_T_target: Optional[float] = None

        try:
            while time_fs < target_fs:
                # Get current target temperature
                time_ps = time_fs / 1000.0
                
                Sbar = self.compute_Sbar()

                if self.adapt_T and self._adaptive_T_controller is not None:
                    etotal_prev = self._get_thermo("etotal")
                    T_target = self._adaptive_T_controller.step(Sbar, etotal_prev)
                elif self.adapt_ramp and self.T_schedule is not None:
                    # ETL-driven ramp: advance virtual clock based on Sbar.
                    # speed_factor >= 1 always (never slower than real time).
                    # Calm phases (Sbar < ref) → speed_factor > 1 → ramp faster.
                    # Hard phases (Sbar >= ref) → speed_factor = 1 → ramp = real time.
                    Sbar_safe = max(Sbar, 1.0)
                    speed_factor = max(1.0, math.sqrt(self._ramp_Sbar_ref / Sbar_safe))
                    chunk_wall_ps = (self._dt_prev * self.p.chunk_steps) / 1000.0 if step_index > 0 else 0.0
                    self._ramp_progress_ps += chunk_wall_ps * speed_factor
                    self._ramp_progress_ps = min(self._ramp_progress_ps, t_ps)
                    T_target = self.T_schedule(self._ramp_progress_ps)
                elif self.T_schedule is not None:
                    T_target = self.T_schedule(time_ps)
                else:
                    T_target = self.p.T_target
                
                Sbar_for_dt = (
                    self._Sbar_tight
                    if (self.adapt_qeq and self._Sbar_tight is not None)
                    else Sbar
                )
                dt_fs = self.compute_dt_fs(Sbar_for_dt, T_current=T_target)

                if self.adapt_qeq:
                    tol = self._pick_tol_for_dt(dt_fs, T_current=T_target)
                else:
                    tol = self.p.tol_fixed

                desired_nevery: Optional[int] = None
                if self.adapt_qeq_nevery and self.adapt_qeq:
                    Sbar_calm = (0.5 * self._ramp_Sbar_ref) if (self._ramp_Sbar_ref is not None) else 200.0
                    dt_calm = 0.35
                    calm = Sbar < Sbar_calm and dt_fs > dt_calm
                    if calm:
                        self._calm_chunk_count += 1
                    else:
                        self._calm_chunk_count = 0
                    desired_nevery = 2 if (calm and self._calm_chunk_count >= 2) else 1
                self.maybe_update_qeq(tol, nevery=desired_nevery)

                # Update thermostat temperature if it changed
                if prev_T_target is None or abs(T_target - prev_T_target) > 0.1:
                    damp = self._current_lang_damp if self._current_lang_damp else self.p.lang_damp_fixed_fs
                    self._update_langevin_temp(T_target, damp)
                    prev_T_target = T_target

                lang_damp = self.compute_lang_damp(dt_fs)
                self.maybe_update_langevin(lang_damp)

                baro_pdamp = self.compute_baro_pdamp(dt_fs)

                remaining_fs = target_fs - time_fs
                nsteps = min(self.p.chunk_steps, max(1, int(math.ceil(remaining_fs / dt_fs))))

                self.lmp.command(f"timestep {dt_fs}")
                
                # If a fix was redefined, we need pre=yes to re-initialize it
                if self._fix_changed:
                    self.lmp.command(f"run {nsteps} post no")
                    self._fix_changed = False
                else:
                    self.lmp.command(f"run {nsteps} pre no post no")

                # Forces are current after run; no need for run 0 next iteration
                self._forces_valid = True

                time_fs += dt_fs * float(nsteps)
                step_index += nsteps

                self._dt_prev = dt_fs

                temp = self._get_thermo("temp")
                press = self._get_thermo("press")
                pe = self._get_thermo("pe")
                ke = self._get_thermo("ke")
                etotal = self._get_thermo("etotal")

                time_ps_val = time_fs / 1000.0
                row = {
                    "step_index": step_index,
                    "time_fs": time_fs,
                    "time_ps": time_ps_val,
                    "dt_fs": dt_fs,
                    "tol": float(self._current_tol) if self._current_tol is not None else float("nan"),
                    "lang_damp_fs": float(self._current_lang_damp) if self._current_lang_damp is not None else float("nan"),
                    "baro_pdamp_fs": float(baro_pdamp) if self.adapt_barostat else float("nan"),
                    "T_target": T_target,
                    "Sbar": Sbar,
                    "temp": temp,
                    "press": press,
                    "pe": pe,
                    "ke": ke,
                    "etotal": etotal,
                    "schedule_ps": self._ramp_progress_ps if self.adapt_ramp else time_ps_val,
                }
                if self.adapt_ramp:
                    row["ramp_progress_ps"] = self._ramp_progress_ps
                if self.adapt_dt and time_fs > 0:
                    ref_dt = max(1e-6, self.p.step_savings_reference_dt_fs)
                    steps_safe = time_fs / ref_dt
                    row["step_savings_pct"] = 100.0 * max(0.0, min(100.0, 1.0 - step_index / steps_safe))
                if self.p.log_charge_stats:
                    row.update(self._get_charge_stats())
                self._log_writer.writerow(row)
                if getattr(self.p, "progress_callback", None) is not None:
                    try:
                        self.p.progress_callback(row)
                    except Exception:
                        pass

                while time_fs >= self._next_snap_fs:
                    self._write_dump(self._next_snap_fs)
                    self._next_snap_fs += self.p.snap_every_ps * 1000.0

                self._chunk_counter += 1
                if self.adapt_qeq and self.p.cal_every > 0 and self._chunk_counter % self.p.cal_every == 0:
                    self._calibrate_model()
                    # Re-pick tol using updated model (with current temperature)
                    if self.adapt_qeq:
                        tol = self._pick_tol_for_dt(dt_fs, T_current=T_target)
                        self.maybe_update_qeq(tol)
                
                # Adaptive ramp: early exit when virtual clock completes the
                # full temperature schedule (all phases traversed).
                if self.adapt_ramp and self._ramp_progress_ps >= t_ps:
                    print(f"[ETL] Ramp completed full schedule at t={time_ps_val:.2f} ps "
                          f"(saved {(1.0 - time_ps_val/t_ps)*100:.1f}% of simulation time)")
                    break

                if self.adapt_T and self._adaptive_T_controller is not None:
                    if self._adaptive_T_controller.is_complete():
                        pass

        finally:
            self.close()
