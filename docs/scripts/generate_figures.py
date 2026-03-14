#!/usr/bin/env python3
"""
Generate ETL report figures (main + supplementary) from etl_log.csv and wall_time.txt.

Run from repo root with the project venv active:
  source .venv/bin/activate
  pip install matplotlib   # if needed
  python docs/scripts/generate_figures.py

Uses reax_sandbox/sio2_etl/ as base for output dirs; falls back to hardcoded report
data when CSV/wall_time files are missing.
"""
from __future__ import annotations

import os
import sys
import csv

# Repo root (parent of docs/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
FIGURES_DIR = os.path.join(REPO_ROOT, "docs", "figures")
SUPP_DIR = os.path.join(REPO_ROOT, "docs", "figures", "supplementary")
BASE_OUT = os.path.join(REPO_ROOT, "reax_sandbox", "sio2_etl")

CASE_ORDER = [
    "baseline_reference",
    "baseline_moderate",
    "baseline_aggressive",
    "etl_dt",
    "etl_dt_ramp",
    "etl_qeq",
    "etl_full",
]

# Fallback from report §3.1 and §3.7 (wall time seconds)
WALL_TIME_576 = {
    "hat": {"baseline_reference": 6729, "baseline_moderate": 3209, "baseline_aggressive": 1854,
            "etl_dt": 2725, "etl_dt_ramp": 2386, "etl_qeq": 2668, "etl_full": 2314},
    "milder": {"baseline_reference": 6665, "baseline_moderate": 3203, "baseline_aggressive": 1827,
              "etl_dt": 2832, "etl_dt_ramp": 2379, "etl_qeq": 2873, "etl_full": 2322},
    "constant_T": {"baseline_reference": 6213, "baseline_moderate": 2944, "baseline_aggressive": 1735,
                   "etl_dt": 2553, "etl_dt_ramp": 1628, "etl_qeq": 2638, "etl_full": 1847},
}
WALL_TIME_1500 = {
    "hat": {"baseline_reference": 14033.7, "baseline_moderate": 5077.7, "baseline_aggressive": 3443.5,
            "etl_dt": 5675.6, "etl_dt_ramp": 5139.1, "etl_qeq": 5504.7, "etl_full": 3685.6},
    "constant_T": {"baseline_reference": 12980.6, "baseline_moderate": 4665.0, "baseline_aggressive": 3228.1,
                  "etl_dt": 5645.9, "etl_dt_ramp": 4278.7, "etl_qeq": 5760.1, "etl_full": 3174.4},
}


def _ensure_dirs():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(SUPP_DIR, exist_ok=True)


def _load_csv(path: str) -> list[dict] | None:
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception:
        return None


def _read_wall_time(dir_path: str) -> float | None:
    wt_path = os.path.join(dir_path, "wall_time.txt")
    if not os.path.isfile(wt_path):
        return None
    try:
        with open(wt_path, encoding="utf-8") as f:
            return float(f.read().strip())
    except Exception:
        return None


def _get_output_dir(schedule: str, case: str, large: bool) -> str:
    if large:
        return os.path.join(BASE_OUT, f"outputs_sio2_large_{schedule}_{case}")
    return os.path.join(BASE_OUT, f"outputs_sio2_v4_{schedule}_{case}")


def fig01_control_loop():
    """Fig. 1: ETL control loop schematic."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(1, 1, figsize=(8, 5.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    style = dict(boxstyle="round,pad=0.3", facecolor="lightgray", edgecolor="black", linewidth=1.5)
    # Boxes: (x, y) = lower-left, width, height. Top row y=3.8, bottom row y=1.5
    ax.add_patch(FancyBboxPatch((1, 3.8), 2, 1.4, **style))   # Trajectory
    ax.add_patch(FancyBboxPatch((4, 3.8), 2, 1.4, **style))   # S̄,T → dt formula
    ax.add_patch(FancyBboxPatch((7, 3.8), 2, 1.4, **style))   # Δt
    ax.add_patch(FancyBboxPatch((4, 1.5), 2, 1.4, **style))   # QEq cap → tol
    ax.add_patch(FancyBboxPatch((7, 1.5), 2, 1.4, **style))   # Virtual clock
    # Labels (centers: x=2,5,8 for cols; y=4.5 top, 2.2 bottom)
    ax.text(2, 4.5, "Trajectory\nF, T", ha="center", va="center", fontsize=10)
    ax.text(5, 4.5, r"$\bar{S}$, $T$" + "\n→ dt formula", ha="center", va="center", fontsize=10)
    ax.text(8, 4.5, r"$\Delta t$", ha="center", va="center", fontsize=11)
    ax.text(5, 2.2, "QEq cap → tol", ha="center", va="center", fontsize=9)
    ax.text(8, 2.2, "Virtual clock\nspeed_factor", ha="center", va="center", fontsize=9)
    # Arrows: short segments in the gap between boxes so they don't overlap text
    # Top row: Trajectory → formula → Δt
    ax.annotate("", xy=(3.55, 4.5), xytext=(2.45, 4.5),
                arrowprops=dict(arrowstyle="->", lw=1.8, color="black"))
    ax.annotate("", xy=(6.55, 4.5), xytext=(5.45, 4.5),
                arrowprops=dict(arrowstyle="->", lw=1.8, color="black"))
    # Bottom row: QEq cap → Virtual clock
    ax.annotate("", xy=(6.55, 2.2), xytext=(5.45, 2.2),
                arrowprops=dict(arrowstyle="->", lw=1.8, color="black"))
    # Δt feeds into QEq cap (cap ∝ 1/Δt²): straight diagonal from Δt bottom to QEq top
    ax.annotate("", xy=(5.45, 2.9), xytext=(7.35, 3.82),
                arrowprops=dict(arrowstyle="->", lw=1.2, color="black"))
    ax.text(5, 0.7, "Fig. 1: ETL control loop (dt, QEq budget, optional ramp)", fontsize=9, ha="center")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig01_control_loop.png"), dpi=150, bbox_inches="tight")
    plt.close()


def fig02_virtual_clock(data_dir: str | None):
    """Fig. 2: Virtual schedule clock — simulated time vs virtual time (s)."""
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    if data_dir:
        rows = _load_csv(os.path.join(data_dir, "etl_log.csv"))
        if rows and "time_ps" in rows[0] and "ramp_progress_ps" in rows[0]:
            time_ps = [float(r["time_ps"]) for r in rows]
            ramp_ps = [float(r.get("ramp_progress_ps", r["time_ps"])) for r in rows]
            ax.plot(time_ps, ramp_ps, "b-", lw=2, label="Virtual time s(t)")
            ax.plot(time_ps, time_ps, "k--", alpha=0.7, label="Simulated time t")
            ax.set_xlabel("Simulated time (ps)")
            ax.set_ylabel("Virtual schedule time (ps)")
            ax.legend()
            ax.set_xlim(0, max(time_ps) * 1.02)
            ax.set_ylim(0, 10.5)
            ax.axhline(10, color="gray", ls=":", alpha=0.7)
            ax.grid(True, alpha=0.3)
        else:
            _fig02_schematic(ax)
    else:
        _fig02_schematic(ax)
    ax.set_title("Fig. 2: Virtual schedule clock (constant T: s runs ahead, early exit)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig02_virtual_clock.png"), dpi=150, bbox_inches="tight")
    plt.close()


def _fig02_schematic(ax):
    t = np.linspace(0, 10, 200)
    ax.plot(t, t, "k--", alpha=0.7, label="Simulated time t")
    # s runs ahead: e.g. s = t * 1.5 capped at 10
    s = np.minimum(t * 1.6, 10)
    ax.plot(t, s, "b-", lw=2, label="Virtual time s(t)")
    ax.axhline(10, color="gray", ls=":", alpha=0.7)
    ax.set_xlabel("Simulated time (ps)")
    ax.set_ylabel("Virtual schedule time (ps)")
    ax.legend()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10.5)
    ax.grid(True, alpha=0.3)


def fig03_wall_time():
    """Fig. 3: Wall time by case (bar chart)."""
    import matplotlib.pyplot as plt
    import numpy as np

    cases_short = ["ref", "mod", "agg", "dt", "dt+r", "qeq", "full"]
    x = np.arange(len(cases_short))
    width = 0.25

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    # 576: hat and constant_T
    wt_hat = [WALL_TIME_576["hat"][c] for c in CASE_ORDER]
    wt_const = [WALL_TIME_576["constant_T"][c] for c in CASE_ORDER]
    ax1.bar(x - width, wt_hat, width, label="Hat", color="C0", alpha=0.8)
    ax1.bar(x, wt_const, width, label="Constant T", color="C1", alpha=0.8)
    ax1.set_ylabel("Wall time (s)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(cases_short, rotation=15)
    ax1.legend()
    ax1.set_title("576 atoms")
    ax1.grid(True, axis="y", alpha=0.3)

    wt_hat_1500 = [WALL_TIME_1500["hat"][c] for c in CASE_ORDER]
    wt_const_1500 = [WALL_TIME_1500["constant_T"][c] for c in CASE_ORDER]
    ax2.bar(x - width, wt_hat_1500, width, label="Hat", color="C0", alpha=0.8)
    ax2.bar(x, wt_const_1500, width, label="Constant T", color="C1", alpha=0.8)
    ax2.set_ylabel("Wall time (s)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(cases_short, rotation=15)
    ax2.legend()
    ax2.set_title("1500 atoms")
    ax2.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Fig. 3: Wall time by case and schedule")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig03_wall_time.png"), dpi=150, bbox_inches="tight")
    plt.close()


def fig04_speedup_vs_moderate():
    """Fig. 4: Speedup vs moderate baseline."""
    import matplotlib.pyplot as plt
    import numpy as np

    mod_576_hat = WALL_TIME_576["hat"]["baseline_moderate"]
    mod_576_const = WALL_TIME_576["constant_T"]["baseline_moderate"]
    mod_1500_hat = WALL_TIME_1500["hat"]["baseline_moderate"]
    mod_1500_const = WALL_TIME_1500["constant_T"]["baseline_moderate"]

    speedup_576_hat = [mod_576_hat / WALL_TIME_576["hat"][c] for c in CASE_ORDER]
    speedup_576_const = [mod_576_const / WALL_TIME_576["constant_T"][c] for c in CASE_ORDER]
    speedup_1500_hat = [mod_1500_hat / WALL_TIME_1500["hat"][c] for c in CASE_ORDER]
    speedup_1500_const = [mod_1500_const / WALL_TIME_1500["constant_T"][c] for c in CASE_ORDER]

    cases_short = ["ref", "mod", "agg", "dt", "dt+r", "qeq", "full"]
    x = np.arange(len(cases_short))
    width = 0.2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.axhline(1.0, color="gray", ls="--", alpha=0.8, label="Moderate (1×)")
    ax1.bar(x - 1.5 * width, speedup_576_hat, width, label="576 hat", color="C0", alpha=0.8)
    ax1.bar(x - 0.5 * width, speedup_576_const, width, label="576 const T", color="C1", alpha=0.8)
    ax1.set_ylabel("Speedup vs moderate")
    ax1.set_xticks(x)
    ax1.set_xticklabels(cases_short, rotation=15)
    ax1.legend(fontsize=8)
    ax1.set_title("576 atoms")
    ax1.set_ylim(0, 2.2)
    ax1.grid(True, axis="y", alpha=0.3)

    ax2.axhline(1.0, color="gray", ls="--", alpha=0.8)
    ax2.bar(x - 0.5 * width, speedup_1500_hat, width, label="1500 hat", color="C0", alpha=0.8)
    ax2.bar(x + 0.5 * width, speedup_1500_const, width, label="1500 const T", color="C1", alpha=0.8)
    ax2.set_ylabel("Speedup vs moderate")
    ax2.set_xticks(x)
    ax2.set_xticklabels(cases_short, rotation=15)
    ax2.legend(fontsize=8)
    ax2.set_title("1500 atoms")
    ax2.set_ylim(0, 2.2)
    ax2.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Fig. 4: Speedup vs moderate baseline (0.25 fs)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig04_speedup_vs_moderate.png"), dpi=150, bbox_inches="tight")
    plt.close()


def supp_simulated_vs_wall_time(schedule: str = "hat", large: bool = False):
    """Supplementary: Simulated time vs estimated elapsed wall time, one schedule, all cases."""
    import matplotlib.pyplot as plt

    prefix = "large_" if large else "v4_"
    case_data = []
    for case in CASE_ORDER:
        out_dir = os.path.join(BASE_OUT, f"outputs_sio2_{prefix}{schedule}_{case}")
        csv_path = os.path.join(out_dir, "etl_log.csv")
        wt = _read_wall_time(out_dir)
        rows = _load_csv(csv_path)
        if not rows or wt is None:
            continue
        try:
            step_final = int(rows[-1]["step_index"])
            time_ps = [float(r["time_ps"]) for r in rows]
            wall_est = [wt * int(r["step_index"]) / step_final for r in rows]
            case_data.append((case, time_ps, wall_est))
        except (KeyError, ValueError, IndexError):
            continue
    if not case_data:
        return
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    for case, time_ps, wall_est in case_data:
        label = case.replace("baseline_", "b.").replace("etl_", "")
        ax.plot(time_ps, wall_est, label=label, lw=1.5)
    ax.set_xlabel("Simulated time (ps)")
    ax.set_ylabel("Estimated elapsed wall time (s)")
    ax.set_title(f"Simulated vs wall time — {schedule}" + (" (1500 at)" if large else " (576 at)"))
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    name = f"fig_s01_simulated_vs_wall_{schedule}" + ("_1500" if large else "_576") + ".png"
    fig.savefig(os.path.join(SUPP_DIR, name), dpi=150, bbox_inches="tight")
    plt.close()


def supp_controller_behavior(data_dir: str, case_label: str, out_name: str):
    """Supplementary: dt, ramp_progress_ps, tol, Sbar vs time_ps for one run."""
    import matplotlib.pyplot as plt

    rows = _load_csv(os.path.join(data_dir, "etl_log.csv"))
    if not rows or len(rows) < 2:
        return
    try:
        time_ps = [float(r["time_ps"]) for r in rows]
        fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
        # dt
        axes[0, 0].plot(time_ps, [float(r.get("dt_fs", 0)) for r in rows], "b-", lw=0.8)
        axes[0, 0].set_ylabel("dt (fs)")
        axes[0, 0].grid(True, alpha=0.3)
        # ramp
        axes[0, 1].plot(time_ps, [float(r.get("ramp_progress_ps", r["time_ps"])) for r in rows], "g-", lw=0.8)
        axes[0, 1].plot(time_ps, time_ps, "k--", alpha=0.5)
        axes[0, 1].set_ylabel("Ramp progress (ps)")
        axes[0, 1].grid(True, alpha=0.3)
        # tol (log)
        tols = [float(r.get("tol", 1e-5)) for r in rows]
        axes[1, 0].semilogy(time_ps, tols, "r-", lw=0.8)
        axes[1, 0].set_ylabel("QEq tol")
        axes[1, 0].set_xlabel("Simulated time (ps)")
        axes[1, 0].grid(True, alpha=0.3)
        # Sbar
        axes[1, 1].plot(time_ps, [float(r.get("Sbar", 0)) for r in rows], "m-", lw=0.8)
        axes[1, 1].set_ylabel(r"$\bar{S}$")
        axes[1, 1].set_xlabel("Simulated time (ps)")
        axes[1, 1].grid(True, alpha=0.3)
        fig.suptitle(f"Controller behavior — {case_label}")
        fig.tight_layout()
        fig.savefig(os.path.join(SUPP_DIR, out_name), dpi=150, bbox_inches="tight")
        plt.close()
    except (KeyError, ValueError) as e:
        print(f"  Skip controller behavior {case_label}: {e}", file=sys.stderr)


def supp_fidelity_trajectories(schedule: str = "hat", large: bool = False):
    """Supplementary: etotal, press, q_t1_mean, q_std vs time for selected cases."""
    import matplotlib.pyplot as plt

    prefix = "large_" if large else "v4_"
    selected = ["baseline_reference", "baseline_moderate", "baseline_aggressive", "etl_full"]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
    for case in selected:
        out_dir = os.path.join(BASE_OUT, f"outputs_sio2_{prefix}{schedule}_{case}")
        rows = _load_csv(os.path.join(out_dir, "etl_log.csv"))
        if not rows:
            continue
        try:
            time_ps = [float(r["time_ps"]) for r in rows]
            label = case.replace("baseline_", "b.").replace("etl_", "")
            axes[0, 0].plot(time_ps, [float(r.get("etotal", 0)) for r in rows], label=label, lw=0.8)
            axes[0, 1].plot(time_ps, [float(r.get("press", 0)) for r in rows], label=label, lw=0.8)
            q1 = [float(r.get("q_t1_mean", 0)) for r in rows if r.get("q_t1_mean", "nan") != "nan"]
            qstd = [float(r.get("q_std", 0)) for r in rows if r.get("q_std", "nan") != "nan"]
            if q1 and len(q1) == len(time_ps):
                axes[1, 0].plot(time_ps, q1, label=label, lw=0.8)
            if qstd and len(qstd) == len(time_ps):
                axes[1, 1].plot(time_ps, qstd, label=label, lw=0.8)
        except (KeyError, ValueError):
            continue
    axes[0, 0].set_ylabel("Etotal")
    axes[0, 0].legend(fontsize=7)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].set_ylabel("Pressure")
    axes[0, 1].legend(fontsize=7)
    axes[0, 1].grid(True, alpha=0.3)
    axes[1, 0].set_ylabel("q_t1 (Si)")
    axes[1, 0].set_xlabel("Simulated time (ps)")
    axes[1, 0].legend(fontsize=7)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 1].set_ylabel("q_std")
    axes[1, 1].set_xlabel("Simulated time (ps)")
    axes[1, 1].legend(fontsize=7)
    axes[1, 1].grid(True, alpha=0.3)
    fig.suptitle(f"Fidelity — {schedule}" + (" (1500 at)" if large else " (576 at)"))
    fig.tight_layout()
    name = f"fig_s02_fidelity_{schedule}" + ("_1500" if large else "_576") + ".png"
    fig.savefig(os.path.join(SUPP_DIR, name), dpi=150, bbox_inches="tight")
    plt.close()


def supp_temperature_schedule():
    """Supplementary: Temperature schedule schematic (hat, milder, constant T)."""
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    s = np.linspace(0, 10, 500)
    # Hat: 20% cold, 30% up, 30% down, 20% cold
    t_ps = 10
    t_hat = np.where(s <= 2, 300, np.where(s <= 5, 300 + (4500 - 300) * (s - 2) / 3,
                 np.where(s <= 8, 4500 - (4500 - 300) * (s - 5) / 3, 300)))
    # Milder: 25% each
    t_milder = np.where(s <= 2.5, 300, np.where(s <= 5, 300 + (2000 - 300) * (s - 2.5) / 2.5,
                    np.where(s <= 7.5, 2000 - (2000 - 300) * (s - 5) / 2.5, 300)))
    ax.plot(s, t_hat, "b-", lw=2, label="Hat (300→4500→300 K)")
    ax.plot(s, t_milder, "g-", lw=1.5, label="Milder (300→2000→300 K)")
    ax.axhline(300, color="gray", ls="--", alpha=0.7, label="Constant T = 300 K")
    ax.set_xlabel("Virtual schedule time (ps)")
    ax.set_ylabel("T (K)")
    ax.legend()
    ax.set_xlim(0, 10)
    ax.grid(True, alpha=0.3)
    fig.suptitle("Temperature schedules (10 ps)")
    fig.tight_layout()
    fig.savefig(os.path.join(SUPP_DIR, "fig_s03_temperature_schedule.png"), dpi=150, bbox_inches="tight")
    plt.close()


def supp_dashboard(schedule: str = "hat", large: bool = False):
    """Supplementary: 2×2 dashboard — wall time, speedup, etotal vs time, dt vs time."""
    import matplotlib.pyplot as plt
    import numpy as np

    prefix = "large_" if large else "v4_"
    mod_key = "baseline_moderate"
    if large:
        wt_dict = WALL_TIME_1500.get(schedule, {})
    else:
        wt_dict = WALL_TIME_576.get(schedule, {})
    mod_wt = wt_dict.get(mod_key, 1.0)
    speedups = [mod_wt / wt_dict.get(c, 1.0) for c in CASE_ORDER]
    cases_short = ["ref", "mod", "agg", "dt", "dt+r", "qeq", "full"]

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    x = np.arange(len(CASE_ORDER))
    axes[0, 0].bar(x, [wt_dict.get(c, 0) for c in CASE_ORDER], color="steelblue", alpha=0.8)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(cases_short, rotation=15)
    axes[0, 0].set_ylabel("Wall time (s)")
    axes[0, 0].set_title("Wall time")
    axes[0, 0].grid(True, axis="y", alpha=0.3)

    axes[0, 1].axhline(1.0, color="gray", ls="--")
    axes[0, 1].bar(x, speedups, color="coral", alpha=0.8)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(cases_short, rotation=15)
    axes[0, 1].set_ylabel("Speedup vs moderate")
    axes[0, 1].set_title("Speedup")
    axes[0, 1].grid(True, axis="y", alpha=0.3)

    # etotal and dt from first available ETL run
    for case in ["etl_full", "baseline_moderate"]:
        out_dir = os.path.join(BASE_OUT, f"outputs_sio2_{prefix}{schedule}_{case}")
        rows = _load_csv(os.path.join(out_dir, "etl_log.csv"))
        if not rows:
            continue
        try:
            time_ps = [float(r["time_ps"]) for r in rows]
            axes[1, 0].plot(time_ps, [float(r.get("etotal", 0)) for r in rows], label=case.replace("_", " "), lw=1)
            axes[1, 1].plot(time_ps, [float(r.get("dt_fs", 0)) for r in rows], label=case.replace("_", " "), lw=1)
        except (KeyError, ValueError):
            pass
    axes[1, 0].set_xlabel("Simulated time (ps)")
    axes[1, 0].set_ylabel("Etotal")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].set_title("Energy")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 1].set_xlabel("Simulated time (ps)")
    axes[1, 1].set_ylabel("dt (fs)")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].set_title("Timestep")
    axes[1, 1].grid(True, alpha=0.3)
    fig.suptitle(f"Summary dashboard — {schedule}" + (" (1500 at)" if large else " (576 at)"))
    fig.tight_layout()
    fig.savefig(os.path.join(SUPP_DIR, f"fig_s04_dashboard_{schedule}" + ("_1500" if large else "_576") + ".png"),
                dpi=150, bbox_inches="tight")
    plt.close()


def main():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        print("matplotlib required: pip install matplotlib", file=sys.stderr)
        sys.exit(1)

    _ensure_dirs()
    os.chdir(REPO_ROOT)

    # Main figures
    print("Generating main figures...")
    fig01_control_loop()
    # Fig 2: use data from large constant_T etl_full if available
    data_dir_clock = _get_output_dir("constant_T", "etl_full", large=True)
    if not os.path.isdir(data_dir_clock):
        data_dir_clock = _get_output_dir("hat", "etl_full", large=False)
    fig02_virtual_clock(data_dir_clock if os.path.isdir(data_dir_clock) else None)
    fig03_wall_time()
    fig04_speedup_vs_moderate()
    print("  Main figures saved to docs/figures/")

    # Supplementary
    print("Generating supplementary figures...")
    supp_temperature_schedule()
    for sched in ["hat", "constant_T"]:
        supp_simulated_vs_wall_time(schedule=sched, large=False)
        supp_simulated_vs_wall_time(schedule=sched, large=True)
    supp_fidelity_trajectories("hat", large=False)
    supp_fidelity_trajectories("hat", large=True)
    supp_dashboard("hat", large=False)
    supp_dashboard("constant_T", large=False)
    # Controller behavior for one ETL run
    for label, sched, case in [("etl_full hat", "hat", "etl_full"), ("etl_full const T", "constant_T", "etl_full")]:
        d = _get_output_dir(sched, case, large=False)
        if os.path.isdir(d):
            supp_controller_behavior(d, label, f"fig_s05_controller_{sched}_{case}.png")
            break
    # 1500-atom controller
    d = _get_output_dir("constant_T", "etl_full", large=True)
    if os.path.isdir(d):
        supp_controller_behavior(d, "etl_full const T (1500)", "fig_s06_controller_1500_constant_T_etl_full.png")
    print("  Supplementary figures saved to docs/figures/supplementary/")
    print("Done.")


if __name__ == "__main__":
    main()
