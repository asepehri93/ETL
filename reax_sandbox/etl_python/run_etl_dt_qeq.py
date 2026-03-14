#!/usr/bin/env python3
"""Stage C: ETL adaptive dt + adaptive QEq tolerance"""
import argparse

from run_common import run_case, DEFAULT_T_TARGET


def main():
    ap = argparse.ArgumentParser(description="Stage C: ETL adaptive dt + adaptive QEq tolerance")
    ap.add_argument("--t-ps", type=float, default=50.0, help="Run length in ps")
    ap.add_argument("--T", type=float, default=DEFAULT_T_TARGET, help="Target temperature in K")
    ap.add_argument(
        "--dt-target0-fs",
        type=float,
        default=0.25,
        help="Auto-calibrate Delta_l so the first proposed dt ~= this value.",
    )
    args = ap.parse_args()

    run_case(
        out_dir="etl_dt_qeq_outputs",
        t_ps=args.t_ps,
        adapt_dt=True,
        adapt_qeq=True,
        dt_target0_fs=args.dt_target0_fs,
        T_target=args.T,
    )


if __name__ == "__main__":
    main()
