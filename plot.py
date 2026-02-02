from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from model import load_params, run_verification_sweep, with_penalty_scale, with_cap_at_pledge


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def _write_csv(outdir: Path, metrics) -> None:
    v = metrics.v_values
    N = metrics.flag_rate.shape[1]

    # Per-actor outputs
    header = "v," + ",".join([f"actor_{i+1}" for i in range(N)])
    np.savetxt(outdir / "flag_rate_vs_v.csv", np.column_stack([v, metrics.flag_rate]), delimiter=",", header=header)
    np.savetxt(
        outdir / "expected_utility_vs_v.csv",
        np.column_stack([v, metrics.expected_utility]),
        delimiter=",",
        header=header,
    )
    np.savetxt(
        outdir / "compliance_gap_vs_v.csv",
        np.column_stack([v, metrics.compliance_gap]),
        delimiter=",",
        header=header,
    )

    # Scalar outputs
    np.savetxt(
        outdir / "gini_compute_vs_v.csv",
        np.column_stack([v, metrics.gini_compute]),
        delimiter=",",
        header="v,gini_compute",
    )
    np.savetxt(
        outdir / "total_welfare_vs_v.csv",
        np.column_stack([v, metrics.total_welfare]),
        delimiter=",",
        header="v,total_welfare",
    )
    np.savetxt(
        outdir / "cumulative_capability_vs_v.csv",
        np.column_stack([v, metrics.cumulative_capability]),
        delimiter=",",
        header="v,cumulative_capability_q_T",
    )
    np.savetxt(
        outdir / "marginal_welfare_vs_v.csv",
        np.column_stack([v, metrics.marginal_welfare]),
        delimiter=",",
        header="v,marginal_welfare_dW_dv",
    )

    # Also save an NPZ for easy reload
    np.savez(
        outdir / "sweep_metrics.npz",
        v_values=metrics.v_values,
        flag_rate=metrics.flag_rate,
        expected_utility=metrics.expected_utility,
        gini_compute=metrics.gini_compute,
        total_welfare=metrics.total_welfare,
        compliance_gap=metrics.compliance_gap,
        cumulative_capability=metrics.cumulative_capability,
        marginal_welfare=metrics.marginal_welfare,
    )


def plot_sweep(metrics, outdir: Path, title_prefix: str = "") -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ModuleNotFoundError:
        _write_csv(outdir, metrics)
        print(
            "matplotlib not installed; wrote CSV/NPZ outputs instead. "
            "Install matplotlib to generate PNG plots (e.g., pip install matplotlib)."
        )
        return

    v = metrics.v_values
    N = metrics.flag_rate.shape[1]

    # 1) Detection flag rate vs v (per actor)
    plt.figure(figsize=(9, 5))
    for i in range(N):
        plt.plot(v, metrics.flag_rate[:, i], label=f"actor {i+1}")
    plt.xlabel("verification level v")
    plt.ylabel("flag rate  Pr(flag)  (mean over t, runs)")
    plt.title(f"{title_prefix}Detection flag rate vs verification".strip())
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(outdir / "flag_rate_vs_v.png", dpi=160)
    plt.close()

    # 2) Expected utility vs v (per actor)
    plt.figure(figsize=(9, 5))
    for i in range(N):
        plt.plot(v, metrics.expected_utility[:, i], label=f"actor {i+1}")
    plt.xlabel("verification level v")
    plt.ylabel("expected utility (mean per round)")
    plt.title(f"{title_prefix}Per-actor expected utility vs verification".strip())
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(outdir / "expected_utility_vs_v.png", dpi=160)
    plt.close()

    # 3) Gini of realized compute vs v
    plt.figure(figsize=(8, 5))
    plt.plot(v, metrics.gini_compute, color="black")
    plt.xlabel("verification level v")
    plt.ylabel("Gini(total realized compute per actor over horizon)")
    plt.title(f"{title_prefix}Compute inequality (Gini) vs verification".strip())
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "gini_compute_vs_v.png", dpi=160)
    plt.close()

    # 4) Total welfare vs v
    plt.figure(figsize=(8, 5))
    plt.plot(v, metrics.total_welfare, color="black")
    plt.xlabel("verification level v")
    plt.ylabel("total welfare (sum of utilities, mean per round)")
    plt.title(f"{title_prefix}Total welfare vs verification".strip())
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "total_welfare_vs_v.png", dpi=160)
    plt.close()

    # 5) Compliance gap vs v (per actor): hat_c_i - a_i (positive = under-contribution)
    plt.figure(figsize=(9, 5))
    for i in range(N):
        plt.plot(v, metrics.compliance_gap[:, i], label=f"actor {i+1}")
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.8, label="perfect compliance")
    plt.xlabel("verification level v")
    plt.ylabel("compliance gap  (pledged - actual)")
    plt.title(f"{title_prefix}Compliance gap vs verification".strip())
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(outdir / "compliance_gap_vs_v.png", dpi=160)
    plt.close()

    # 6) Cumulative capability q(T) vs v
    plt.figure(figsize=(8, 5))
    plt.plot(v, metrics.cumulative_capability, color="darkblue")
    plt.xlabel("verification level v")
    plt.ylabel("cumulative capability q(T) at final period")
    plt.title(f"{title_prefix}Cumulative capability vs verification".strip())
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "cumulative_capability_vs_v.png", dpi=160)
    plt.close()

    # 7) Marginal welfare gain dW/dv vs v
    plt.figure(figsize=(8, 5))
    plt.plot(v, metrics.marginal_welfare, color="darkgreen")
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.xlabel("verification level v")
    plt.ylabel("marginal welfare  dW/dv")
    plt.title(f"{title_prefix}Marginal welfare gain vs verification".strip())
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "marginal_welfare_vs_v.png", dpi=160)
    plt.close()


def plot_compliance_gap_vs_v_penalty_sweep(
    params,
    v_values: np.ndarray,
    penalty_scales: list[float],
    runs: int,
    seed: int,
    outdir: Path,
    title_prefix: str = "",
) -> None:
    """
    Plot compliance gap vs v for multiple penalty multipliers.
    Creates one subplot per penalty scale showing all actors.
    """
    from model import with_penalty_scale

    results_by_scale = []
    for s in penalty_scales:
        m = run_verification_sweep(with_penalty_scale(params, s), v_values=v_values, runs=runs, seed=seed)
        results_by_scale.append(m)

    N = params.N

    # Write CSV with aggregate (mean across actors) compliance gap
    header = "v," + ",".join([f"penalty_x{s:g}" for s in penalty_scales])
    agg_gap = np.column_stack([m.compliance_gap.mean(axis=1) for m in results_by_scale])
    np.savetxt(
        outdir / "compliance_gap_vs_v_penalty_sweep.csv",
        np.column_stack([v_values, agg_gap]),
        delimiter=",",
        header=header,
    )

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return

    n_scales = len(penalty_scales)
    fig, axes = plt.subplots(1, n_scales, figsize=(6 * n_scales, 5), sharey=True)
    if n_scales == 1:
        axes = [axes]

    for idx, (s, m) in enumerate(zip(penalty_scales, results_by_scale)):
        ax = axes[idx]
        for i in range(N):
            ax.plot(v_values, m.compliance_gap[:, i], label=f"actor {i+1}")
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("verification level v")
        if idx == 0:
            ax.set_ylabel("compliance gap  (pledged - actual)")
        ax.set_title(f"Penalty x{s:g}")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2, fontsize=8)

    fig.suptitle(f"{title_prefix}Compliance gap vs verification (penalty comparison)".strip(), fontsize=12)
    plt.tight_layout()
    plt.savefig(outdir / "compliance_gap_vs_v_penalty_sweep.png", dpi=160)
    plt.close()


def plot_welfare_vs_v_penalty_sweep(
    params,
    v_values: np.ndarray,
    penalty_scales: list[float],
    runs: int,
    seed: int,
    outdir: Path,
    title_prefix: str = "",
) -> None:
    """
    Plot (and/or write CSV for) total welfare vs v for multiple penalty multipliers.
    """
    welfare_by_scale = []
    for s in penalty_scales:
        m = run_verification_sweep(with_penalty_scale(params, s), v_values=v_values, runs=runs, seed=seed)
        welfare_by_scale.append(m.total_welfare)

    welfare_by_scale = np.asarray(welfare_by_scale)  # (S, V)

    # Always write a CSV for this plot.
    header = "v," + ",".join([f"penalty_x{s:g}" for s in penalty_scales])
    np.savetxt(
        outdir / "total_welfare_vs_v_penalty_sweep.csv",
        np.column_stack([v_values, welfare_by_scale.T]),
        delimiter=",",
        header=header,
    )

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ModuleNotFoundError:
        return

    plt.figure(figsize=(9, 5))
    for idx, s in enumerate(penalty_scales):
        plt.plot(v_values, welfare_by_scale[idx], label=f"penalty x{s:g}")
    plt.xlabel("verification level v")
    plt.ylabel("total welfare (sum of utilities, mean per round)")
    plt.title(f"{title_prefix}Total welfare vs verification (penalty sweep)".strip())
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(outdir / "total_welfare_vs_v_penalty_sweep.png", dpi=160)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", type=str, default="parameters.json")
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument(
        "--runs",
        type=int,
        default=0,
        help="Monte Carlo runs. Use 0 for fast expected-value sweep (recommended).",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vmin", type=float, default=0.0)
    ap.add_argument("--vmax", type=float, default=1.0)
    ap.add_argument("--vsteps", type=int, default=21)
    ap.add_argument(
        "--penalty-scales",
        type=str,
        default="",
        help="Comma-separated multipliers for P_i, e.g. '0.5,1,2'. If provided, writes an extra welfare-vs-v plot/CSV.",
    )
    ap.add_argument("--title", type=str, default="")
    ap.add_argument(
        "--cap-at-pledge",
        action="store_true",
        help="Constrain actors to never exceed their pledged compute (a_i <= hat_c_i).",
    )
    args = ap.parse_args()

    params = load_params(args.params)
    if args.cap_at_pledge:
        params = with_cap_at_pledge(params, True)
    outdir = Path(args.outdir)
    _ensure_dir(outdir)

    v_values = np.linspace(args.vmin, args.vmax, args.vsteps)
    metrics = run_verification_sweep(params, v_values=v_values, runs=args.runs, seed=args.seed)
    plot_sweep(metrics, outdir=outdir, title_prefix=(args.title + " " if args.title else ""))

    if args.penalty_scales.strip():
        penalty_scales = [float(x) for x in args.penalty_scales.split(",") if x.strip()]
        plot_welfare_vs_v_penalty_sweep(
            params=params,
            v_values=v_values,
            penalty_scales=penalty_scales,
            runs=args.runs,
            seed=args.seed,
            outdir=outdir,
            title_prefix=(args.title + " " if args.title else ""),
        )
        plot_compliance_gap_vs_v_penalty_sweep(
            params=params,
            v_values=v_values,
            penalty_scales=penalty_scales,
            runs=args.runs,
            seed=args.seed,
            outdir=outdir,
            title_prefix=(args.title + " " if args.title else ""),
        )

    # Print a tiny summary to stdout (useful when running headless).
    best_idx = int(np.argmax(metrics.total_welfare))
    print(
        f"done: wrote plots to {outdir.resolve()} | "
        f"best welfare at v={metrics.v_values[best_idx]:.3f} (welfare={metrics.total_welfare[best_idx]:.6g})"
    )


if __name__ == "__main__":
    main()

