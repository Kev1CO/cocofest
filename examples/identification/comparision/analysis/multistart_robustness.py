"""
Whether the identification finds the same answer from anywhere in the box, or only from where it starts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import _campaign
import _shared

import matplotlib.pyplot as plt

import steps

ANALYSIS = "multistart_robustness"

PERTURBED = ("A", "km_rest", "tau2", "tau1_rest")
DEFAULT_STARTS = 20
DEFAULT_TOLERANCE = 0.02


def ranges(subject: str, route: str) -> dict:
    """The box a start is drawn from, with A's ceiling read off what the route already identified."""
    box = {name: (steps.IDENTIFIED_PARAMETERS[name][1], steps.IDENTIFIED_PARAMETERS[name][2])
           for name in PERTURBED if steps.IDENTIFIED_PARAMETERS[name][2] is not None}
    try:
        stored = _shared.cohort(route)
        row = stored[stored["subject"] == subject].iloc[0]
        # parameter_table gives A and a_scale the same ceiling, so a_scale's upper bound is A's
        box["A"] = (steps.IDENTIFIED_PARAMETERS["A"][1], float(row["a_scale__max"]))
    except (IndexError, KeyError):
        pass
    return box


def draw(box: dict, starts: int, seed: int) -> list:
    """`starts` initial guesses, log-uniform in the box, the first of which is the procedure's default."""
    generator = np.random.default_rng(seed)
    samples = [{}]
    for _ in range(starts - 1):
        samples.append({name: float(np.exp(generator.uniform(np.log(low), np.log(high))))
                        for name, (low, high) in box.items()})
    return samples


def run_subject(route, subject: str, starts: int, seed: int, scope: str) -> pd.DataFrame:
    """Every start of one participant, as one row each."""
    box = ranges(subject, route.METHOD)
    rows = []
    for index, guess in enumerate(draw(box, starts, seed)):
        print(f"  P{subject} start {index + 1}/{starts}"
              + ("" if not guess else " " + " ".join(f"{k}={v:.4g}" for k, v in guess.items())), flush=True)
        outcome = _campaign.identify(route, subject, initial_guesses=guess or None,
                                     **({"max_iter": 5000} if scope == "full" else {}))
        rows.append(_campaign.flat(outcome, route=route.METHOD, start=index,
                                   is_default=index == 0,
                                   **{f"guess_{name}": value for name, value in guess.items()}))
    return pd.DataFrame(rows)


def summarise(detail: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    """One row per participant: how many basins, how wide the best one, and what the default missed."""
    rows = []
    for (route, subject), group in detail.groupby(["route", "subject"], sort=False):
        converged = group[group["converged"] & (group["failed"] == "")]
        if converged.empty:
            rows.append({"route": route, "subject": subject, "n_starts": len(group), "n_converged": 0})
            continue

        best = float(converged["step2_cost"].min())
        # A start is "in the best basin" when its cost is within a relative tolerance of the best found
        in_basin = converged[converged["step2_cost"] <= best * (1 + tolerance) + 1e-12]
        default = group[group["is_default"]]
        default_cost = float(default["step2_cost"].iloc[0]) if len(default) else np.nan

        row = {"route": route, "subject": subject,
               "n_starts": int(len(group)), "n_converged": int(len(converged)),
               "best_cost": best, "default_cost": default_cost,
               "best_vs_default_pct": float(100 * (default_cost - best) / default_cost)
               if np.isfinite(default_cost) and default_cost else np.nan,
               "basin_share": float(len(in_basin) / len(converged)),
               "cost_spread_pct": float(100 * (converged["step2_cost"].max() - best) / best) if best else np.nan}

        for name in _shared.PARAMETERS:
            if name not in in_basin:
                continue
            values = in_basin[name].astype(float)
            row[f"{name}_median"] = float(values.median())
            row[f"{name}_cv_pct"] = float(100 * values.std() / values.mean()) if values.mean() else np.nan
            row[f"{name}_min"], row[f"{name}_max"] = float(values.min()), float(values.max())
        rows.append(row)
    return pd.DataFrame(rows)


def figure(detail: pd.DataFrame, name="multistart_robustness") -> None:
    """The cost of every start against the best found, and the spread each parameter kept."""
    converged = detail[detail["converged"] & (detail["failed"] == "")]
    if converged.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    axis = axes[0]
    for subject, group in converged.groupby("subject"):
        best = group["step2_cost"].min()
        axis.scatter([subject] * len(group), 100 * (group["step2_cost"] - best) / best,
                     s=10, alpha=0.6, edgecolor="none")
    axis.set_yscale("symlog", linthresh=1.0)
    axis.set_xlabel("participant", fontsize=8)
    axis.set_ylabel("step 2 cost above the best start (%)", fontsize=8)
    axis.tick_params(axis="x", labelrotation=90, labelsize=6)
    axis.tick_params(axis="y", labelsize=7)

    axis = axes[1]
    present = [name_of for name_of in _shared.PARAMETERS if name_of in converged]
    _shared.boxplot(axis, [converged.groupby("subject")[name_of].apply(
        lambda values: 100 * values.std() / values.mean() if values.mean() else np.nan).dropna()
        for name_of in present], present, showfliers=False)
    axis.set_ylabel("between-start CV of the identified value (%)", fontsize=8)
    axis.tick_params(axis="x", labelrotation=90, labelsize=7)
    axis.tick_params(axis="y", labelsize=7)

    fig.tight_layout()
    _shared.save_figure(fig, ANALYSIS, name)


def main():
    _shared.headless()
    parser = _shared.arguments(__doc__.splitlines()[1], routes=("sensor_id",))
    parser.add_argument("--starts", type=int, default=DEFAULT_STARTS, help="initial guesses per participant")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE,
                        help="relative cost gap that still counts as the same basin")
    parser.add_argument("--scope", choices=("step2", "full"), default="step2",
                        help="step2 perturbs the start of step 2 only, full re-runs every step")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--refresh", action="store_true", help="re-run participants already on disk")
    arguments = parser.parse_args()

    print(f"{arguments.starts} starts per participant, scope {arguments.scope}")
    detail = _campaign.per_subject(
        ANALYSIS, arguments,
        lambda route, subject: run_subject(route, subject, arguments.starts, arguments.seed,
                                           arguments.scope),
    )
    if detail.empty:
        return
    _shared.banner("multistart summary")
    summary = summarise(detail, arguments.tolerance)
    print(summary.round(4).to_string(index=False))
    _shared.write(detail, ANALYSIS, "all_starts")
    _shared.write(summary, ANALYSIS, "summary")
    figure(detail)


if __name__ == "__main__":
    main()
