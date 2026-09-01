"""
Whether the identification finds the same answer from anywhere in the box, or only from where it starts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import _campaign
import _figures
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
        outcome = _campaign.identify(route, subject, initial_guesses=guess or None)
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


def figure(detail: pd.DataFrame, summary: pd.DataFrame, tolerance=DEFAULT_TOLERANCE,
           name="multistart_robustness") -> None:
    """
    Whether the answer is the optimum or the one the default start happens to reach.

    The participants are ordered by how wide their best basin is, because that order is information: the
    ones on the left are the ones a single start is least likely to land in.
    """
    usable = detail[detail["failed"] == ""]
    if usable.empty:
        return
    routes = _figures.ordered(list(dict.fromkeys(usable["route"])))

    fig, axes = plt.subplots(len(routes), 2, figsize=(15, 4.4 * len(routes)), squeeze=False,
                             gridspec_kw={"width_ratios": [2.2, 1]})
    for row, route in enumerate(routes):
        block = usable[usable["route"] == route]
        shares = summary[summary["route"] == route].set_index("subject") if not summary.empty \
            else pd.DataFrame()
        order = (shares["basin_share"].sort_values().index.tolist()
                 if "basin_share" in shares else sorted(block["subject"].unique()))

        axis = axes[row][0]
        for position, subject in enumerate(order):
            group = block[block["subject"] == subject]
            converged = group[group["converged"]]
            if converged.empty:
                continue
            best = float(converged["step2_cost"].min())
            above = 100 * (group["step2_cost"] - best) / best
            axis.scatter(position + _figures.jitter(len(group), 0.16), above, s=12, alpha=0.6,
                         edgecolor="none", color=_figures.colour(route))
            default = group[group["is_default"]]
            if not default.empty:
                axis.scatter([position], 100 * (default["step2_cost"].iloc[0] - best) / best, s=44,
                             marker="*", color="black", zorder=4)
        axis.set_yscale("symlog", linthresh=1.0)
        axis.axhline(100 * tolerance, color=_figures.THRESHOLD, ls="--", lw=0.9)
        axis.set_xticks(range(len(order)))
        axis.set_xticklabels([f"P{subject}" for subject in order], fontsize=6, rotation=90)
        axis.set_xlabel("participant, ordered by how wide the best basin is", fontsize=8)
        axis.set_ylabel("step 2 cost above the best start (%)", fontsize=8)
        axis.set_title(_figures.label(route), fontsize=10, fontweight="bold", loc="left")
        axis.tick_params(axis="y", labelsize=7)
        axis.grid(alpha=0.3, axis="y")
        told = ""
        if "basin_share" in shares:
            told = (f"median basin share {shares['basin_share'].median():.2f}: that share of the "
                    f"converged starts reach the best cost\n"
                    f"median cost the default start missed by "
                    f"{shares['best_vs_default_pct'].median():.2f} %\n")
        _figures.note(axis, told + f"black star: the start the procedure itself uses\n"
                                   f"dashed: the {100 * tolerance:.0f} % tolerance that defines the basin, "
                                   f"and the starts panel (b) reads", size=6)

        axis = axes[row][1]
        present = [name_of for name_of in _shared.PARAMETERS if name_of in block]
        spreads = []
        for name_of in present:
            spreads.append(block.groupby("subject")[name_of].apply(
                lambda values: 100 * values.std() / values.mean() if values.mean() else np.nan).dropna())
        if spreads:
            axis.boxplot(spreads, positions=range(len(present)), widths=0.55, showfliers=False,
                         medianprops=dict(color="black", linewidth=1.2))
            for position, values in enumerate(spreads):
                _figures.points(axis, position, values.to_numpy(float), route, size=12)
        axis.set_xticks(range(len(present)))
        axis.set_xticklabels(present, rotation=45, ha="right", fontsize=7)
        axis.set_ylabel("between-start CV of the identified value (%)", fontsize=8)
        axis.tick_params(axis="y", labelsize=7)
        axis.grid(alpha=0.3, axis="y")
        _figures.count(axis, f"{block['subject'].nunique()} participants")

    fig.suptitle("Is the optimum reachable from anywhere, or only from where the procedure starts",
                 fontsize=13, fontweight="bold")
    _figures.caption(fig, f"(b) is computed on the starts of the best basin only, the ones within "
                          f"{100 * tolerance:.0f} % of the best cost found.")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
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
    figure(detail, summary, arguments.tolerance)


if __name__ == "__main__":
    main()
