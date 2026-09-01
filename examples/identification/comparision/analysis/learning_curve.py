"""
How many stimulation trains the identification actually needs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import _campaign
import _figures
import _shared

import matplotlib.pyplot as plt

import protocol

ANALYSIS = "learning_curve"

# How many of the ten non-reference identification trains to keep. The reference column adds three more,
# so the x axis of the curve runs from 5 to 13 identification trains.
DEFAULT_SIZES = (2, 4, 6, 8, 10)
DEFAULT_REPEATS = 3
REFERENCE_TRAINS = 3


def run_subject(route, subject: str, sizes, repeats: int, seed: int) -> pd.DataFrame:
    """Every size and every draw of one participant, as one row each."""
    available = _campaign.cells(protocol.IDENTIFICATION)
    generator = np.random.default_rng(seed)
    rows = []

    for size in sizes:
        draws = 1 if size >= len(available) else repeats
        for draw in range(draws):
            kept = generator.choice(len(available), size=min(size, len(available)), replace=False)
            dropped = [cell for i, cell in enumerate(available) if i not in set(kept.tolist())]
            design = _campaign.design_with(to_dropped=dropped)
            print(f"  P{subject} size {size} draw {draw + 1}/{draws} "
                  f"[{_campaign.describe(design)}]", flush=True)

            outcome = _campaign.identify(route, subject, design=design)
            rows.append(_campaign.flat(outcome, route=route.METHOD, size=size, draw=draw,
                                       n_identification=size + REFERENCE_TRAINS,
                                       kept="|".join(protocol.condition_name(f, w * 1e-6)
                                                     for f, w in
                                                     [available[i] for i in sorted(kept.tolist())])))
    return pd.DataFrame(rows)


def curve(detail: pd.DataFrame) -> pd.DataFrame:
    """The curve itself: the two errors against the number of identification trains, pooled per route."""
    usable = detail[detail["failed"] == ""]
    if usable.empty:
        return usable
    grouped = usable.groupby(["route", "n_identification"], sort=True)
    table = grouped.agg(
        n_runs=("subject", "count"),
        n_subjects=("subject", "nunique"),
        testing_peak_error_pct=("testing_peak_error_pct", "mean"),
        testing_peak_error_sd=("testing_peak_error_pct", "std"),
        testing_r2=("testing_r2", "mean"),
        identification_peak_error_pct=("identification_peak_error_pct", "mean"),
        identification_peak_error_sd=("identification_peak_error_pct", "std"),
        converged=("converged", "sum"),
    ).reset_index()
    return table


def plateau(table: pd.DataFrame, tolerance_pct: float = 1.0) -> pd.DataFrame:
    """The smallest training set whose testing error is within `tolerance_pct` of the best on the curve."""
    rows = []
    for route, group in table.groupby("route", sort=False):
        group = group.sort_values("n_identification")
        best = float(group["testing_peak_error_pct"].min())
        enough = group[group["testing_peak_error_pct"] <= best + tolerance_pct]
        rows.append({"route": route,
                     "best_testing_peak_error_pct": best,
                     "trains_needed": int(enough["n_identification"].min()) if not enough.empty else np.nan,
                     "trains_recorded": int(group["n_identification"].max()),
                     "error_at_full_set": float(group["testing_peak_error_pct"].iloc[-1])})
    return pd.DataFrame(rows)


def figure(table: pd.DataFrame, reached=None, tolerance=1.0, name="learning_curve") -> None:
    """
    What the identification gains from more trains, route by route.

    The x axis starts where the experiment starts: the three reference trains are always there, so a
    curve drawn from zero would describe a study that never happened.
    """
    if table.empty:
        return
    routes = _figures.ordered(list(dict.fromkeys(table["route"])))
    reached = pd.DataFrame() if reached is None else reached
    fig, axes = plt.subplots(1, len(routes), figsize=(4.8 * len(routes), 4.8), squeeze=False,
                             sharey=True)

    for axis, route in zip(axes[0], routes):
        group = table[table["route"] == route].sort_values("n_identification")
        for column, spread, label, tint in (
                ("testing_peak_error_pct", "testing_peak_error_sd", "testing (held out)",
                 _figures.colour(route)),
                ("identification_peak_error_pct", "identification_peak_error_sd", "identification",
                 "0.45")):
            axis.plot(group["n_identification"], group[column], "-o", ms=4, color=tint, label=label)
            axis.fill_between(group["n_identification"], group[column] - group[spread].fillna(0),
                              group[column] + group[spread].fillna(0), color=tint, alpha=0.15)

        block = reached[reached["route"] == route] if not reached.empty else pd.DataFrame()
        if not block.empty:
            entry = block.iloc[0]
            axis.axhspan(entry["best_testing_peak_error_pct"],
                         entry["best_testing_peak_error_pct"] + tolerance, color="0.6", alpha=0.20)
            if np.isfinite(entry["trains_needed"]):
                axis.axvline(entry["trains_needed"], color=_figures.THRESHOLD, ls="--", lw=1.0)
                _figures.note(axis, f"{int(entry['trains_needed'])} identification trains are enough, "
                                    f"within {tolerance:.0f} error point of the best on the curve "
                                    f"(--tolerance)\n"
                                    f"error on the full set {entry['error_at_full_set']:.1f} %", size=6)
        axis.set_xlim(group["n_identification"].min() - 0.4, group["n_identification"].max() + 0.4)
        axis.set_title(_figures.label(route), fontsize=10, fontweight="bold")
        axis.set_xlabel("identification trains", fontsize=8)
        axis.set_ylabel("peak %model error", fontsize=8)
        axis.legend(fontsize=6, loc="lower left")
        axis.tick_params(labelsize=7)
        axis.grid(alpha=0.3)
        _figures.count(axis, f"{int(group['n_subjects'].max())} participants, "
                             f"{int(group['n_runs'].sum())} runs")

    fig.suptitle("How many stimulation trains the protocol really needs", fontsize=13,
                 fontweight="bold")
    _figures.caption(fig, f"The axis starts at {int(table['n_identification'].min())}, not at 0: the "
                          f"{REFERENCE_TRAINS} reference trains are in every point of the curve.\n"
                          f"Only equation 3 moves along this axis - a_scale, pd0 and pdt. A, Km, tau1 and "
                          f"tau2 are settled on the reference column alone and their gap along the curve "
                          f"is exactly zero.")
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    _shared.save_figure(fig, ANALYSIS, name)


def main():
    _shared.headless()
    parser = _shared.arguments(__doc__.splitlines()[1], routes=("movement_id",))
    parser.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES),
                        help="how many of the ten non-reference identification trains to keep")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS, help="draws per size")
    parser.add_argument("--tolerance", type=float, default=1.0,
                        help="error points above the best that still count as the plateau")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--refresh", action="store_true", help="re-run participants already on disk")
    arguments = parser.parse_args()

    print(f"learning curve over sizes {arguments.sizes}, {arguments.repeats} draws each")
    detail = _campaign.per_subject(
        ANALYSIS, arguments,
        lambda route, subject: run_subject(route, subject, arguments.sizes, arguments.repeats,
                                           arguments.seed),
    )
    if detail.empty:
        return
    table = curve(detail)
    _shared.banner("learning curve")
    print(table.round(2).to_string(index=False))
    _shared.banner("how many trains the protocol needs")
    print(plateau(table, arguments.tolerance).round(2).to_string(index=False))

    _shared.write(detail, ANALYSIS, "all_runs")
    _shared.write(table, ANALYSIS, "curve")
    _shared.write(plateau(table, arguments.tolerance), ANALYSIS, "plateau")
    figure(table, plateau(table, arguments.tolerance), arguments.tolerance)


if __name__ == "__main__":
    main()
