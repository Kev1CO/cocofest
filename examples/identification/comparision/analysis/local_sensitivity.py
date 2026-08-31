"""
How much the model's output moves when one parameter is nudged, around each participant's own optimum.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import _campaign
import _shared
import _simulator

import matplotlib.pyplot as plt

ANALYSIS = "local_sensitivity"

# Multiplicative nudges applied one parameter at a time, symmetric in log so +20% and -20% weigh the same
DEFAULT_FACTORS = (0.8, 0.9, 0.95, 1.05, 1.1, 1.2)
DEFAULT_CONDITIONS = ((20, 400), (33, 400), (50, 400), (20, 250), (20, 500))


def sweep(route, subject: str, parameters: dict, factors, conditions) -> pd.DataFrame:
    """One participant's sweep: every parameter, every factor, every condition."""
    rows = []
    for frequency, pulse_width in conditions:
        reference = _simulator.outcome_of(route, subject, parameters, frequency, pulse_width)
        for name in _simulator.FREE:
            if name not in parameters:
                continue
            for factor in factors:
                moved = dict(parameters)
                moved[name] = float(parameters[name]) * factor
                produced = _simulator.outcome_of(route, subject, moved, frequency, pulse_width)
                row = {"parameter": name, "factor": factor,
                       "frequency": frequency, "pulse_width_us": pulse_width}
                for scalar in ("peak", "fti"):
                    base = reference[scalar]
                    row[f"{scalar}_reference"] = base
                    row[f"{scalar}_value"] = produced[scalar]
                    change = (produced[scalar] - base) / base if base else np.nan
                    # Elasticity: the % the output moves per % the parameter moves
                    row[f"{scalar}_elasticity"] = change / (factor - 1) if factor != 1 else np.nan
                rows.append(row)
    return pd.DataFrame(rows)


def run_subject(route, subject: str, factors, conditions) -> pd.DataFrame:
    """One participant of one route, sweeping around the values that route identified."""
    from helper import results

    stored = results.load(subject, route.METHOD)
    detail = sweep(route, subject, stored["parameters"], factors, conditions)
    detail.insert(0, "subject", subject)
    detail.insert(0, "route", route.METHOD)
    return detail


def summarise(detail: pd.DataFrame) -> pd.DataFrame:
    """One row per route and parameter: the elasticity, averaged over factors, conditions and participants."""
    if detail.empty:
        return detail
    grouped = detail.groupby(["route", "parameter"], sort=False)
    table = grouped.agg(
        n=("subject", "count"),
        peak_elasticity=("peak_elasticity", lambda values: float(np.nanmean(np.abs(values)))),
        peak_elasticity_sd=("peak_elasticity", lambda values: float(np.nanstd(np.abs(values), ddof=1))),
        fti_elasticity=("fti_elasticity", lambda values: float(np.nanmean(np.abs(values)))),
        signed_peak_elasticity=("peak_elasticity", lambda values: float(np.nanmean(values))),
    ).reset_index()
    return table.sort_values(["route", "peak_elasticity"], ascending=[True, False])


def figure(summary: pd.DataFrame, name="local_sensitivity") -> None:
    """The elasticity of every parameter, route by route, on a log axis because they span decades."""
    if summary.empty:
        return
    routes = list(dict.fromkeys(summary["route"]))
    fig, axis = plt.subplots(figsize=(1.6 * len(routes) + 4, 4.2))

    width = 0.8 / max(len(routes), 1)
    positions = np.arange(len(_simulator.FREE))
    for index, route in enumerate(routes):
        group = summary[summary["route"] == route].set_index("parameter")
        heights = [group["peak_elasticity"].get(name, np.nan) for name in _simulator.FREE]
        axis.bar(positions + index * width, heights, width, label=route.replace("_id", ""))

    axis.set_xticks(positions + 0.4 - width / 2)
    axis.set_xticklabels(_simulator.FREE, rotation=45, ha="right", fontsize=7)
    axis.set_yscale("log")
    axis.set_ylabel("|elasticity| of the peak force", fontsize=8)
    axis.axhline(0.05, color="C3", lw=0.9, ls="--")
    axis.text(0.01, 0.95, "below the dashed line: a 10% error in the parameter moves the peak by under 0.5%",
              transform=axis.transAxes, fontsize=6, va="top")
    axis.legend(fontsize=6)
    axis.tick_params(axis="y", labelsize=7)

    fig.tight_layout()
    _shared.save_figure(fig, ANALYSIS, name)


def main():
    _shared.headless()
    parser = _shared.arguments(__doc__.splitlines()[1], routes=_shared.ISOMETRIC_ROUTES)
    parser.add_argument("--factors", type=float, nargs="+", default=list(DEFAULT_FACTORS),
                        help="multiplicative nudges applied one parameter at a time")
    parser.add_argument("--refresh", action="store_true", help="re-run participants already on disk")
    arguments = parser.parse_args()

    detail = _campaign.per_subject(
        ANALYSIS, arguments,
        lambda route, subject: run_subject(route, subject, arguments.factors, DEFAULT_CONDITIONS),
    )
    if detail.empty:
        return
    summary = summarise(detail)
    _shared.banner("local sensitivity around each participant's optimum")
    print(summary.round(4).to_string(index=False))
    _shared.write(detail, ANALYSIS, "all_sweeps")
    _shared.write(summary, ANALYSIS, "summary")
    figure(summary)


if __name__ == "__main__":
    main()
