"""
How much the model's output moves when one parameter is nudged, around each participant's own optimum.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import _campaign
import _figures
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


VISIBLE = 0.05  # below this a 10 % error in the parameter moves the peak by under 0.5 %


def draw_local(axis, summary: pd.DataFrame, factors=DEFAULT_FACTORS,
               conditions=DEFAULT_CONDITIONS) -> None:
    """
    The amplitude of the elasticity as the bar, its sign as the hatch.

    Both columns are drawn because they say two different things: the amplitude whether the measurement
    sees the parameter at all, the sign which way it pushes.
    """
    routes = _figures.ordered(list(dict.fromkeys(summary["route"])))
    width = 0.8 / max(len(routes), 1)
    positions = np.arange(len(_simulator.FREE))

    for index, route in enumerate(routes):
        group = summary[summary["route"] == route].set_index("parameter")
        heights = [abs(group["peak_elasticity"].get(name, np.nan)) for name in _simulator.FREE]
        signs = [group.get("signed_peak_elasticity", pd.Series(dtype=float)).get(name, np.nan)
                 for name in _simulator.FREE]
        errors = [group.get("peak_elasticity_sd", pd.Series(dtype=float)).get(name, np.nan)
                  for name in _simulator.FREE]
        bars = axis.bar(positions + index * width, heights, width, yerr=errors, capsize=2,
                        color=_figures.colour(route), alpha=0.75, label=_figures.label(route),
                        error_kw={"lw": 0.6})
        for bar, sign in zip(bars, signs):
            if np.isfinite(sign) and sign < 0:
                bar.set_hatch("xx")
                bar.set_edgecolor("white")

    axis.set_xticks(positions + 0.4 - width / 2)
    axis.set_xticklabels(_simulator.FREE, rotation=45, ha="right", fontsize=7)
    axis.set_yscale("log")
    axis.set_ylabel("|elasticity| of the peak force", fontsize=8)
    axis.axhline(VISIBLE, color=_figures.BOUND, lw=0.9, ls="--")
    axis.legend(fontsize=6)
    axis.tick_params(axis="y", labelsize=7)
    _figures.note(axis, f"bar: the mean of the absolute elasticity. Cross hatch: the signed mean is "
                        f"negative,\nso the parameter pushes the peak down.\n"
                        f"red dashed: {VISIBLE}, under which a 10 % error in the parameter moves the peak "
                        f"by under 0.5 %\n"
                        f"factors {', '.join(str(factor) for factor in factors)} around each "
                        f"participant's own values\n"
                        f"conditions " + ", ".join(f"({frequency} Hz, {pulse_width} us)"
                                                   for frequency, pulse_width in conditions), size=5.5)


def figure(summary: pd.DataFrame, name="local_sensitivity") -> None:
    """The elasticity of every parameter, route by route, on a log axis because they span decades."""
    if summary.empty:
        return
    routes = list(dict.fromkeys(summary["route"]))
    fig, axis = plt.subplots(figsize=(1.6 * len(routes) + 5, 4.6))
    draw_local(axis, summary)
    fig.suptitle("What the peak force feels when one parameter is nudged", fontsize=11,
                 fontweight="bold")
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
