"""
Practical identifiability, by profiling the cost along one parameter at a time, Raue et al. (2009).

One parameter is held at each value of a grid while every other free parameter of step 2 is re-optimised,
and the shape of the resulting profile is the answer: a minimum with the cost rising on both sides is
identifiable and its width at the threshold is the confidence interval, flat on one side is practically
non-identifiable in that direction, flat on both is not identifiable at all.

The threshold is the likelihood-ratio one, where the cost rises by (1 + chi2(0.95, 1) / (n - p)) above the
optimum, with n the tracked nodes of the reference phases. Runs step 2 only, once per grid point.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import _campaign
import _figures
import _shared

import matplotlib.pyplot as plt

import ding_model
import steps

ANALYSIS = "profile_likelihood"

DEFAULT_POINTS = 9
DEFAULT_SPAN = 4.0  # the grid runs from value/span to value*span, clipped to the parameter's bounds
CONFIDENCE = 0.95


def grid(value: float, low: float, high: float, points: int, span: float) -> np.ndarray:
    """A log-spaced grid around the identified value, inside the parameter's own bounds."""
    start, stop = max(low, value / span), min(high, value * span)
    if not (stop > start > 0):
        return np.array([value])
    return np.unique(np.concatenate([np.geomspace(start, stop, points), [value]]))


def threshold(best_cost: float, n_data: int, n_free: int) -> float:
    """The cost above which a parameter value is rejected, at 95% for one parameter at a time."""
    degrees = max(n_data - n_free, 1)
    return float(best_cost * (1.0 + stats.chi2.ppf(CONFIDENCE, 1) / degrees))


def profile_one(route, subject: str, name: str, points: int, span: float, max_iter: int) -> pd.DataFrame:
    """One parameter of one participant, profiled: hold it, re-optimise the rest, record the cost."""
    context = steps.step2_context(route, subject, debug=False)
    if name not in context["step2_free"]:
        print(f"  {name} is not free at step 2 of {route.METHOD}, skipped")
        return pd.DataFrame()
    free = tuple(n for n in context["step2_free"] if n != name)

    # The unconstrained step 2, which the profile is measured against
    best = steps.solve(route, subject, context["reference_phases"], context["step2_free"],
                       ding_model.DingModelSingleIntensity, context["table"],
                       constants=context["constants"], initial_guesses=context["guesses"],
                       max_iter=max_iter)
    low, high = context["table"][name][1], context["table"][name][2]
    values = grid(float(best["parameters"][name]), low, high, points, span)
    n_data = int(sum(len(np.asarray(phase["target"])) for phase in context["reference_phases"]))

    rows = []
    for value in values:
        held = {**context["constants"], name: float(value)}
        step = steps.solve(route, subject, context["reference_phases"], free,
                           ding_model.DingModelSingleIntensity, context["table"],
                           constants=held, initial_guesses=context["guesses"], max_iter=max_iter)
        print(f"  P{subject} {name}={value:.5g} -> cost {step['cost']:.6g} "
              f"(converged {step['converged']})", flush=True)
        rows.append({"route": route.METHOD, "subject": subject, "parameter": name,
                     "value": float(value), "cost": float(step["cost"]),
                     "converged": bool(step["converged"]), "status": step["status"],
                     "iterations": int(step["iterations"]),
                     "seconds": float(step["seconds"]),
                     "best_cost": float(best["cost"]), "best_value": float(best["parameters"][name]),
                     "bound_low": float(low), "bound_high": float(high),
                     "n_data": n_data, "n_free": len(context["step2_free"]),
                     **{f"refit_{other}": float(step["parameters"].get(other, np.nan)) for other in free}})
    return pd.DataFrame(rows)


def profile_all(route, subject: str, names, points: int, span: float, max_iter: int) -> pd.DataFrame:
    """Every asked parameter of one participant, a profile that raises reported and skipped."""
    pieces = []
    for name in names:
        try:
            pieces.append(profile_one(route, subject, name, points, span, max_iter))
        except Exception as error:
            print(f"  {name}: {type(error).__name__}: {error}", flush=True)
    usable = [piece for piece in pieces if not piece.empty]
    return pd.concat(usable, ignore_index=True) if usable else pd.DataFrame()


def read_profile(group: pd.DataFrame) -> dict:
    """The verdict of one profile: the interval it supports, and whether it is open on either side."""
    usable = group[group["converged"]].sort_values("value")
    if usable.empty:
        return {}
    best_cost = float(usable["cost"].min())
    limit = threshold(best_cost, int(usable["n_data"].iloc[0]), int(usable["n_free"].iloc[0]))
    inside = usable[usable["cost"] <= limit]
    if inside.empty:
        return {}

    low, high = float(inside["value"].min()), float(inside["value"].max())
    open_low = bool(np.isclose(low, usable["value"].min()))
    open_high = bool(np.isclose(high, usable["value"].max()))
    return {
        "best_value": float(usable.loc[usable["cost"].idxmin(), "value"]),
        "best_cost": best_cost, "threshold": limit,
        "ci_low": low, "ci_high": high,
        "ci_width_decades": float(np.log10(high / low)) if low > 0 else np.nan,
        "open_low": open_low, "open_high": open_high,
        "cost_rise_pct": float(100 * (usable["cost"].max() - best_cost) / best_cost) if best_cost else np.nan,
        "verdict": ("not identifiable" if open_low and open_high else
                    "one-sided" if open_low or open_high else "identifiable"),
    }


def summarise(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (route, subject, name), group in detail.groupby(["route", "subject", "parameter"], sort=False):
        block = read_profile(group)
        if block:
            rows.append({"route": route, "subject": subject, "parameter": name, **block})
    return pd.DataFrame(rows)


VERDICT_COLOURS = {"identifiable": "tab:green", "one-sided": "tab:orange",
                   "not identifiable": "tab:purple"}


def figure(detail: pd.DataFrame, summary=None, name="profile_likelihood") -> None:
    """
    One panel per parameter and per route, every participant's profile normalised to its own optimum.

    The x axis is logarithmic because the intervals span orders of magnitude, which is exactly why the
    summary writes their width in decades.
    """
    if detail.empty:
        return
    summary = summarise(detail) if summary is None else summary
    routes = _figures.ordered(list(dict.fromkeys(detail["route"])))
    parameters = [name_of for name_of in _shared.PARAMETERS if name_of in set(detail["parameter"])]
    if not parameters:
        return

    fig, axes = plt.subplots(len(routes), len(parameters),
                             figsize=(3.4 * len(parameters), 3.4 * len(routes)), squeeze=False)
    for row, route in enumerate(routes):
        for column, parameter in enumerate(parameters):
            axis = axes[row][column]
            group = detail[(detail["route"] == route) & (detail["parameter"] == parameter)]
            if group.empty:
                axis.axis("off")
                continue
            verdicts = (summary[(summary["route"] == route) & (summary["parameter"] == parameter)]
                        .set_index("subject")["verdict"].to_dict() if not summary.empty else {})

            middle = sorted(group["subject"].unique())[len(group["subject"].unique()) // 2]
            thresholds = []
            for subject, curve in group.groupby("subject"):
                curve = curve[curve["converged"]].sort_values("value")
                if curve.empty:
                    continue
                best = float(curve["cost"].min())
                verdict = verdicts.get(subject, "identifiable")
                axis.plot(curve["value"], curve["cost"] / best,
                          color=VERDICT_COLOURS.get(verdict, "0.5"),
                          lw=2.0 if subject == middle else 0.9,
                          alpha=1.0 if subject == middle else 0.25, zorder=3 if subject == middle else 2)
                block = read_profile(curve)
                if block:
                    thresholds.append(block["threshold"] / block["best_cost"])
            if thresholds:
                axis.axhline(float(np.median(thresholds)), color=_figures.THRESHOLD, lw=1.0, ls="--")
            for bound in ("bound_low", "bound_high"):
                edge = float(group[bound].iloc[0])
                if np.isfinite(edge) and edge > 0:
                    axis.axvline(edge, color=_figures.BOUND, lw=3.0, alpha=0.45)

            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.set_xlabel(parameter, fontsize=8)
            if column == 0:
                axis.set_ylabel(f"{_figures.label(route)}\ncost / best cost", fontsize=8)
            axis.tick_params(labelsize=6.5)
            axis.grid(alpha=0.25)
            shares = [f"{verdict} {sum(1 for value in verdicts.values() if value == verdict)}"
                      f"/{len(verdicts)}" for verdict in VERDICT_COLOURS if verdicts]
            _figures.note(axis, "\n".join(shares) if shares else "no verdict stored", size=6)

    fig.suptitle("Practical identifiability, one profile per participant", fontsize=13,
                 fontweight="bold")
    _figures.caption(fig, "Colour: the verdict of that participant's profile. Thick: the middle "
                          "participant. Dashed grey: the median 95 % threshold. Red: the parameter's "
                          "bounds.\n"
                          "The x axis is logarithmic because the intervals cover orders of magnitude, "
                          "which is why ci_width_decades is written in decades.")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    _shared.save_figure(fig, ANALYSIS, name)


def figure_compensation(detail: pd.DataFrame, name="compensation") -> None:
    """
    Which parameter takes over when a profiled one is held away from its optimum.

    Read a row as: this parameter was held at the value on the x axis, and the column parameter was
    re-optimised to whatever the y axis says, both relative to the best fit. A trajectory that falls as
    the other rises is a compensation, and it is why the profiled parameter is not identifiable alone.
    """
    if detail.empty:
        return
    routes = _figures.ordered(list(dict.fromkeys(detail["route"])))
    profiled = [name_of for name_of in _shared.PARAMETERS if name_of in set(detail["parameter"])]
    refitted = [name_of for name_of in _shared.PARAMETERS if f"refit_{name_of}" in detail]
    if not profiled or not refitted:
        return

    for route in routes:
        block = detail[detail["route"] == route]
        fig, axes = plt.subplots(len(profiled), len(refitted),
                                 figsize=(2.3 * len(refitted) + 1, 2.3 * len(profiled) + 1),
                                 squeeze=False)
        for row, held in enumerate(profiled):
            for column, other in enumerate(refitted):
                axis = axes[row][column]
                axis.set_xticks([])
                axis.set_yticks([])
                if held == other:
                    axis.set_facecolor("0.95")
                    if row == 0:
                        axis.set_title(other, fontsize=8)
                    if column == 0:
                        axis.set_ylabel(held, fontsize=8)
                    continue

                slopes = []
                for subject, curve in block[block["parameter"] == held].groupby("subject"):
                    curve = curve[curve["converged"]].sort_values("value")
                    if len(curve) < 3 or f"refit_{other}" not in curve:
                        continue
                    best_value = float(curve["best_value"].iloc[0])
                    at_best = curve.iloc[(curve["value"] - best_value).abs().argmin()][f"refit_{other}"]
                    x = curve["value"].to_numpy(float) / (best_value or 1.0)
                    y = curve[f"refit_{other}"].to_numpy(float) / (float(at_best) or 1.0)
                    usable = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
                    if usable.sum() < 3:
                        continue
                    axis.plot(x[usable], y[usable], color="0.35", lw=0.7, alpha=0.35)
                    slopes.append(np.polyfit(np.log10(x[usable]), np.log10(y[usable]), 1)[0])
                if slopes:
                    middle = float(np.median(slopes))
                    axis.set_facecolor(plt.get_cmap("PuOr")(0.5 + max(-1.0, min(1.0, middle / 2)) / 2))
                    axis.text(0.04, 0.92, f"slope {middle:+.2f}", transform=axes[row][column].transAxes,
                              fontsize=6, va="top")
                axis.set_xscale("log")
                axis.set_yscale("log")
                axis.axhline(1.0, color="0.6", lw=0.6)
                axis.axvline(1.0, color="0.6", lw=0.6)
                if row == 0:
                    axis.set_title(other, fontsize=8)
                if column == 0:
                    axis.set_ylabel(held, fontsize=8)

        fig.suptitle(f"What compensates what, {_figures.label(route)}", fontsize=12, fontweight="bold")
        _figures.caption(fig, "Row: the parameter held at the value on the x axis, relative to its best "
                              "fit. Column: the parameter re-optimised at each of those values, relative "
                              "to its own best fit.\n"
                              "This is the trajectory of a refit, not a correlation between estimates. "
                              "A firmly negative slope, drawn orange, is a compensation; purple is a "
                              "parameter that follows the profiled one instead.")
        fig.tight_layout(rect=(0, 0.06, 1, 0.94))
        _shared.save_figure(fig, ANALYSIS, f"{name}_{route}")


def main():
    _shared.headless()
    parser = _shared.arguments(__doc__.splitlines()[1], routes=("sensor_id",))
    parser.add_argument("--parameters", nargs="+", default=["km_rest", "tau2", "tau1_rest"],
                        help="which step 2 parameters to profile")
    parser.add_argument("--points", type=int, default=DEFAULT_POINTS, help="grid points per profile")
    parser.add_argument("--span", type=float, default=DEFAULT_SPAN,
                        help="the grid runs from value/span to value*span")
    parser.add_argument("--max-iter", type=int, default=steps.MAX_ITER,
                        help=f"iterations per solve, capped at {steps.MAX_ITER} whatever is asked")
    parser.add_argument("--refresh", action="store_true", help="re-run participants already on disk")
    arguments = parser.parse_args()

    print(f"profiling {', '.join(arguments.parameters)}")
    detail = _campaign.per_subject(
        ANALYSIS, arguments,
        lambda route, subject: profile_all(route, subject, arguments.parameters, arguments.points,
                                           arguments.span, arguments.max_iter),
    )
    if detail.empty:
        return
    summary = summarise(detail)
    _shared.banner("profile likelihood verdicts")
    print(summary.round(5).to_string(index=False))
    _shared.write(detail, ANALYSIS, "all_profiles")
    _shared.write(summary, ANALYSIS, "summary")
    figure(detail, summary)
    figure_compensation(detail)


if __name__ == "__main__":
    main()
