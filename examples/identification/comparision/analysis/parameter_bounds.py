"""
Which parameters the data actually resolves, read off the bounds and off what step 3 changes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import _figures
import _shared

import matplotlib.pyplot as plt
from matplotlib import ticker

import ding_model
import protocol
import steps

ANALYSIS = "parameter_bounds"

PINNED_SHARE = 0.7
FLAT_CV_PCT = 5.0


def bounds(routes) -> pd.DataFrame:
    """One row per route and per parameter: where it landed, and against which bound."""
    rows = []
    for route in routes:
        table = _shared.cohort(route)
        if table.empty:
            continue
        for name in _shared.PARAMETERS:
            if name not in table:
                continue
            values = table[name].astype(float)
            low = table.get(f"{name}__min", pd.Series(np.nan, index=table.index)).astype(float)
            high = table.get(f"{name}__max", pd.Series(np.nan, index=table.index)).astype(float)
            margin = 1e-3 * (high - low)
            rows.append({
                "route": route, "parameter": name, "n": len(values),
                "median": float(values.median()),
                "q1": float(values.quantile(0.25)), "q3": float(values.quantile(0.75)),
                "cv_pct": float(100 * values.std() / values.mean()) if values.mean() else np.nan,
                "at_bound": int(sum(name in entry for entry in table["at_bound"])),
                "at_low": int(((values - low) <= margin).sum()),
                "at_high": int(((high - values) <= margin).sum()),
                "bound_low": float(low.median()), "bound_high": float(high.median()),
            })
    return pd.DataFrame(rows)


def refinement(routes) -> pd.DataFrame:
    """How far the step 3 values of A travel from the step 2 one, and what eq. 3 makes of them."""
    rows = []
    for route in routes:
        table = _shared.cohort(route)
        for _, entry in table.iterrows():
            extra = entry["extra"]
            points = extra.get("a_points_us", {})
            if not points:
                continue
            values = np.array([float(value) for value in points.values()])
            reference = float(points.get(str(extra.get("reference_pulse_width_us", 400)), np.nan))
            rows.append({
                "route": route, "subject": entry["subject"], "n_intensities": len(values),
                "a_reference": reference,
                "a_min": float(values.min()), "a_max": float(values.max()),
                "range_pct_of_median": float(100 * (values.max() - values.min()) / np.median(values)),
                "cv_around_reference_pct": float(100 * values.std(ddof=1) / reference)
                if np.isfinite(reference) and reference else np.nan,
                "eq3_r2": float(extra.get("a_pulse_width_r2", np.nan)),
                "eq3_saturated": bool(extra.get("eq3_saturated", False)),
                "a_monotonic": bool(extra.get("a_pulse_width_monotonic", False)),
            })
    return pd.DataFrame(rows)


def refinement_summary(detail: pd.DataFrame) -> pd.DataFrame:
    """The A refinement pooled per route, the table the report quotes."""
    if detail.empty:
        return detail
    return (detail.groupby("route", sort=False)
            .agg(n=("subject", "count"),
                 range_pct_of_median=("range_pct_of_median", "median"),
                 cv_around_reference_pct=("cv_around_reference_pct", "median"),
                 eq3_r2=("eq3_r2", "median"),
                 eq3_saturated=("eq3_saturated", "sum"),
                 a_monotonic=("a_monotonic", "sum"))
            .reset_index())


def shortlist(bound_table: pd.DataFrame) -> pd.DataFrame:
    """The parameters a route should fix rather than identify, on the evidence of this program alone."""
    if bound_table.empty:
        return bound_table
    pinned = bound_table["at_bound"] >= PINNED_SHARE * bound_table["n"]
    flat = bound_table["cv_pct"].fillna(0.0) <= FLAT_CV_PCT
    picked = bound_table[pinned].copy()
    picked["no_dispersion_left"] = flat[pinned]
    picked["which_bound"] = np.where(picked["at_low"] >= picked["at_high"], "lower", "upper")
    return picked[["route", "parameter", "n", "at_bound", "which_bound", "cv_pct",
                   "no_dispersion_left", "median"]]


# a_scale and A start from the measured peak, so they have no single initial value to draw
GUESSED = ("km_rest", "tau1_rest", "tau2", "pd0", "pdt")


def figure(routes, name="parameter_bounds") -> None:
    """
    Every parameter of every route against its own bounds, on a log axis, one point per participant.

    A parameter every participant is pinned against is drawn as a flat line of hollow markers. That is the
    result and the panel says so, rather than leaving it to look like a drawing fault.
    """
    order = [route for route in _figures.ordered(routes) if not _shared.cohort(route).empty]
    if not order:
        return
    tables = {route: _shared.cohort(route) for route in order}
    counted = bounds(order)

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.6))
    for axis, parameter in zip(axes.ravel(), _shared.PARAMETERS):
        told = []
        for index, route in enumerate(order):
            table = tables[route]
            if parameter not in table:
                continue
            values = table[parameter].astype(float).to_numpy()
            pinned = np.array([parameter in entry for entry in table["at_bound"]], dtype=bool)
            axis.boxplot([values], positions=[index], widths=0.55, showfliers=False,
                         medianprops=dict(color="black", linewidth=1.2))
            _figures.points(axis, index, values, route, at_bound=pinned, size=16)

            for column in (f"{parameter}__min", f"{parameter}__max"):
                if column in table:
                    edge = float(table[column].astype(float).median())
                    if np.isfinite(edge) and edge > 0:
                        axis.plot([index - 0.45, index + 0.45], [edge, edge], color=_figures.BOUND,
                                  lw=3.0, alpha=0.55, solid_capstyle="butt", zorder=1)

            row = counted[(counted["route"] == route) & (counted["parameter"] == parameter)]
            if not row.empty:
                entry = row.iloc[0]
                told.append(f"{_figures.label(route)}: {int(entry['at_bound'])}/{int(entry['n'])} at a "
                            f"bound ({int(entry['at_low'])} low, {int(entry['at_high'])} high), "
                            f"CV {entry['cv_pct']:.0f} %"
                            + ("  -> the box is a line because every participant is pinned there, which "
                               "is the result" if entry["at_bound"] >= entry["n"] and entry["cv_pct"] < 1
                               else ""))

        if parameter in GUESSED:
            axis.axhline(steps.IDENTIFIED_PARAMETERS[parameter][0], color=_figures.BOUND, lw=0.9, ls="--")
        axis.set_yscale("log")
        axis.set_title(parameter, fontsize=10, fontweight="bold")
        _figures.route_ticks(axis, order)
        axis.tick_params(axis="y", which="major", labelsize=7)
        axis.tick_params(axis="y", which="minor", labelsize=5)
        axis.yaxis.set_minor_formatter(ticker.NullFormatter())
        axis.grid(alpha=0.3, axis="y")
        _figures.note(axis, "\n".join(told), size=5.5, x=0.02, y=-0.30, va="top")

    fig.suptitle("What the data resolves, and what the optimiser pushes against a bound",
                 fontsize=13, fontweight="bold")
    _figures.caption(fig, "Thick red: the bounds this route was given. Thin red dashed: the value the "
                          "parameter started from. Hollow marker: the solver declared that value on a "
                          "bound.\n"
                          "at_bound is the solver's own word and is what counts; at_low and at_high count "
                          "values within a thousandth of a bound and can differ from it.\n"
                          "A parameter pinned on a candidate route and free on the reference is a point "
                          "of infeasibility, not a nuance.")
    fig.tight_layout(rect=(0, 0.10, 1, 0.94), h_pad=6.5)
    _shared.save_figure(fig, ANALYSIS, name)


def figure_a_refinement(detail: pd.DataFrame, name="a_refinement") -> None:
    """What step 3 buys: how far the six A travel with the pulse width, and what eq. 3 makes of them."""
    if detail.empty:
        return
    order = [route for route in _figures.ordered(list(dict.fromkeys(detail["route"])))]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))

    axis = axes[0]
    saturated, monotonic = {}, {}
    for route in order:
        table = _shared.cohort(route)
        saturated[route] = int(detail.loc[detail["route"] == route, "eq3_saturated"].sum())
        monotonic[route] = int(detail.loc[detail["route"] == route, "a_monotonic"].sum())
        for _, entry in table.iterrows():
            extra = entry["extra"]
            points = extra.get("a_points_us", {})
            reference = float(points.get(str(extra.get("reference_pulse_width_us", 400)), np.nan))
            if not points or not np.isfinite(reference) or not reference:
                continue
            widths = np.array(sorted(float(width) for width in points))
            values = np.array([float(points[f"{width:.0f}"]) for width in widths]) / reference
            pinned = bool(extra.get("eq3_saturated", False))
            axis.plot(widths, values, color=_figures.BOUND if pinned else _figures.colour(route),
                      lw=0.8, alpha=0.45 if pinned else 0.30, zorder=2)

    grid = np.linspace(min(protocol.PULSE_WIDTHS_US), max(protocol.PULSE_WIDTHS_US), 100)
    for route in order:
        table = _shared.cohort(route)
        middle = {name_of: float(table[name_of].astype(float).median())
                  for name_of in ("a_scale", "pd0", "pdt") if name_of in table}
        if len(middle) == 3:
            curve = ding_model.a_calculation(grid * 1e-6, middle["a_scale"], middle["pd0"], middle["pdt"])
            at_reference = ding_model.a_calculation(protocol.REFERENCE_PULSE_WIDTH_US * 1e-6,
                                                    middle["a_scale"], middle["pd0"], middle["pdt"])
            axis.plot(grid, curve / at_reference, color=_figures.colour(route), lw=2.4, zorder=4,
                      label=f"eq. 3, {_figures.label(route)} (median parameters)")
    axis.axhline(1.0, color="0.5", lw=0.8)
    axis.axvline(protocol.REFERENCE_PULSE_WIDTH_US, color="0.5", lw=0.8, ls=":")
    axis.set_xlabel(r"pulse width ($\mu$s)", fontsize=9)
    axis.set_ylabel(f"A / A at {protocol.REFERENCE_PULSE_WIDTH_US} us", fontsize=9)
    axis.set_title("(a) the six A of each participant", fontsize=10, fontweight="bold", loc="left")
    axis.legend(fontsize=6)
    axis.grid(alpha=0.3)
    _figures.note(axis, "red: a participant whose a' saturated at its floor", size=6)

    axis = axes[1]
    for index, route in enumerate(order):
        block = detail[detail["route"] == route]
        values = block["eq3_r2"].to_numpy(float)
        _figures.half_violin(axis, values, index, route)
        spread = _figures.offsets(len(values), side=1)
        for offset, value, pinned, rising in zip(spread, values, block["eq3_saturated"],
                                                 block["a_monotonic"]):
            axis.scatter(index + offset, value, s=18, zorder=3,
                         marker="o" if rising else "X",
                         color=_figures.BOUND if pinned else _figures.colour(route))
        _figures.median_bar(axis, values, index)

    axis.axhline(0.0, color="0.5", lw=0.8)
    axis.set_ylabel(r"$R^2$ of eq. 3 through the six A", fontsize=9)
    axis.set_title("(b) what eq. 3 makes of them", fontsize=10, fontweight="bold", loc="left")
    _figures.route_ticks(axis, order)
    axis.grid(alpha=0.3, axis="y")
    fig.suptitle("What step 3 adds, route by route", fontsize=13, fontweight="bold")
    _figures.caption(fig, "X marker: A does not rise with the pulse width for that participant. Red: a' "
                          "sits on its floor, so eq. 3 has saturated.\n"
                          + "   ".join(f"{_figures.label(route)}: {saturated[route]} saturated, "
                                       f"{monotonic[route]} monotonic of "
                                       f"{int((detail['route'] == route).sum())}" for route in order)
                          + "\nThese are counts, not rates: a_refinement.csv writes them that way.")
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    _shared.save_figure(fig, ANALYSIS, name)


def main():
    _shared.headless()
    parser = _shared.arguments(__doc__.splitlines()[1], subjects=False)
    routes = _shared.available(parser.parse_args().routes)

    _shared.banner("parameters against their bounds")
    bound_table = bounds(routes)
    print(bound_table[["route", "parameter", "n", "median", "cv_pct", "at_bound", "at_low", "at_high"]]
          .round(4).to_string(index=False))
    _shared.write(bound_table, ANALYSIS, "bounds")

    _shared.banner("A between step 2 and step 3, and what eq. 3 makes of it")
    detail = refinement(routes)
    print(refinement_summary(detail).round(2).to_string(index=False))
    _shared.write(detail, ANALYSIS, "a_refinement_per_subject")
    _shared.write(refinement_summary(detail), ANALYSIS, "a_refinement")

    _shared.banner("shortlist: fix rather than identify")
    picked = shortlist(bound_table)
    print(picked.round(4).to_string(index=False) if not picked.empty else "  nothing pinned")
    _shared.write(picked, ANALYSIS, "shortlist")
    figure(routes)
    figure_a_refinement(detail)


if __name__ == "__main__":
    main()
