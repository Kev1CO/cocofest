"""
Which parameters the data actually resolves, read off the bounds and off what step 3 changes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import _shared

import matplotlib.pyplot as plt
from matplotlib import ticker

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


def figure(routes, name="parameter_bounds") -> None:
    """Every parameter of every route, on a log axis, with its bounds drawn as a band."""
    tables = {route: _shared.cohort(route) for route in routes}
    routes = [route for route in routes if not tables[route].empty]
    if not routes:
        return

    fig, axes = plt.subplots(1, len(_shared.PARAMETERS), figsize=(3.0 * len(_shared.PARAMETERS), 4.4),
                             squeeze=False)
    for axis, name_of in zip(axes[0], _shared.PARAMETERS):
        series, labels = [], []
        for route in routes:
            table = tables[route]
            if name_of in table:
                series.append(table[name_of].astype(float).to_numpy())
                labels.append(route.replace("_id", ""))
        if not series:
            continue
        _shared.boxplot(axis, series, labels, showfliers=True, flierprops={"markersize": 2})

        # a_scale is bounded per participant, so its floor is the lowest any route allowed, not a median
        drawn = [np.concatenate(series)]
        floors = [float(tables[route][f"{name_of}__min"].astype(float).min())
                  for route in routes if f"{name_of}__min" in tables[route]]
        if floors:
            axis.axhline(min(floors), color="C3", lw=0.9, ls=":")
            drawn.append([min(floors)])
        ceilings = tables[routes[0]].get(f"{name_of}__max")
        if ceilings is not None:
            drawn.append([float(ceilings.astype(float).median())])
            axis.axhline(drawn[-1][0], color="C3", lw=0.9, ls="--")
        axis.set_yscale("log")

        # The decades enclosing what is drawn, so every panel ticks in whole decades and none carries an
        # empty one; taken from the values rather than the autoscaled limits, whose margin costs a decade
        values = np.concatenate([np.asarray(part, dtype=float).ravel() for part in drawn])
        values = values[np.isfinite(values) & (values > 0)]
        if values.size:
            # A hair beyond the decade, or a bound landing exactly on one is drawn under the frame
            axis.set_ylim(10.0 ** (np.floor(np.log10(values.min()) + 1e-9) - 0.05),
                          10.0 ** (np.ceil(np.log10(values.max()) - 1e-9) + 0.05))
        axis.yaxis.set_major_locator(ticker.LogLocator(base=10.0))
        axis.yaxis.set_major_formatter(ticker.LogFormatterMathtext(base=10.0))
        axis.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=tuple(range(2, 10))))
        axis.yaxis.set_minor_formatter(ticker.NullFormatter())
        axis.set_title(name_of, fontsize=8)
        axis.tick_params(axis="x", labelrotation=90, labelsize=7)
        axis.tick_params(axis="y", which="major", labelsize=7, length=3.5)
        axis.tick_params(axis="y", which="minor", length=2)

    fig.suptitle("identified parameters against their bounds "
                 "(red: lower dotted, the lowest of the routes; upper dashed)", fontsize=9)
    fig.tight_layout()
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


if __name__ == "__main__":
    main()
