"""
How accurate a prediction is, and how wide an interval one has to draw around it to be right 95% of the time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import _figures
import _shared

import matplotlib.pyplot as plt

ANALYSIS = "prediction_accuracy"


def interval(residuals) -> dict:
    """The 95% interval on one prediction, both ways, with the test that says which one to quote."""
    residuals = np.asarray(residuals, float)
    residuals = residuals[np.isfinite(residuals)]
    if len(residuals) < 3:
        return {}
    low, high = np.percentile(residuals, [2.5, 97.5])
    spread = float(residuals.std(ddof=1))
    normal = stats.shapiro(residuals) if len(residuals) <= 5000 else None
    return {
        "n": int(len(residuals)),
        "bias_pct": float(residuals.mean()),
        "sd_pct": spread,
        "pi95_empirical_low": float(low),
        "pi95_empirical_high": float(high),
        "pi95_gaussian_low": float(residuals.mean() - 1.96 * spread),
        "pi95_gaussian_high": float(residuals.mean() + 1.96 * spread),
        "shapiro_p": float(normal.pvalue) if normal is not None else np.nan,
        "quote": "empirical" if normal is not None and normal.pvalue < 0.05 else "gaussian",
    }


def accuracy(route: str) -> dict:
    """Every accuracy measure of one route, pooled over its testing trains and participants."""
    held_out = _shared.testing(route)
    if held_out.empty:
        return {}

    fits, errors = [], []
    for subject in _shared.subjects_of(route):
        _, measured, predicted = _shared.signal_of(route, subject)
        fits.append(_shared.best_fit_rate(measured, predicted))
        errors.append(_shared.normalised_rmse(measured, predicted))

    return {
        "route": route,
        "n_trains": int(len(held_out)),
        "n_subjects": len(_shared.subjects_of(route)),
        "peak_error_pct": float(held_out["peak_error"].mean()),
        "peak_error_sd_pct": float(held_out["peak_error"].std()),
        "fti_error_pct": float(held_out["fti_error"].mean()),
        "fti_error_sd_pct": float(held_out["fti_error"].std()),
        "r2": float(held_out["r2"].mean()),
        "r2_sd": float(held_out["r2"].std()),
        "bfr_pct": float(np.nanmean(fits)),
        "bfr_sd_pct": float(np.nanstd(fits, ddof=1)),
        "nrmse_pct": float(np.nanmean(errors)),
        "nrmse_sd_pct": float(np.nanstd(errors, ddof=1)),
    }


def table(routes) -> pd.DataFrame:
    return pd.DataFrame([block for block in (accuracy(route) for route in routes) if block])


def intervals(routes) -> pd.DataFrame:
    """The 95% prediction interval of every route and every scalar."""
    rows = []
    for route in routes:
        held_out = _shared.testing(route)
        if held_out.empty:
            continue
        for scalar in _shared.SCALARS:
            block = interval(held_out[f"{scalar}_signed_error"])
            if block:
                rows.append({"route": route, "scalar": scalar, **block})
    return pd.DataFrame(rows)


def per_subject(routes) -> pd.DataFrame:
    """The same accuracy one participant at a time, the frame the cohort figures box-plot."""
    rows = []
    for route in routes:
        held_out = _shared.testing(route)
        if held_out.empty:
            continue
        for subject, group in held_out.groupby("subject"):
            _, measured, predicted = _shared.signal_of(route, subject)
            rows.append({
                "route": route, "subject": subject, "n_trains": len(group),
                "peak_error_pct": float(group["peak_error"].mean()),
                "fti_error_pct": float(group["fti_error"].mean()),
                "r2": float(group["r2"].mean()),
                "bfr_pct": _shared.best_fit_rate(measured, predicted),
                "nrmse_pct": _shared.normalised_rmse(measured, predicted),
            })
    return pd.DataFrame(rows)


SCORES = (("peak_error_pct", "peak %model error", True),
          ("fti_error_pct", "integral %model error", True),
          ("r2", r"$R^2$ on the shape of the signal", False),
          ("bfr_pct", "Best Fit Rate (%)", False),
          ("nrmse_pct", "NRMSE (% of the measured range)", False))

# The line scores.svg has always drawn. It is this study's own reading line, not a published threshold
READING_LINE = 25.0


def figure(routes, name="prediction_accuracy") -> None:
    """
    The five dimensionless scores, one panel each, the reference route first.

    Half a violin, the participants as points on the other half, and one line per participant across the
    routes: at n = 20 a box alone hides exactly what the paired result says.
    """
    subjects = per_subject(routes)
    if subjects.empty:
        return
    order = [route for route in _figures.ordered(routes) if route in set(subjects["route"])]

    fig, axes = plt.subplots(1, len(SCORES), figsize=(4.0 * len(SCORES), 4.8))
    positions = list(range(len(order)))
    spread = _figures.offsets(subjects["subject"].nunique(), side=1)

    for axis, (column, label, has_line) in zip(axes, SCORES):
        _figures.paired_lines(axis, positions, subjects, column, order, spread=spread)
        for position, route in zip(positions, order):
            values = subjects.loc[subjects["route"] == route, column].to_numpy(float)
            _figures.half_violin(axis, values, position, route)
            _figures.points(axis, position, values, route, offset=spread[:len(values)])
            _figures.median_bar(axis, values, position)
        if has_line:
            axis.axhline(READING_LINE, color=_figures.THRESHOLD, ls="--", lw=0.9)
            _figures.note(axis, f"dashed: {READING_LINE:.0f} %, the reading line this study has always\n"
                                f"drawn. It is a convention of the study, not a published threshold.",
                          size=5.5)
        if column == "r2":
            axis.axhline(0.0, color="0.6", lw=0.8)
            _figures.note(axis, "this is a coefficient of determination on the shape of the\n"
                                "signal, not a correlation: it can be firmly negative, and\n"
                                "the axis is left free so those participants stay visible",
                          size=5.5)
        axis.set_ylabel(label, fontsize=8)
        _figures.route_ticks(axis, order)
        axis.tick_params(axis="y", labelsize=7)
        axis.grid(alpha=0.3, axis="y")
        _figures.count(axis, f"{subjects['subject'].nunique()} participants")

    fig.suptitle("The five scores that compare between routes, one point per participant",
                 fontsize=13, fontweight="bold")
    _figures.caption(fig, "Only dimensionless scores are put side by side: N, N.s and deg do not share "
                          "an axis.\n" + _figures.ASYMMETRY)
    fig.tight_layout(rect=(0, 0.09, 1, 0.94))
    _shared.save_figure(fig, ANALYSIS, name)


def figure_intervals(routes, name="prediction_intervals") -> None:
    """The interval one would quote for a new participant, and the one the code says not to quote."""
    table = intervals(routes)
    if table.empty:
        return
    table = table.set_index(["route", "scalar"])
    rows = [(route, scalar) for route in _figures.ordered(routes) for scalar in _shared.SCALARS
            if (route, scalar) in table.index]

    fig, axis = plt.subplots(figsize=(11, 1.0 * len(rows) + 3.2))
    for position, (route, scalar) in enumerate(rows):
        block = table.loc[(route, scalar)]
        held_out = _shared.testing(route)
        errors = held_out[f"{scalar}_signed_error"].to_numpy(float)

        # The errors the interval is read off, drawn behind the interval rather than in a panel of their own
        heights, edges = np.histogram(errors[np.isfinite(errors)], bins=25, density=True)
        if heights.max() > 0:
            # The rows run downwards, so the density grows away from its own line rather than into the next
            axis.bar(0.5 * (edges[:-1] + edges[1:]), -0.8 * heights / heights.max(), bottom=position,
                     width=np.diff(edges), color=_figures.colour(route), alpha=0.18, zorder=1)

        for kind, width, alpha in (("empirical", 2.6, 1.0), ("gaussian", 1.0, 0.45)):
            quoted = kind == block["quote"]
            axis.plot([block[f"pi95_{kind}_low"], block[f"pi95_{kind}_high"]], [position] * 2,
                      color=_figures.colour(route), lw=width if quoted else 1.0,
                      alpha=1.0 if quoted else 0.45, solid_capstyle="butt", zorder=3,
                      label=None)
            axis.text(block[f"pi95_{kind}_high"], position + (0.16 if quoted else -0.28),
                      f" {kind}{' (quoted)' if quoted else ''}", fontsize=6, va="center",
                      color=_figures.colour(route), alpha=1.0 if quoted else 0.6)
        axis.scatter(block["bias_pct"], position, s=26, color=_figures.colour(route), zorder=4)
        axis.text(-0.01, position, f"{_figures.label(route)}  {scalar.upper()}", ha="right", va="center",
                  fontsize=8, transform=axis.get_yaxis_transform())
        axis.text(1.005, position, f"Shapiro p {block['shapiro_p']:.3g}   n {int(block['n'])}",
                  ha="left", va="center", fontsize=6, color="0.3",
                  transform=axis.get_yaxis_transform())

    axis.axvline(0.0, color="0.6", lw=0.8)
    axis.set_yticks([])
    axis.set_ylim(len(rows) - 0.2, -0.8)
    axis.set_xlabel("signed %model error on one testing train", fontsize=9)
    axis.grid(alpha=0.3, axis="x")
    _figures.note(axis, "The quote column has already decided: a Shapiro p under 5 % means the "
                        "percentiles,\nnot +/- 1.96 SD. A figure drawing the gaussian interval "
                        "everywhere would contradict it.", size=6)

    fig.suptitle("What error to announce for a participant nobody has identified yet",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _shared.save_figure(fig, ANALYSIS, name)


def main():
    _shared.headless()
    parser = _shared.arguments(__doc__.splitlines()[1], subjects=False)
    routes = _shared.available(parser.parse_args().routes)

    _shared.banner("prediction accuracy on the testing trains")
    cohort_table = table(routes)
    print(cohort_table.round(2).to_string(index=False))
    _shared.write(cohort_table, ANALYSIS, "cohort")

    _shared.banner("95% interval on one prediction")
    interval_table = intervals(routes)
    print(interval_table.round(1).to_string(index=False))
    _shared.write(interval_table, ANALYSIS, "prediction_intervals")

    _shared.write(per_subject(routes), ANALYSIS, "per_subject")
    figure(routes)
    figure_intervals(routes)


if __name__ == "__main__":
    main()
