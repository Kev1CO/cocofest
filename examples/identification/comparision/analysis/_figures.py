"""
The drawing conventions the figures share, so no two of them contradict each other.

    sensor_id first, always the same colour      hollow marker: the value sat on a bound
    red: a bound of the optimisation problem     grey dashed: a reading threshold
    one unit per axis                            the count is written on every panel
    the jitter is deterministic
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import stats

import _shared

import routes as route_table

REFERENCE = "sensor_id"
CANDIDATES = tuple(name for name in _shared.ROUTES if name != REFERENCE)

BOUND = "red"  # a bound of the optimisation problem, nothing else
THRESHOLD = "0.45"  # a reading threshold, dashed, nothing else

JITTER_SEED = 20250901
JITTER_WIDTH = 0.085

# The one sentence every figure carrying both comparisons has to print
ASYMMETRY = ("sensor_id -> muscle_force_isometric_id reads the same recording twice, so the gap is the "
             "observable alone.\nsensor_id -> movement_id reads two different recordings, so the gap "
             "confounds the observable with the protocol.")

# Four parameters of six never see more than the reference column
STEP2_ONLY = ("A", "km_rest", "tau1_rest", "tau2")
EQ3_ONLY = ("a_scale", "pd0", "pdt")
REFERENCE_TRAINS = 3


# ---- Who is who ---- #
def ordered(routes=None) -> list:
    """The routes with the reference first, which is where a reader looks for it in every figure."""
    names = list(routes if routes is not None else _shared.ROUTES)
    return ([REFERENCE] if REFERENCE in names else []) + [name for name in names if name != REFERENCE]


def label(route: str) -> str:
    """The route's own short name, as routes.py spells it."""
    try:
        return route_table.of(route).label
    except KeyError:
        return route.replace("_id", "")


def colour(route: str) -> str:
    """The route's own colour, the same one from the first figure to the last."""
    try:
        return route_table.of(route).colour
    except KeyError:
        return "0.4"


def unit_of(route: str) -> str:
    """The unit that route's observable is written in."""
    try:
        return route_table.of(route).UNIT
    except KeyError:
        return ""


# ---- The small print every panel carries ---- #
def note(axis, text: str, x=0.02, y=0.98, size=5.5, ha="left", va="top") -> None:
    """One annotation box, drawn the same way in every figure."""
    axis.text(x, y, text, transform=axis.transAxes, ha=ha, va=va, fontsize=size,
              bbox=dict(boxstyle="round,pad=0.28", facecolor="white", alpha=0.78, linewidth=0.3))


def count(axis, text: str, size=5.5) -> None:
    """The effectif of one panel, which the holes in the design make unequal from panel to panel."""
    axis.text(0.98, 0.02, text, transform=axis.transAxes, ha="right", va="bottom", fontsize=size,
              color="0.25")


def caption(figure, text: str, size=6.0) -> None:
    """The lines a figure has to carry to be read the right way round."""
    figure.text(0.01, 0.005, text, ha="left", va="bottom", fontsize=size, color="0.25")


# ---- Points, boxes and violins ---- #
def jitter(n: int, width=JITTER_WIDTH) -> np.ndarray:
    """A deterministic spread, so two compilations write the same file and paired lines meet the dots."""
    return np.random.default_rng(JITTER_SEED).uniform(-width, width, int(n))


def offsets(n: int, side=0, width=JITTER_WIDTH) -> np.ndarray:
    """The x offset of every point of one group: centred, or pushed to one side of a violin."""
    spread = jitter(n, width)
    return spread if side == 0 else side * (0.06 + np.abs(spread))


def points(axis, position, values, route: str, at_bound=None, side=0, size=13, offset=None) -> np.ndarray:
    """One dot per participant, hollow when the value sat on a bound."""
    values = np.asarray(values, float)
    spread = offsets(len(values), side) if offset is None else np.asarray(offset, float)
    marks = np.zeros(len(values), bool) if at_bound is None else np.asarray(at_bound, bool)
    tint = colour(route)
    for x, value, bound in zip(position + spread, values, marks):
        axis.scatter(x, value, s=size, zorder=3,
                     facecolor="none" if bound else tint, edgecolor=BOUND if bound else tint,
                     linewidths=1.1 if bound else 0.5)
    return position + spread


def half_violin(axis, values, position, route: str, width=0.30, side=-1) -> None:
    """Half a density, so the dots of the same group sit on the other half rather than on top of it."""
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if values.size < 3 or np.allclose(values, values[0]):
        return
    density = stats.gaussian_kde(values)
    grid = np.linspace(values.min(), values.max(), 128)
    shape = width * density(grid) / density(grid).max()
    axis.fill_betweenx(grid, position, position + side * shape, facecolor=colour(route), alpha=0.30,
                       edgecolor=colour(route), linewidth=0.6)


def median_bar(axis, values, position, width=0.34, colour_of="black") -> float:
    """The median as a thick rule, the only summary a n = 20 panel needs next to its points."""
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not values.size:
        return np.nan
    centre = float(np.median(values))
    axis.plot([position - width, position + width], [centre, centre], color=colour_of, lw=2.0, zorder=4)
    return centre


# ---- Paired drawings, the only honest ones at n = 20 ---- #
def paired_lines(axis, positions, frame: pd.DataFrame, column: str, routes, spread=None,
                 alpha=0.35, lw=0.5) -> None:
    """One thin line per participant across the routes of a panel, which is what a boxplot hides."""
    wide = frame.pivot_table(index="subject", columns="route", values=column, aggfunc="mean")
    wide = wide.reindex(columns=[route for route in routes if route in wide])
    if wide.empty:
        return
    spread = np.zeros(len(wide)) if spread is None else np.asarray(spread, float)
    for i, (_, row) in enumerate(wide.iterrows()):
        x = [positions[routes.index(route)] + spread[i] for route in wide.columns]
        axis.plot(x, row.to_numpy(float), color="0.55", lw=lw, alpha=alpha, zorder=1)


def slope_lines(axis, left, right, smaller_is_better=True, x=(0, 1), labels=None) -> dict:
    """A paired slope diagram: one line per participant, coloured by which way it went."""
    left, right = np.asarray(left, float), np.asarray(right, float)
    usable = np.isfinite(left) & np.isfinite(right)
    left, right = left[usable], right[usable]
    better, worse = 0, 0
    for start, stop in zip(left, right):
        improved = (stop < start) if smaller_is_better else (stop > start)
        better, worse = better + int(improved), worse + int(not improved)
        axis.plot(x, [start, stop], color="tab:green" if improved else "tab:red", lw=0.8, alpha=0.65,
                  zorder=2)
        axis.scatter(x, [start, stop], s=8, color="0.35", zorder=3)
    for position, values in zip(x, (left, right)):
        median_bar(axis, values, position, width=0.10)
    axis.set_xticks(list(x))
    axis.set_xticklabels(list(labels) if labels else ["", ""], fontsize=6)
    axis.set_xlim(x[0] - 0.35, x[1] + 0.35)
    return {"n": int(len(left)), "better": better, "worse": worse,
            "left_median": float(np.median(left)) if len(left) else np.nan,
            "right_median": float(np.median(right)) if len(right) else np.nan}


# ---- Identity plots ---- #
def bisector(axis, values, colour_of="0.4") -> None:
    """The y = x line, and square limits, so a point above it reads as an overestimate."""
    values = np.asarray([v for v in np.ravel(values) if np.isfinite(v)], float)
    if not values.size:
        return
    low, high = float(values.min()), float(values.max())
    pad = 0.05 * (high - low or abs(high) or 1.0)
    limits = (low - pad, high + pad)
    axis.plot(limits, limits, color=colour_of, lw=0.8, ls="--", zorder=1)
    axis.set_xlim(limits)
    axis.set_ylim(limits)
    axis.set_aspect("equal", adjustable="box")


def log_bisector(axis, values, colour_of="0.4") -> None:
    """The same on two log axes, for the parameters, whose panels span decades."""
    values = np.asarray([v for v in np.ravel(values) if np.isfinite(v) and v > 0], float)
    if not values.size:
        return
    low, high = float(values.min()) / 1.6, float(values.max()) * 1.6
    axis.plot([low, high], [low, high], color=colour_of, lw=0.8, ls="--", zorder=1)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(low, high)
    axis.set_ylim(low, high)
    axis.set_aspect("equal", adjustable="box")


def route_ticks(axis, routes, size=6.5) -> None:
    """One tick per route, in the order the reference comes first."""
    axis.set_xticks(range(len(routes)))
    axis.set_xticklabels([label(route) for route in routes], rotation=20, ha="right", fontsize=size)


@lru_cache(maxsize=None)
def illustration_subject(route=REFERENCE) -> str:
    """The participant every illustration is drawn on: the one nearest the cohort median NRMSE."""
    scores = {subject: _shared.normalised_rmse(*_shared.signal_of(route, subject)[1:])
              for subject in _shared.subjects_of(route)}
    scores = {subject: value for subject, value in scores.items() if np.isfinite(value)}
    if not scores:
        return ""
    middle = float(np.median(list(scores.values())))
    return min(scores, key=lambda subject: abs(scores[subject] - middle))
