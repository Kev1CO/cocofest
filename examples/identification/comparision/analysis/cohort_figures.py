"""
The cohort figures of every route, read off the stored npz and re-solving nothing.

    parameters      one box per route per parameter, the initial guess marked in red
    scores          what each route predicted on the trains it never saw
    agreement       measured against predicted, peak and time integral
    relationships   the force-frequency and force-intensity relationships the article reports
    step times      what each step of the identification costs
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import _shared

import metrics
import routes as route_table
import steps
import store
from helper import results

FIGURES_ROOT = store.RESULTS_ROOT / "figures"

ROUTES = {route.METHOD: (route.label, route.colour) for route in route_table.ROUTES}

PANELS = (
    ("plateau_at_reference", r"$A \cdot \tau_1$ (N)", 1.0, False),
    ("a_scale", r"$a'$ (N/s)", 1.0, False),
    ("km_rest", r"$K_m$", 1.0, True),
    ("tau1_rest", r"$\tau_1$ (ms)", 1e3, True),
    ("tau2", r"$\tau_2$ (ms)", 1e3, True),
    ("pd0", r"$pd_0$ ($\mu$s)", 1e6, True),
    ("pdt", r"$pd_t$ ($\mu$s)", 1e6, True),
)

GUESS_COLOUR = "red"
BOUND_COLOUR = "red"
CFT_FREQUENCIES = (20, 33, 50)


# ---- Reading what the six routes stored ---- #
def parameter_frame() -> pd.DataFrame:
    """One row per (route, subject, parameter): the value, and whether it sat on a bound."""
    rows = []
    for route in ROUTES:
        for subject, result in results.load_method(route).items():
            values = dict(result["parameters"])
            values["plateau_at_reference"] = result["extra"].get("plateau_at_reference", np.nan)
            bounded = set(result["parameters_at_bound"])
            for name, _, _, _ in PANELS:
                rows.append({"route": route, "subject": subject, "parameter": name,
                             "value": float(values.get(name, np.nan)), "at_bound": name in bounded})
    return pd.DataFrame(rows)


def score_frame(testing_only=True, cft_only=True) -> pd.DataFrame:
    """The per train scores of every route, every subject stacked, in that route's own units."""
    frames = []
    for route in ROUTES:
        for subject, result in results.load_method(route).items():
            table = pd.DataFrame(result["extra"]["trains"])
            if table.empty:
                continue
            table["route"], table["subject"] = route, subject
            table["unit"] = result["unit"]
            frames.append(table)
    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if table.empty:
        return table
    if testing_only:
        table = table[table["set"] == "testing"]
    if cft_only:
        table = table[table["frequency"].round().isin(CFT_FREQUENCIES)]
    return table


def timing_frame() -> pd.DataFrame:
    """One row per (route, subject) with the seconds each step took."""
    rows = []
    for route in ROUTES:
        for subject, result in results.load_method(route).items():
            seconds = result["extra"].get("step_seconds", {})
            if seconds:
                rows.append({"route": route, "subject": subject,
                             **{step: float(seconds.get(step, np.nan)) for step in STEPS}})
    return pd.DataFrame(rows)


STEPS = ("step1", "step2", "step3", "step4", "step5")
STEP_LABELS = {"step1": r"1  $\tau_1$", "step2": r"2  $A$, $K_m$", "step3": r"3  $A$ per intensity",
               "step4": "4  eq. 3", "step5": "5  prediction"}


def _save(figure, name) -> Path:
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
    path = FIGURES_ROOT / f"{name}.svg"
    figure.savefig(path, format="svg", bbox_inches="tight")
    plt.close(figure)
    print(f"  {path}")
    return path


def _boxes(axes, groups, colours):
    """One box per route on one axes, drawn the same way everywhere."""
    drawn = axes.boxplot(groups, widths=0.6, patch_artist=True, showfliers=False,
                         medianprops=dict(color="black", linewidth=1.2))
    for patch, colour in zip(drawn["boxes"], colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.45)
    return drawn


# ---- The parameters, one box per route ---- #
def figure_parameters(table=None, name="parameters") -> Path:
    """
    What each route identified, parameter by parameter, with the initial guess marked in red.
    """
    table = parameter_frame() if table is None else table
    if table.empty:
        raise ValueError(f"No stored result under {results.RESULTS_ROOT}")

    figure, axes_grid = plt.subplots(2, 4, figsize=(17, 8))
    labels = [ROUTES[route][0] for route in ROUTES]
    colours = [ROUTES[route][1] for route in ROUTES]

    for axes, (parameter, label, scale, has_guess) in zip(axes_grid.ravel(), PANELS):
        groups = []
        for i, route in enumerate(ROUTES):
            rows = table[(table["route"] == route) & (table["parameter"] == parameter)]
            values = scale * rows["value"].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            groups.append(values if values.size else np.array([np.nan]))
            spread = 0.10 * np.random.default_rng(0).standard_normal(len(rows))
            for x, value, bound in zip(spread + i + 1, scale * rows["value"], rows["at_bound"]):
                axes.plot(x, value, marker="o", markersize=4,
                          markerfacecolor="none" if bound else ROUTES[route][1],
                          markeredgecolor=BOUND_COLOUR if bound else ROUTES[route][1],
                          markeredgewidth=1.2 if bound else 0.5, zorder=3)
        _boxes(axes, groups, colours)

        if has_guess:
            axes.axhline(scale * steps.IDENTIFIED_PARAMETERS[parameter][0], color=GUESS_COLOUR,
                         linewidth=1.4, linestyle="--", zorder=1, label="initial guess")
            axes.legend(fontsize=7, loc="best")
        axes.set_title(label, fontsize=11, fontweight="bold")
        axes.set_xticks(range(1, len(ROUTES) + 1))
        axes.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
        axes.grid(alpha=0.3, axis="y")

    hidden = axes_grid.ravel()[len(PANELS)]
    hidden.axis("off")
    hidden.text(0.0, 0.95, "hollow point: the value sat on a bound\n"
                           "red dashed: where the parameter started\n\n"
                           "A and a' start from the measured peak, so\n"
                           "they have no single guess to draw",
                transform=hidden.transAxes, va="top", fontsize=9)

    figure.suptitle("What each route identified", fontsize=14, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(figure, name)


# ---- What each route predicted ---- #
def figure_scores(table=None, name="scores") -> Path:
    """The three numbers the article scores its prediction on, one box per route."""
    table = score_frame() if table is None else table
    figure, axes_grid = plt.subplots(1, 3, figsize=(15, 4.6))
    colours = [ROUTES[route][1] for route in ROUTES]

    for axes, (column, label) in zip(axes_grid, (("peak_error", "Peak error (%)"),
                                                 ("fti_error", "Integral error (%)"),
                                                 ("r2", r"$R^2$"))):
        groups = [table[table["route"] == route].groupby("subject")[column].mean().to_numpy()
                  for route in ROUTES]
        groups = [g if g.size else np.array([np.nan]) for g in groups]
        _boxes(axes, groups, colours)
        if column != "r2":
            axes.axhline(25, color="grey", linestyle=":", linewidth=1)
        axes.set_ylabel(label)
        axes.set_xticks(range(1, len(ROUTES) + 1))
        axes.set_xticklabels([ROUTES[route][0] for route in ROUTES], rotation=35, ha="right", fontsize=8)
        axes.grid(alpha=0.3, axis="y")

    figure.suptitle("What each route predicted on the trains it never saw, one point per participant",
                    fontsize=13, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(figure, name)


# ---- Measured against predicted ---- #
def _agreement_text(measured, predicted) -> str:
    """The agreement block of one panel, as the article reports it."""
    block = metrics.agreement(measured, predicted)
    if not np.isfinite(block["icc"]):
        return "\n".join([f"n = {block['n']}", "too few to score"])
    return "\n".join([
        f"ICC {block['icc']:.2f} [{block['icc_ci'][0]:.2f}, {block['icc_ci'][1]:.2f}]",
        f"slope {block['slope']:.2f}",
        f"error {block['model_error']:.0f} +/- {block['model_error_sd']:.0f} %",
        f"p {block['p']:.3f}   n {block['n']}",
    ])


def figure_agreement(table=None, name="agreement") -> Path:
    table = score_frame() if table is None else table
    figure, axes_grid = plt.subplots(2, len(ROUTES), figsize=(19, 7))

    for column, route in enumerate(ROUTES):
        rows = table[table["route"] == route]
        unit = rows["unit"].iloc[0] if not rows.empty else ""
        for row, (measure, name_of) in enumerate((("peak", f"PK ({unit})"), ("fti", f"FTI ({unit} s)"))):
            axes = axes_grid[row, column]
            measured = rows[f"{measure}_measured"].to_numpy(dtype=float)
            predicted = rows[f"{measure}_predicted"].to_numpy(dtype=float)
            axes.scatter(measured, predicted, s=10, color=ROUTES[route][1], alpha=0.7)
            if measured.size:
                limits = [min(measured.min(), predicted.min()), max(measured.max(), predicted.max())]
                axes.plot(limits, limits, color="black", linewidth=0.8, linestyle=":")
                axes.text(0.03, 0.97, _agreement_text(measured, predicted), transform=axes.transAxes,
                          va="top", ha="left", fontsize=6, family="monospace",
                          bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75, linewidth=0.4))
            axes.set_xlabel(f"measured {name_of}", fontsize=8)
            if column == 0:
                axes.set_ylabel(f"predicted {name_of}", fontsize=8)
            if row == 0:
                axes.set_title(ROUTES[route][0], fontsize=10, fontweight="bold")
            axes.tick_params(labelsize=7)
            axes.grid(alpha=0.3)

    figure.suptitle("Measured against predicted, every testing train of every participant",
                    fontsize=13, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(figure, name)


# ---- The two relationships the article reports ---- #
def figure_relationships(table=None, name="relationships") -> Path:
    """
    The force-frequency and force-intensity relationships, measured and predicted, route by route.
    """
    table = score_frame() if table is None else table
    figure, axes_grid = plt.subplots(2, len(ROUTES), figsize=(19, 7))

    for column, route in enumerate(ROUTES):
        rows = table[table["route"] == route].copy()
        if not rows.empty:
            largest = rows.groupby("subject")["peak_measured"].transform("max")
            rows["measured"] = rows["peak_measured"] / largest
            rows["predicted"] = rows["peak_predicted"] / largest

        for row, (by, label) in enumerate((("frequency", "Frequency (Hz)"),
                                           ("pulse_width_us", r"Pulse width ($\mu$s)"))):
            axes = axes_grid[row, column]
            for side, marker, colour in (("measured", "o", "black"),
                                         ("predicted", "D", ROUTES[route][1])):
                if rows.empty:
                    continue
                grouped = rows.groupby(by)[side].agg(["mean", "std"])
                axes.errorbar(grouped.index, grouped["mean"], yerr=grouped["std"], marker=marker,
                              markersize=4, capsize=3, linewidth=1, color=colour, label=side)
            axes.set_xlabel(label, fontsize=8)
            if column == 0:
                axes.set_ylabel("peak / participant's largest", fontsize=8)
            if row == 0:
                axes.set_title(ROUTES[route][0], fontsize=10, fontweight="bold")
            if row == 0 and column == 0:
                axes.legend(fontsize=7)
            axes.tick_params(labelsize=7)
            axes.grid(alpha=0.3)

    figure.suptitle("The force-frequency and force-intensity relationships, on the testing trains",
                    fontsize=13, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(figure, name)


# ---- What each step costs ---- #
def figure_step_times(table=None, name="step_times") -> Path:
    """What each step of the identification costs, one box per route."""
    table = timing_frame() if table is None else table
    figure, axes_grid = plt.subplots(1, len(STEPS), figsize=(18, 4.2), sharey=True)
    colours = [ROUTES[route][1] for route in ROUTES]

    for axes, step in zip(axes_grid, STEPS):
        groups = [table[table["route"] == route][step].dropna().to_numpy() for route in ROUTES]
        groups = [g if g.size else np.array([np.nan]) for g in groups]
        _boxes(axes, groups, colours)
        axes.set_title(STEP_LABELS[step], fontsize=10, fontweight="bold")
        axes.set_xticks(range(1, len(ROUTES) + 1))
        axes.set_xticklabels([ROUTES[route][0] for route in ROUTES], rotation=35, ha="right", fontsize=7)
        axes.set_yscale("log")
        axes.grid(alpha=0.3, axis="y")
    axes_grid[0].set_ylabel("seconds")

    figure.suptitle("What each step costs, one point per participant", fontsize=13, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(figure, name)


def main():
    """Draw every figure. Nothing here re-solves, so it is safe to run after any change."""
    _shared.headless()
    print(f"reading {results.RESULTS_ROOT}")
    parameters, scores, timings = parameter_frame(), score_frame(), timing_frame()
    figure_parameters(parameters)
    figure_scores(scores)
    figure_agreement(scores)
    figure_relationships(scores)
    figure_step_times(timings)


if __name__ == "__main__":
    main()
