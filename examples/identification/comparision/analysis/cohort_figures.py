"""
The cohort figures of every route, read off the stored npz and re-solving nothing.

    design             the plan as it was recorded, two families of recordings
    cascade            the five steps, and what each one is allowed to move
    route_redundancy   three measurements, or one quantity written three times
    traces             measured against identified, where the model is wrong
    residual_envelope  the same residual over the whole cohort
    step_times         convergence and numerical cost
    agreement          measured against predicted, peak and time integral
    relationships      the force-frequency and force-intensity relationships
    parameters         does a reconstructed observable find the same model
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import _figures
import _shared

import metrics
import ocp
import protocol
import routes as route_table
import steps
import store
from helper import results

FIGURES_ROOT = store.RESULTS_ROOT / "figures"
ANALYSIS = "cohort_figures"

ROUTES = {route.METHOD: (route.label, route.colour) for route in route_table.ROUTES}
ORDER = _figures.ordered(list(ROUTES))

PANELS = (
    ("plateau_at_reference", r"$A \cdot \tau_1$ (N)", 1.0, False),
    ("a_scale", r"$a'$ (N/s)", 1.0, False),
    ("km_rest", r"$K_m$", 1.0, True),
    ("tau1_rest", r"$\tau_1$ (ms)", 1e3, True),
    ("tau2", r"$\tau_2$ (ms)", 1e3, True),
    ("pd0", r"$pd_0$ ($\mu$s)", 1e6, True),
    ("pdt", r"$pd_t$ ($\mu$s)", 1e6, True),
)

GUESS_COLOUR = _figures.BOUND
CFT_FREQUENCIES = (20, 33, 50)

# One recording per family: the two isometric routes read the same trains, the movement route the flexions
FAMILIES = (("isometric trains", _figures.REFERENCE,
             "sensor_id and muscle_force_isometric_id read these same trains"),
            ("elbow flexions", "movement_id", "movement_id reads these flexions"))

STEPS = ("step1", "step2", "step3", "step4", "step5")
STEP_LABELS = {"step1": r"1  $\tau_1$", "step2": r"2  $A$, $K_m$", "step3": r"3  $A$ per intensity",
               "step4": "4  eq. 3", "step5": "5  prediction"}


# ---- Reading what the routes stored ---- #
def parameter_frame() -> pd.DataFrame:
    """One row per (route, subject, parameter): the value, and whether it sat on a bound."""
    rows = []
    for route in ORDER:
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
    for route in ORDER:
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
    for route in ORDER:
        for subject, result in results.load_method(route).items():
            seconds = result["extra"].get("step_seconds", {})
            if seconds:
                rows.append({"route": route, "subject": subject,
                             **{step: float(seconds.get(step, np.nan)) for step in STEPS},
                             "total": float(sum(seconds.values()))})
    return pd.DataFrame(rows)


def status_frame() -> pd.DataFrame:
    """One row per (route, subject): what the solver said, or the boolean the older npz carry instead."""
    rows = []
    for route in ORDER:
        for subject, result in results.load_method(route).items():
            blocks = result["extra"].get("step_diagnostics", {})
            said = sorted({block["status"] for block in blocks.values()
                           if isinstance(block, dict) and block.get("status")})
            rows.append({"route": route, "subject": subject, "reported": bool(said),
                         "status": "|".join(said) if said else "",
                         "verdict": _verdict_of(said, bool(result["converged"]))})
    return pd.DataFrame(rows)


def _verdict_of(said, converged: bool) -> str:
    """The three levels a solver exit is read at, from its own words when the npz carries them."""
    if not said:
        return "converged (no status stored)" if converged else "failure"
    if any("Acceptable" in status for status in said):
        return "acceptable level"
    if all("Succeeded" in status for status in said):
        return "optimum"
    return "failure"


def design_frame() -> pd.DataFrame:
    """One row per (family, subject, condition): what that train served, or that it was never recorded."""
    rows = []
    for family, route, _ in FAMILIES:
        table = _shared.trains(route)
        if table.empty:
            continue
        recorded = {(subject, condition): kind for subject, condition, kind
                    in zip(table["subject"], table["condition"], table["set"])}
        for subject in sorted(table["subject"].unique()):
            for frequency, row in protocol.DESIGN.items():
                for pulse_width, cell in row.items():
                    condition = protocol.condition_name(frequency, pulse_width)
                    kind = recorded.get((subject, condition))
                    rows.append({"family": family, "route": route, "subject": subject,
                                 "frequency": frequency, "pulse_width_us": pulse_width,
                                 "condition": condition, "planned": cell,
                                 "role": "missing" if kind is None
                                 else "reference" if cell == protocol.REFERENCE else kind})
    return pd.DataFrame(rows)


def _save(figure, name) -> Path:
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
    path = FIGURES_ROOT / f"{name}.svg"
    figure.savefig(path, format="svg", bbox_inches="tight")
    plt.close(figure)
    print(f"  {path}")
    return path


def figure_design(table=None, name="design") -> Path:
    """What each route works on, what the two families share, and what is missing from each."""
    table = design_frame() if table is None else table
    if table.empty:
        raise ValueError(f"No stored result under {results.RESULTS_ROOT}")

    tints = {"identification": 0.35, "reference": 0.35, "testing": 0.85}
    figure, axes_grid = plt.subplots(1, len(FAMILIES), figsize=(15, 6.4))
    conditions = [(frequency, pulse_width)
                  for frequency in protocol.FREQUENCIES for pulse_width in protocol.PULSE_WIDTHS_US]

    for axes, (family, route, shared) in zip(np.atleast_1d(axes_grid), FAMILIES):
        block = table[table["family"] == family]
        subjects = sorted(block["subject"].unique())
        cells = {(row["subject"], row["frequency"], row["pulse_width_us"]): row["role"]
                 for _, row in block.iterrows()}

        for column, (frequency, pulse_width) in enumerate(conditions):
            for row, subject in enumerate(subjects):
                role = cells.get((subject, frequency, pulse_width), "missing")
                if role == "missing":
                    axes.add_patch(Rectangle((column, row), 1, 1, facecolor="white", edgecolor="0.75",
                                             linewidth=0.4))
                    axes.plot([column + 0.15, column + 0.85], [row + 0.15, row + 0.85], color="0.3", lw=0.7)
                    axes.plot([column + 0.15, column + 0.85], [row + 0.85, row + 0.15], color="0.3", lw=0.7)
                    continue
                axes.add_patch(Rectangle((column, row), 1, 1, facecolor=_figures.colour(route),
                                         alpha=tints[role], edgecolor="0.75", linewidth=0.4))
                if role == "reference":
                    axes.add_patch(Rectangle((column + 0.03, row + 0.03), 0.94, 0.94, facecolor="none",
                                             edgecolor="black", linewidth=1.6))

        axes.set_xlim(0, len(conditions))
        axes.set_ylim(0, len(subjects))
        axes.set_xticks(np.arange(len(conditions)) + 0.5)
        axes.set_xticklabels([f"{pulse_width}" for _, pulse_width in conditions], fontsize=6, rotation=90)
        axes.set_yticks(np.arange(len(subjects)) + 0.5)
        axes.set_yticklabels([f"P{subject}" for subject in subjects], fontsize=6)
        for boundary in range(len(protocol.FREQUENCIES) + 1):
            axes.axvline(boundary * len(protocol.PULSE_WIDTHS_US), color="black", lw=1.0)
        for index, frequency in enumerate(protocol.FREQUENCIES):
            axes.text((index + 0.5) * len(protocol.PULSE_WIDTHS_US), len(subjects) + 0.3, f"{frequency} Hz",
                      ha="center", va="bottom", fontsize=8, fontweight="bold")
        axes.set_xlabel(r"pulse width ($\mu$s)", fontsize=8)
        axes.set_title(f"{family}\n{shared}", fontsize=9, fontweight="bold", pad=22)

        missing = block[block["role"] == "missing"]
        lines = [f"P{subject}: " + ", ".join(f"{row['condition']} ({row['planned']})"
                                             for _, row in group.sort_values('condition').iterrows())
                 for subject, group in missing.groupby("subject")]
        consequences = []
        for subject, group in missing.groupby("subject"):
            if (group["planned"] == protocol.TESTING).any():
                kept = len(block[(block["subject"] == subject) & (block["role"] == "testing")])
                consequences.append(f"P{subject} is predicted on {kept} testing phases, not 5")
            if (group["planned"] == protocol.REFERENCE).any():
                kept = len(block[(block["subject"] == subject) & (block["role"] == "reference")])
                consequences.append(f"P{subject} lost a reference train: steps 1 and 2 ran on {kept}, not 3")
        _figures.note(axes, "not recorded:\n  " + ("\n  ".join(lines) if lines else "nothing")
                      + ("\n" + "\n".join(consequences) if consequences else ""),
                      x=0.01, y=-0.14, va="top", size=6)

    figure.legend(handles=[Rectangle((0, 0), 1, 1, facecolor="0.5", alpha=0.35, label="identification"),
                           Rectangle((0, 0), 1, 1, facecolor="0.5", alpha=0.85, label="testing"),
                           Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="black", linewidth=1.6,
                                     label="reference column, 400 us"),
                           Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="0.3", label="not recorded")],
                  loc="upper right", fontsize=7, ncol=4)
    figure.suptitle("The plan as it was recorded, not as it was designed", fontsize=13, fontweight="bold")
    _figures.caption(figure, "The two blocks are not the same recording: an isometric train and a flexion "
                             "are different files, and their holes are different too.")
    figure.tight_layout(rect=(0, 0.14, 1, 0.93))
    return _save(figure, name)



def cascade_free() -> dict:
    """What each step is allowed to move, read off the stored diagnostics rather than typed in twice."""
    free = {route: {"step1": [], "step2": [], "step3": ["A"], "step4": list(steps.STEP34_FREE), "step5": []}
            for route in ORDER}
    for route in ORDER:
        stored = results.load_method(route)
        if not stored:
            continue
        first = next(iter(stored.values()))
        block = first["extra"].get("step_diagnostics", {}).get("step2", {})
        free[route]["step2"] = list(block.get("free", []))
        if first["extra"].get("tau1_policy") == "relaxation":
            free[route]["step1"] = ["tau1_rest"]
    return free


def figure_cascade(name="cascade") -> Path:
    """The five steps, identical on the three routes, and the trains each one is allowed to read."""
    free = cascade_free()
    rows = ("A", "km_rest", "tau1_rest", "tau2", "a_scale", "pd0", "pdt")
    labels = {"A": r"$A$", "km_rest": r"$K_m$", "tau1_rest": r"$\tau_1$", "tau2": r"$\tau_2$",
              "a_scale": r"$a'$", "pd0": r"$pd_0$", "pdt": r"$pd_t$"}
    held = ("tauc", "r0")
    trains = {"step1": f"{_figures.REFERENCE_TRAINS} reference trains, no solver",
              "step2": f"the same {_figures.REFERENCE_TRAINS} reference trains",
              "step3": "10 trains, 2 per intensity", "step4": "the 6 values of A, no train",
              "step5": "5 testing trains, nothing free"}

    reference = free[_figures.REFERENCE]
    identical = [route for route in ORDER if free[route] == reference]
    figure, axes = plt.subplots(figsize=(13, 5.6))

    for index, name_of in enumerate(rows + held):
        y = len(rows + held) - index - 1
        grey = name_of in held
        axes.text(-0.15, y + 0.5, labels.get(name_of, name_of), ha="right", va="center",
                  fontsize=11, color="0.55" if grey else "black")
        for column, step in enumerate(STEPS):
            movable = not grey and name_of in reference[step]
            axes.add_patch(Rectangle((column + 0.1, y + 0.2), 0.8, 0.6,
                                     facecolor="0.30" if movable else "white",
                                     edgecolor="0.85" if grey else "0.55", linewidth=0.8))
        if grey:
            axes.text(len(STEPS) + 0.08, y + 0.5, "held at its measured value throughout",
                      ha="left", va="center", fontsize=7, color="0.55")

    for column, step in enumerate(STEPS):
        axes.text(column + 0.5, len(rows + held) + 0.35, STEP_LABELS[step], ha="center", va="bottom",
                  fontsize=10, fontweight="bold")
        axes.text(column + 0.5, -0.35, trains[step], ha="center", va="top", fontsize=7, color="0.3")
        if column:
            axes.annotate("", xy=(column + 0.08, len(rows + held) + 0.15),
                          xytext=(column - 0.08, len(rows + held) + 0.15),
                          arrowprops=dict(arrowstyle="->", color="0.4", lw=1.0))

    axes.set_xlim(-1.4, len(STEPS) + 1.7)
    axes.set_ylim(-1.5, len(rows + held) + 1.0)
    axes.axis("off")
    axes.text(-1.35, -0.9,
              f"filled: free at that step   empty: held\n"
              f"1. the procedure is the same on {len(identical)} of the {len(ORDER)} routes "
              f"({', '.join(_figures.label(route) for route in identical)}): only the tracked observable "
              f"and the cost change, which is what lets a gap be blamed on the observable.\n"
              f"2. four parameters of seven - {', '.join(_figures.STEP2_ONLY)} - never see more than the "
              f"{_figures.REFERENCE_TRAINS} reference trains. The 10 other identification trains only "
              f"feed {', '.join(_figures.EQ3_ONLY)}.",
              ha="left", va="top", fontsize=7.5)
    figure.suptitle("What the identification does, step by step", fontsize=13, fontweight="bold")
    figure.tight_layout()
    return _save(figure, name)


def _normalised(route: str, subject: str) -> np.ndarray:
    """One participant's tracked recording, scaled by its own largest excursion."""
    _, tracked, _ = _shared.signal_of(route, subject)
    tracked = np.asarray(tracked, float)
    span = float(np.nanmax(np.abs(tracked)))
    return tracked / span if span else tracked


def signal_gap(first: str, second: str, subject: str, points=400) -> float:
    """The largest gap between two normalised recordings, on a common index grid when lengths differ."""
    left, right = _normalised(first, subject), _normalised(second, subject)
    if len(left) != len(right):
        grid = np.linspace(0, 1, points)
        left = np.interp(grid, np.linspace(0, 1, len(left)), left)
        right = np.interp(grid, np.linspace(0, 1, len(right)), right)
    return float(np.nanmax(np.abs(left - right)))


def redundancy_frame(pairs=None) -> pd.DataFrame:
    """The gap between two routes, participant by participant, on the recordings and on the parameters."""
    pairs = pairs or [(first, second) for i, first in enumerate(ORDER) for second in ORDER[i + 1:]]
    rows = []
    for first, second in pairs:
        shared = sorted(set(_shared.subjects_of(first)) & set(_shared.subjects_of(second)))
        for subject in shared:
            rows.append({"first": first, "second": second, "subject": subject,
                         "kind": "signal", "parameter": "tracked", "gap": signal_gap(first, second, subject)})
            left = results.load(subject, first)["parameters"]
            right = results.load(subject, second)["parameters"]
            for parameter in _shared.PARAMETERS:
                if parameter not in left or parameter not in right:
                    continue
                scale = max(abs(left[parameter]), abs(right[parameter]))
                rows.append({"first": first, "second": second, "subject": subject, "kind": "parameter",
                             "parameter": parameter,
                             "gap": abs(left[parameter] - right[parameter]) / scale if scale else np.nan})
    return pd.DataFrame(rows)


def figure_route_redundancy(table=None, counter_example=None, name="route_redundancy") -> Path:
    """
    Whether an agreement between two routes is a result or an algebraic consequence.

    The pair that was dropped for being one quantity written twice is measured again here, and it is
    what gives the scale: a couple falling back to that level is not two measurements.
    """
    table = redundancy_frame() if table is None else table
    counter_example = withdrawn_pair() if counter_example is None else counter_example
    figure, axes_grid = plt.subplots(len(ORDER), 2 * len(ORDER), figsize=(15, 7.5),
                                     gridspec_kw={"wspace": 0.05, "hspace": 0.08})

    for block, (kind, title) in enumerate((("signal", "(a) the recordings, normalised"),
                                           ("parameter", "(b) the identified parameters"))):
        for row, second in enumerate(ORDER):
            for column, first in enumerate(ORDER):
                axes = axes_grid[row][block * len(ORDER) + column]
                axes.set_xticks([])
                axes.set_yscale("log")
                axes.set_ylim(1e-8, 1e0)
                axes.set_xlim(-0.5, 0.5)
                axes.grid(alpha=0.25, axis="y")
                if row == 0:
                    axes.set_title(_figures.label(first), fontsize=7)
                if block == 0 and column == 0:
                    axes.set_ylabel(_figures.label(second), fontsize=7)
                axes.tick_params(labelsize=5.5, labelleft=(block * len(ORDER) + column) == 0)

                if row == column:
                    axes.set_facecolor("0.95")
                    continue
                rows = table[(table["kind"] == kind)
                             & (((table["first"] == first) & (table["second"] == second))
                                | ((table["first"] == second) & (table["second"] == first)))]
                values = rows["gap"].to_numpy(float)
                if values.size:
                    axes.scatter(_figures.jitter(len(values), 0.22), np.clip(values, 1e-8, None), s=7,
                                 color="0.25", alpha=0.7, edgecolor="none")
                    _figures.count(axes, f"n {len(values)}", size=5)
                axes.axhline(1e-6, color=_figures.THRESHOLD, ls="--", lw=0.8)
                if counter_example and np.isfinite(counter_example.get(kind, np.nan)):
                    axes.axhline(counter_example[kind], color="tab:purple", lw=0.8, ls=":")
        axes_grid[0][block * len(ORDER)].text(0.0, 1.18, title, transform=axes_grid[0][block * len(ORDER)]
                                              .transAxes, fontsize=10, fontweight="bold", ha="left")

    told = ("no pair of the withdrawn routes is on disk"
            if not counter_example else
            f"purple dotted: {_figures.label(counter_example['first'])} against "
            f"{_figures.label(counter_example['second'])}, the pair dropped for being one quantity written "
            f"twice - {counter_example['signal']:.1e} on the recordings, "
            f"{counter_example['parameter']:.1e} on the parameters")
    figure.suptitle("Three routes, or one quantity written three times", fontsize=13, fontweight="bold")
    _figures.caption(figure, f"One point per participant (a) and per participant and parameter (b), "
                             f"largest relative gap, log axis over eight decades.\n{told}.\n"
                             f"grey dashed: 1e-6, under which two routes are two spellings of the same "
                             f"number rather than two measurements.")
    figure.tight_layout(rect=(0, 0.10, 1, 0.92))
    return _save(figure, name)


def withdrawn_pair(first="muscle_force_isometric_id", second="torque_isometric_id") -> dict:
    """The gap of the pair that was dropped, measured rather than quoted, when it is still on disk."""
    if not (store.RESULTS_ROOT / second).exists():
        return {}
    table = redundancy_frame([(first, second)])
    if table.empty:
        return {}
    return {"first": first, "second": second,
            "signal": float(table.loc[table["kind"] == "signal", "gap"].max()),
            "parameter": float(table.loc[table["kind"] == "parameter", "gap"].max())}


def testing_phases(route: str, subject: str) -> list:
    """The stored step 5 phases of one participant, each with the condition it was recorded at."""
    result = results.load(subject, route)
    extra = result["extra"]
    conditions = list(zip(extra.get("frequency", []), extra.get("pulse_width", [])))
    return [{"condition": protocol.condition_name(frequency, pulse_width), "frequency": float(frequency),
             "time": time, "tracked": tracked, "identified": identified}
            for (frequency, pulse_width), (time, tracked, identified)
            in zip(conditions, results.phases_of(result))]


def figure_traces(subject=None, name="traces") -> Path:
    """One measured trace against its identified one, route by route, at a low and a high frequency."""
    subject = subject or _figures.illustration_subject()
    figure, axes_grid = plt.subplots(3 * len(ORDER), 2, figsize=(12, 10.5),
                                     height_ratios=[3, 1, 0.75] * len(ORDER),
                                     gridspec_kw={"hspace": 0.22})
    for spacer in axes_grid[2::3].ravel():
        spacer.axis("off")

    for index, route in enumerate(ORDER):
        phases = testing_phases(route, subject)
        if not phases:
            continue
        chosen = [min(phases, key=lambda phase: phase["frequency"]),
                  max(phases, key=lambda phase: phase["frequency"])]
        unit = _figures.unit_of(route)
        for column, phase in enumerate(chosen):
            top, bottom = axes_grid[3 * index][column], axes_grid[3 * index + 1][column]
            bottom.sharex(top)
            time = np.asarray(phase["time"], float) - float(phase["time"][0])
            measured = np.asarray(phase["tracked"], float)
            produced = np.asarray(phase["identified"], float)

            top.plot(time, measured, color="0.55", lw=2.2, label="measured")
            top.plot(time, produced, color=_figures.colour(route), lw=1.1, label="identified")
            bottom.fill_between(time, produced - measured, color=_figures.colour(route), alpha=0.35)
            bottom.axhline(0.0, color="0.5", lw=0.7)

            top.set_ylabel(f"{_figures.label(route)}\n({unit})", fontsize=8)
            bottom.set_ylabel(f"residual ({unit})", fontsize=6)
            top.set_title(f"{phase['condition']}", fontsize=9, fontweight="bold")
            top.tick_params(labelsize=6, labelbottom=False)
            bottom.tick_params(labelsize=6)
            bottom.set_xlabel("time since the start of the train (s)", fontsize=7)
            top.grid(alpha=0.25)
            bottom.grid(alpha=0.25)
            if index == 0 and column == 0:
                top.legend(fontsize=6, loc="upper right")
            _figures.note(top, f"RMSE {float(results.rmse(measured, produced)):.3g} {unit}", size=6)

    figure.suptitle(f"Measured against identified, participant P{subject}", fontsize=13, fontweight="bold")
    _figures.caption(figure, f"P{subject} is not a hand-picked participant: it is the one whose NRMSE is "
                             f"nearest the cohort median on {_figures.REFERENCE}, and the same participant "
                             f"is used in every illustration of this study.\nOne testing train per "
                             f"cell, one unit per row.")
    figure.tight_layout(rect=(0, 0.06, 1, 0.95))
    return _save(figure, name)


GRID = np.linspace(0.0, 1.0, 101)


def residual_frame(route: str) -> np.ndarray:
    """Every testing phase of one route on one normalised time base, so a median has a meaning."""
    stacked = []
    for subject in _shared.subjects_of(route):
        for phase in testing_phases(route, subject):
            time = np.asarray(phase["time"], float)
            if len(time) < 3 or time[-1] <= time[0]:
                continue
            span = (time - time[0]) / (time[-1] - time[0])
            residual = np.asarray(phase["identified"], float) - np.asarray(phase["tracked"], float)
            stacked.append(np.interp(GRID, span, residual))
    return np.vstack(stacked) if stacked else np.empty((0, len(GRID)))


def figure_residual_envelope(name="residual_envelope") -> Path:
    """Whether the error of one trace is that participant's, or the model's on the whole cohort."""
    figure, axes_grid = plt.subplots(1, len(ORDER), figsize=(15, 4.4))

    for axes, route in zip(np.atleast_1d(axes_grid), ORDER):
        stacked = residual_frame(route)
        if not stacked.size:
            axes.axis("off")
            continue
        for trace in stacked:
            axes.plot(GRID, trace, color=_figures.colour(route), lw=0.6, alpha=0.06, zorder=1)
        low, first, middle, third, high = np.percentile(stacked, [5, 25, 50, 75, 95], axis=0)
        axes.fill_between(GRID, low, high, color=_figures.colour(route), alpha=0.18, label="5-95 %")
        axes.fill_between(GRID, first, third, color=_figures.colour(route), alpha=0.38, label="quartiles")
        axes.plot(GRID, middle, color=_figures.colour(route), lw=1.6, label="median")
        axes.axhline(0.0, color="0.5", lw=0.8)

        axes.set_title(_figures.label(route), fontsize=10, fontweight="bold")
        axes.set_xlabel("time within the phase, normalised", fontsize=8)
        axes.set_ylabel(f"identified - measured ({_figures.unit_of(route)})", fontsize=8)
        axes.tick_params(labelsize=7)
        axes.grid(alpha=0.25)
        axes.legend(fontsize=6, loc="upper right")
        _figures.count(axes, f"{len(_shared.subjects_of(route))} participants, {len(stacked)} phases")

    figure.suptitle("The residual over the whole cohort, phase by phase", fontsize=13, fontweight="bold")
    _figures.caption(figure, f"Resampling: every testing phase is put on {len(GRID)} points of its own "
                             f"normalised duration by linear interpolation, then the percentiles are taken "
                             f"across phases. Phases are of unequal length, so an unresampled average "
                             f"would not exist. One unit per panel, they do not compare.")
    figure.tight_layout(rect=(0, 0.08, 1, 0.93))
    return _save(figure, name)


def figure_step_times(table=None, statuses=None, name="step_times") -> Path:
    """What each step costs, and what the solver said it reached, which decides whether the cost counts."""
    table = timing_frame() if table is None else table
    statuses = status_frame() if statuses is None else statuses
    figure, axes_grid = plt.subplots(1, len(STEPS) + 1, figsize=(19, 4.4),
                                     gridspec_kw={"width_ratios": [1] * len(STEPS) + [1.5]})

    for position, (axes, step) in enumerate(zip(axes_grid[:len(STEPS)], STEPS)):
        if position:
            axes.sharey(axes_grid[0])
        for index, route in enumerate(ORDER):
            values = table.loc[table["route"] == route, step].dropna().to_numpy(float)
            if not values.size:
                continue
            axes.boxplot([values], positions=[index], widths=0.55, showfliers=False,
                         medianprops=dict(color="black", linewidth=1.2))
            _figures.points(axes, index, values, route, size=10)
        axes.set_yscale("log")
        axes.set_title(STEP_LABELS[step], fontsize=10, fontweight="bold")
        _figures.route_ticks(axes, ORDER)
        axes.grid(alpha=0.3, axis="y")
        axes.tick_params(axis="y", labelsize=7)
    axes_grid[0].set_ylabel("seconds")

    axes = axes_grid[len(STEPS)]
    levels = ["optimum", "acceptable level", "converged (no status stored)", "failure"]
    tints = {"optimum": "tab:green", "acceptable level": "tab:orange",
             "converged (no status stored)": "0.6", "failure": "tab:red"}
    drawn = {}
    for index, route in enumerate(ORDER):
        block = statuses[statuses["route"] == route]
        bottom = 0
        for level in levels:
            height = int((block["verdict"] == level).sum())
            if not height:
                continue
            drawn[level] = axes.bar(index, height, bottom=bottom, width=0.6, color=tints[level],
                                    label=level)
            bottom += height
    axes.set_title("what the solver reached", fontsize=10, fontweight="bold")
    axes.set_ylim(0, 1.7 * max(len(statuses[statuses["route"] == route]) for route in ORDER))
    axes.set_ylabel("participants", fontsize=8)
    _figures.route_ticks(axes, ORDER)
    axes.legend(handles=[drawn[level] for level in levels if level in drawn], fontsize=6,
                loc="lower right")
    axes.tick_params(axis="y", labelsize=7)
    if not statuses["reported"].any():
        _figures.note(axes, "no npz on disk carries the IPOPT status yet, so this bar only says\n"
                            "that every step converged; a Solved_To_Acceptable_Level cannot be\n"
                            "told from an optimum until the routes are relaunched", size=6)

    figure.suptitle("What each step costs, and what it converged to", fontsize=13, fontweight="bold")
    totals = ", ".join(
        f"{_figures.label(route)} {table.loc[table['route'] == route, 'total'].median():.0f} s"
        for route in ORDER if (table["route"] == route).any())
    _figures.caption(figure, f"Median total per route: {totals}.\n"
                             f"This is solver time, not session time: recording, rest between trains and "
                             f"setting up the participant are in none of these numbers.\n"
                             f"A Solved_To_Acceptable_Level read as an optimum would invalidate every "
                             f"conclusion drawn from the cost.")
    figure.tight_layout(rect=(0, 0.08, 1, 0.93))
    return _save(figure, name)


def _agreement_text(measured, predicted) -> str:
    """The agreement block of one panel, as the article reports it."""
    block = metrics.agreement(measured, predicted)
    if not np.isfinite(block["icc"]):
        return "\n".join([f"n = {block['n']}", "too few to score"])
    return "\n".join([
        f"ICC {block['icc']:.2f} [{block['icc_ci'][0]:.2f}, {block['icc_ci'][1]:.2f}] {_shared.grade_icc(block['icc'])}",
        f"slope (zero intercept) {block['slope']:.2f}",
        f"%model error {block['model_error']:.0f} +/- {block['model_error_sd']:.0f}",
        f"paired p {block['p']:.3f}   n {block['n']}",
    ])


def figure_agreement(table=None, name="agreement") -> Path:
    """Measured against predicted on the trains no identification saw, the reference route first."""
    table = score_frame() if table is None else table
    figure, axes_grid = plt.subplots(2, len(ORDER), figsize=(16, 9), squeeze=False)

    for column, route in enumerate(ORDER):
        rows = table[table["route"] == route]
        unit = rows["unit"].iloc[0] if not rows.empty else _figures.unit_of(route)
        for row, (measure, name_of) in enumerate((("peak", f"PK ({unit})"), ("fti", f"FTI ({unit} s)"))):
            axes = axes_grid[row][column]
            measured = rows[f"{measure}_measured"].to_numpy(dtype=float)
            predicted = rows[f"{measure}_predicted"].to_numpy(dtype=float)

            for _, group in rows.groupby("subject"):
                ordered_group = group.sort_values(f"{measure}_measured")
                axes.plot(ordered_group[f"{measure}_measured"], ordered_group[f"{measure}_predicted"],
                          color="0.6", lw=0.5, alpha=0.6, zorder=1)
            axes.scatter(measured, predicted, s=12, color=_figures.colour(route), alpha=0.75, zorder=2,
                         edgecolor="none")
            if measured.size:
                _figures.bisector(axes, np.concatenate([measured, predicted]))
                _figures.note(axes, _agreement_text(measured, predicted), size=6)
                _figures.count(axes, f"{len(measured)} testing trains, "
                                     f"{rows['subject'].nunique()} participants")
            axes.set_xlabel(f"measured {name_of}", fontsize=8)
            axes.set_ylabel(f"predicted {name_of}", fontsize=8)
            if row == 0:
                axes.set_title(_figures.label(route), fontsize=10, fontweight="bold")
            axes.tick_params(labelsize=7)
            axes.grid(alpha=0.3)

    figure.suptitle("Measured against predicted, every testing train of every participant",
                    fontsize=13, fontweight="bold")
    _figures.caption(figure, "The thin lines join the 5 testing trains of one participant: the points are "
                             "not independent, and a cloud of 98 of them flatters the agreement.\n"
                             "Grey dashed: y = x. Axes are square and share their limits.")
    figure.tight_layout(rect=(0, 0.06, 1, 0.94))
    return _save(figure, name)


def figure_relationships(table=None, name="relationships") -> Path:
    """The force-frequency and force-intensity relationships, measured and predicted, route by route."""
    table = score_frame(testing_only=False) if table is None else table
    figure, axes_grid = plt.subplots(2, len(ORDER), figsize=(16, 8), squeeze=False)
    reference_us = protocol.REFERENCE_PULSE_WIDTH_US

    for column, route in enumerate(ORDER):
        rows = table[table["route"] == route].copy()
        if not rows.empty:
            largest = rows.groupby("subject")["peak_measured"].transform("max")
            rows["measured"] = rows["peak_measured"] / largest
            rows["predicted"] = rows["peak_predicted"] / largest

        slices = ((rows[rows["pulse_width_us"].round() == reference_us] if not rows.empty else rows,
                   "frequency", "Frequency (Hz)", f"at {reference_us} us"),
                  (rows[rows["frequency"].round() == protocol.FREQUENCIES[0]] if not rows.empty else rows,
                   "pulse_width_us", r"Pulse width ($\mu$s)", f"at {protocol.FREQUENCIES[0]} Hz"))

        for row, (block, by, label, where) in enumerate(slices):
            axes = axes_grid[row][column]
            if not block.empty:
                for _, group in block.groupby("subject"):
                    group = group.sort_values(by)
                    axes.plot(group[by], group["measured"], color="0.4", lw=0.6, alpha=0.15, zorder=1)
                for side, marker, fill, colour_of in (("measured", "o", True, "black"),
                                                      ("predicted", "D", False, _figures.colour(route))):
                    grouped = block.groupby(by)[side].agg(["mean", "std", "count"])
                    axes.errorbar(grouped.index, grouped["mean"], yerr=grouped["std"], marker=marker,
                                  markersize=5, capsize=3, linewidth=1, color=colour_of, zorder=3,
                                  markerfacecolor=colour_of if fill else "none", label=side)
                    if side == "measured":
                        for x, entry in grouped.iterrows():
                            axes.annotate(f"n {int(entry['count'])}", (x, entry["mean"]),
                                          textcoords="offset points", xytext=(0, -14), ha="center",
                                          fontsize=5.5, color="0.35")
                held_out = int((block["set"] == "testing").sum())
                _figures.count(axes, f"{len(block)} trains, {held_out} of them testing")
            axes.set_xlabel(f"{label}, {where}", fontsize=8)
            axes.set_ylabel("peak / that participant's largest measured peak", fontsize=7)
            if row == 0:
                axes.set_title(_figures.label(route), fontsize=10, fontweight="bold")
            if row == 0 and column == 0:
                axes.legend(fontsize=7)
            axes.tick_params(labelsize=7)
            axes.grid(alpha=0.3)

    figure.suptitle("The force-frequency and force-intensity relationships", fontsize=13,
                    fontweight="bold")
    _figures.caption(figure, "Filled: measured. Hollow and joined: predicted. Mean +/- SD, individual "
                             "trajectories behind at alpha 0.15.\nBoth slices hold identification trains "
                             "as well as testing ones: the 400 us column is the reference column, which "
                             "the testing set never contains.")
    figure.tight_layout(rect=(0, 0.06, 1, 0.94))
    return _save(figure, name)


def _log_slope(x, y) -> float:
    """The slope of the cloud in log space, which is what a scale factor between routes looks like."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    usable = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if usable.sum() < 3:
        return np.nan
    return float(np.polyfit(np.log10(x[usable]), np.log10(y[usable]), 1)[0])


def figure_parameters(table=None, name="parameters") -> Path:
    """The reference route against each candidate, parameter by parameter, one point per participant."""
    table = parameter_frame() if table is None else table
    if table.empty:
        raise ValueError(f"No stored result under {results.RESULTS_ROOT}")

    candidates = [route for route in ORDER if route != _figures.REFERENCE]
    markers = dict(zip(candidates, ("o", "^", "s", "D")))
    figure, axes_grid = plt.subplots(2, 4, figsize=(17, 8.6))

    for axes, (parameter, label, scale, _) in zip(axes_grid.ravel(), PANELS):
        drawn, ratios = [], {}
        reference = table[(table["route"] == _figures.REFERENCE) & (table["parameter"] == parameter)]
        reference = reference.set_index("subject")
        for candidate in candidates:
            other = table[(table["route"] == candidate) & (table["parameter"] == parameter)]
            other = other.set_index("subject")
            shared = [subject for subject in reference.index if subject in other.index]
            x = scale * reference.loc[shared, "value"].to_numpy(float)
            y = scale * other.loc[shared, "value"].to_numpy(float)
            bounded = (reference.loc[shared, "at_bound"].to_numpy(bool)
                       | other.loc[shared, "at_bound"].to_numpy(bool))
            for point_x, point_y, bound in zip(x, y, bounded):
                axes.scatter(point_x, point_y, s=22, marker=markers[candidate], zorder=3,
                             facecolor="none" if bound else _figures.colour(candidate),
                             edgecolor=_figures.BOUND if bound else _figures.colour(candidate),
                             linewidths=1.2 if bound else 0.6)
            drawn += [x, y]
            slope = _log_slope(x, y)
            ratios[candidate] = float(np.median(y / x)) if len(x) and np.all(x > 0) else np.nan
            if np.isfinite(slope) and abs(slope - 1) > 0.1:
                grid = np.array([np.nanmin(x), np.nanmax(x)], float)
                intercept = np.nanmedian(np.log10(y) - slope * np.log10(x))
                axes.plot(grid, 10 ** (intercept + slope * np.log10(grid)), ls=":", lw=1.0,
                          color=_figures.colour(candidate))

        _figures.log_bisector(axes, np.concatenate(drawn) if drawn else np.array([1.0]))
        if parameter == "plateau_at_reference":
            largest = float(np.nanmax(np.concatenate(drawn))) if drawn else 0.0
            near = largest > ocp.FMAX / 10
            if near:
                axes.axhline(ocp.FMAX, color=_figures.BOUND, lw=1.0)
                axes.axvline(ocp.FMAX, color=_figures.BOUND, lw=1.0)
            _figures.note(axes, (f"red: the {ocp.FMAX:.0f} N ceiling of the F state, on it a participant "
                                 f"is saturated rather than strong" if near else
                                 f"the {ocp.FMAX:.0f} N ceiling of the F state is off scale: the largest "
                                 f"plateau is {largest:.0f} N, so nobody is saturated")
                          + ".\nThe one amplitude that is in N on all three routes.", size=5.5)
        if parameter == "a_scale":
            _figures.note(axes, "read this panel as a slope, not as an agreement:\n"
                                + "\n".join(f"  x{ratios[candidate]:.1f} against the reference "
                                            f"({_figures.label(candidate)})"
                                            for candidate in candidates if np.isfinite(ratios[candidate]))
                                + "\na point on the bisector here would be suspect", size=5.5)
        axes.set_title(label, fontsize=11, fontweight="bold")
        axes.set_xlabel(f"{_figures.label(_figures.REFERENCE)} (reference)", fontsize=7)
        axes.set_ylabel("candidate route", fontsize=7)
        axes.tick_params(which="major", labelsize=6.5)
        axes.tick_params(which="minor", labelsize=5)
        axes.grid(alpha=0.3)

    handles = [plt.Line2D([], [], ls="none", marker=markers[candidate], color=_figures.colour(candidate),
                          label=_figures.label(candidate)) for candidate in candidates]
    hidden = axes_grid.ravel()[len(PANELS)]
    hidden.axis("off")
    hidden.legend(handles=handles, loc="upper left", fontsize=8, frameon=False)
    hidden.text(0.0, 0.72, "hollow point: the value sat on a bound on one of the two routes\n"
                           "dotted line: the cloud's slope in log space, drawn when it is not 1\n"
                           "dashed line: y = x\n\n"
                           + _figures.ASYMMETRY,
                transform=hidden.transAxes, va="top", fontsize=7.5)

    figure.suptitle("Do two observables find the same model, participant by participant",
                    fontsize=14, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(figure, name)


def main():
    """Draw every figure. Nothing here re-solves, so it is safe to run after any change."""
    _shared.headless()
    print(f"reading {results.RESULTS_ROOT}")
    parameters, scores, timings = parameter_frame(), score_frame(), timing_frame()

    figure_design()
    figure_cascade()
    gaps = redundancy_frame()
    figure_route_redundancy(gaps)
    figure_traces()
    figure_residual_envelope()
    figure_step_times(timings)
    figure_agreement(scores)
    figure_relationships(score_frame(testing_only=False))
    figure_parameters(parameters)

    _shared.write(timings, ANALYSIS, "step_times")
    _shared.write(gaps, ANALYSIS, "route_redundancy")


if __name__ == "__main__":
    main()
