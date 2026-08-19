"""
The cohort figures of the two routes, built from what `isometric.py` and `movement.py` stored.

Every figure is written as svg under `figures/`, and every one of them reads `results/<method>/*.npz` only,
so a partial cohort draws whatever is on disk rather than failing.

    figure 1  every identified parameter, isometric against movement, paired per participant
    figure 2  how well each parameter set predicts the testing trains, including the two swaps
    figure 3  the identification pipeline itself, the two routes side by side
    figure 4  the article's figure 4, measured against predicted with the trendline and the ICC
    figure 5  the article's figure 5, the force-time integral against frequency, per pulse duration
    figure 6  the article's figure 6, the peak against frequency, per pulse duration

Figure 2 needs the cross prediction, which is a solve rather than a lookup: each route's parameter set is
run through the *other* route's step 5. `cross_predict` computes it once per subject and caches it, since
the movement side is one ocp per subject.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

COMPARISON_ROOT = Path(__file__).resolve().parent.parent
for _root in (Path(__file__).resolve().parent, COMPARISON_ROOT, COMPARISON_ROOT / "optimization"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

import protocol  # noqa: E402
from helper import results  # noqa: E402

FIGURES_ROOT = Path(__file__).resolve().parent / "figures"
CACHE = Path(__file__).resolve().parent / "tables" / "ding_cross_prediction.csv"

ISOMETRIC, MOVEMENT = "ding_force_id", "ding_movement_id"
ROUTE_LABEL = {ISOMETRIC: "isometric", MOVEMENT: "movement"}

# P01 lightest, P20 darkest, so a participant keeps its shade across every figure
SHADE_RANGE = (0.20, 0.88)

ISOMETRIC_COLOR = "tab:blue"
MOVEMENT_COLOR = "tab:red"

# The parameters worth a panel, with how they are written and what they are scaled by
PANELS = (
    ("plateau_at_reference", r"$A \cdot \tau_1$ (N)", 1.0),
    ("a_scale", r"$a'$ (N/s)", 1.0),
    ("km_rest", r"$K_m$", 1.0),
    ("tau1_rest", r"$\tau_1$ (ms)", 1e3),
    ("tau2", r"$\tau_2$ (ms)", 1e3),
    ("pd0", r"$pd_0$ ($\mu$s)", 1e6),
    ("pdt", r"$pd_t$ ($\mu$s)", 1e6),
)


# ---- Reading what the two routes stored ---- #
def load_route(method) -> dict:
    """Every stored result of one route, keyed by subject."""
    return results.load_method(method)


def parameter_frame() -> pd.DataFrame:
    """
    One row per (subject, parameter), with the value each route identified and whether it sat on a bound.

    Returns
    -------
    pd.DataFrame
        Columns: subject, parameter, isometric, movement, isometric_at_bound, movement_at_bound
    """
    stored = {method: load_route(method) for method in (ISOMETRIC, MOVEMENT)}
    shared = sorted(set(stored[ISOMETRIC]) & set(stored[MOVEMENT]))

    rows = []
    for subject in shared:
        values, bounded = {}, {}
        for method in (ISOMETRIC, MOVEMENT):
            result = stored[method][subject]
            values[method] = dict(result["parameters"])
            values[method]["plateau_at_reference"] = result["extra"].get("plateau_at_reference", np.nan)
            bounded[method] = set(result["parameters_at_bound"])

        for name, _, _ in PANELS:
            rows.append(
                {
                    "subject": subject,
                    "parameter": name,
                    "isometric": float(values[ISOMETRIC].get(name, np.nan)),
                    "movement": float(values[MOVEMENT].get(name, np.nan)),
                    "isometric_at_bound": name in bounded[ISOMETRIC],
                    "movement_at_bound": name in bounded[MOVEMENT],
                }
            )
    return pd.DataFrame(rows)


def train_frame(method) -> pd.DataFrame:
    """The per train scores of one route, every subject stacked."""
    frames = []
    for subject, result in load_route(method).items():
        table = pd.DataFrame(result["extra"]["trains"])
        table["subject"] = subject
        frames.append(table)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def shades(subjects) -> dict:
    """A grey per participant, lightest for the first, darkest for the last."""
    levels = np.linspace(*SHADE_RANGE, max(len(subjects), 2))
    return {subject: plt.cm.Greys(level) for subject, level in zip(subjects, levels)}


def shade_bar(figure, subjects, ax=None, orientation="vertical"):
    """
    The participant scale itself, drawn so a grey on the figure can be read back to a participant.

    Without it the gradient carries no information: the reader sees that the points differ in shade and
    cannot tell which is which. Ticked at the first, middle and last participant rather than at all of
    them, which would be unreadable at twenty.
    """
    if not len(subjects):
        return None
    colours = [plt.cm.Greys(level) for level in np.linspace(*SHADE_RANGE, max(len(subjects), 2))]
    scale = ScalarMappable(norm=Normalize(0, max(len(subjects) - 1, 1)),
                           cmap=LinearSegmentedColormap.from_list("participants", colours,
                                                                  N=max(len(subjects), 2)))
    bar = figure.colorbar(scale, ax=ax, orientation=orientation, fraction=0.02, pad=0.02)
    positions = sorted({0, (len(subjects) - 1) // 2, len(subjects) - 1})
    bar.set_ticks(positions)
    bar.set_ticklabels([f"P{subjects[i]}" for i in positions])
    bar.ax.tick_params(labelsize=8)
    bar.set_label("participant", fontsize=9)
    return bar


def _save(figure, name) -> Path:
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
    path = FIGURES_ROOT / f"{name}.svg"
    figure.savefig(path, format="svg", bbox_inches="tight")
    plt.close(figure)
    print(f"  {path}")
    return path


# ---- Figure 1: the same parameter through the two observables ---- #
def figure_1(table=None, name="figure1_parameters_paired") -> Path:
    """
    Every identified parameter, isometric on the left of its panel and movement on the right.

    One box per route, the participants drawn as paired points joined by a thin line, and the two means
    joined by a thicker dashed one. The pairing is the point: a parameter can agree on average and still
    disagree for every participant, and only the lines show that. A participant keeps its grey across the
    whole figure, lightest for the first, darkest for the last.

    Parameters
    ----------
    table: pd.DataFrame
        As `parameter_frame` builds it. Read from `results/` when not given.
    """
    table = parameter_frame() if table is None else table
    if table.empty:
        print("figure 1: no subject identified through both routes")
        return None

    subjects = sorted(table["subject"].unique())
    grey = shades(subjects)

    columns = min(4, len(PANELS))
    rows = int(np.ceil(len(PANELS) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(3.6 * columns, 3.9 * rows), squeeze=False)

    for index, (parameter, label, scale) in enumerate(PANELS):
        ax = axes[index // columns][index % columns]
        part = table[table["parameter"] == parameter]
        left = scale * part["isometric"].to_numpy(dtype=float)
        right = scale * part["movement"].to_numpy(dtype=float)
        paired = np.isfinite(left) & np.isfinite(right)

        if not paired.any():
            ax.set_axis_off()
            continue

        ax.boxplot([left[paired], right[paired]], positions=[0, 1], widths=0.45, showfliers=False,
                   medianprops={"color": "black", "linewidth": 1.4},
                   boxprops={"color": "0.35"}, whiskerprops={"color": "0.35"}, capprops={"color": "0.35"})

        # --- One thin line per participant, so a shared trend is visible under the two boxes --- #
        for subject, x_value, y_value, on_bound in zip(
            part["subject"], left, right,
            part["isometric_at_bound"].to_numpy() | part["movement_at_bound"].to_numpy(),
        ):
            if not (np.isfinite(x_value) and np.isfinite(y_value)):
                continue
            ax.plot([0, 1], [x_value, y_value], linewidth=0.8, linestyle=":", color=grey[subject], zorder=2)
            # A parameter that sat on a bound is a limit, not a measurement: hollow it out
            ax.plot([0, 1], [x_value, y_value], "o", markersize=5, color=grey[subject], zorder=3,
                    markerfacecolor="none" if on_bound else grey[subject], markeredgecolor=grey[subject])

        means = (float(np.mean(left[paired])), float(np.mean(right[paired])))
        # Hollow, so a participant sitting exactly under it is still visible
        ax.plot([0, 1], means, linewidth=2.6, linestyle="--", color="black", zorder=5,
                marker="D", markersize=9, markerfacecolor="none", markeredgewidth=1.8, label="mean")

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["isometric", "movement"])
        ax.set_ylabel(label)
        ax.set_xlim(-0.45, 1.45)
        ax.grid(True, axis="y", alpha=0.25)
        ratio = means[1] / means[0] if means[0] else np.nan
        ax.set_title(f"{label}   movement / isometric = {ratio:.2f}", fontsize=10)

    for index in range(len(PANELS), rows * columns):
        axes[index // columns][index % columns].set_axis_off()

    handles = [
        Line2D([], [], color="black", linestyle="--", linewidth=2.6, marker="D", label="cohort mean"),
        Line2D([], [], color="0.5", linestyle=":", marker="o", label="one participant"),
        Line2D([], [], color="0.5", linestyle="none", marker="o", markerfacecolor="none",
               label="on a bound (a limit, not a measurement)"),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, frameon=False)
    figure.suptitle(f"Identified parameters, {len(subjects)} participant(s), the two routes paired",
                    fontsize=14, fontweight="bold")
    figure.tight_layout(rect=(0, 0.04, 1, 0.97))
    shade_bar(figure, subjects, ax=axes.ravel().tolist())
    return _save(figure, name)


# ---- The cross prediction figure 2 stands on ---- #
def _own_route_scores(method) -> list:
    """What each route already stored for its own testing trains, as figure 2 wants them."""
    rows = []
    for subject, result in load_route(method).items():
        table = pd.DataFrame(result["extra"]["trains"])
        testing = table[table["set"] == "testing"]
        rows.append({"subject": subject, "parameters_from": method, "predicts": method,
                     "r2": float(testing["r2"].median()),
                     "peak_error": float(testing["peak_error"].mean()),
                     "fti_error": float(testing["fti_error"].mean())})
    return rows


def cross_predict(subjects=None, max_iter=200, write=True) -> pd.DataFrame:
    """
    Run each route's parameter set through the other route's step 5, and score it there.

    The own-route halves are read from what the two routes already stored; only the two swaps are computed.
    The isometric swap is a forward integration and costs nothing; the movement swap is one ocp per subject,
    which is why the result is cached to `tables/ding_cross_prediction.csv`.

    R2 is summarised per subject by its **median** over the testing trains, not its mean: it is unbounded
    below, so one badly predicted train would otherwise decide the subject's whole value.

    Parameters
    ----------
    subjects: list[str]
        Only these, ex: ["01", "02"]. Every subject present in both routes otherwise.
    max_iter: int
        The solver cap of the movement swap
    write: bool
        Whether the table is cached to disk for `figure_2`
    """
    import isometric  # local: they pull in biorbd and cocofest, which figure 1 and 3 do not need
    import movement
    import force_ocp

    stored = {method: load_route(method) for method in (ISOMETRIC, MOVEMENT)}
    shared = sorted(set(stored[ISOMETRIC]) & set(stored[MOVEMENT]))
    shared = [s for s in shared if subjects is None or s in subjects]
    if not shared:
        print("cross prediction: no subject identified through both routes")
        return pd.DataFrame()

    rows = _own_route_scores(ISOMETRIC) + _own_route_scores(MOVEMENT)
    for subject in shared:
        parameters = {method: dict(stored[method][subject]["parameters"]) for method in stored}
        print(f"P{subject}: cross predicting")

        # --- movement parameters on the isometric testing trains --- #
        try:
            data = isometric.usable(force_ocp.load_force_data(subject, data_folder="force"), subject)
            _, testing_index = protocol.split(data["frequency"], data["pulse_width"])
            scored = isometric.score(force_ocp.select_trains(data, testing_index),
                                     parameters[MOVEMENT], "testing")
            table = scored["table"]
            rows.append({"subject": subject, "parameters_from": MOVEMENT, "predicts": ISOMETRIC,
                         "r2": float(table["r2"].median()),
                         "peak_error": float(table["peak_error"].mean()),
                         "fti_error": float(table["fti_error"].mean())})
        except Exception as error:
            print(f"  movement -> isometric failed: {type(error).__name__}: {error}")

        # --- isometric parameters on the movement testing trains --- #
        try:
            phases = movement.load_movements(subject, debug=False)
            _, testing_index = protocol.split(*movement.conditions_of(phases))
            testing = [phases[i] for i in testing_index]
            step = movement.solve(subject, testing, free=(),
                                  theta_bounds=movement.theta_bounds_of(phases),
                                  model_class=movement.model.DingModelArticle,
                                  constants=parameters[ISOMETRIC],
                                  identify_passive_stop=False, max_iter=max_iter)
            table, _ = movement.score(step, testing, "testing")
            rows.append({"subject": subject, "parameters_from": ISOMETRIC, "predicts": MOVEMENT,
                         "r2": float(table["r2"].median()),
                         "peak_error": float(table["peak_error"].mean()),
                         "fti_error": float(table["fti_error"].mean())})
        except Exception as error:
            print(f"  isometric -> movement failed: {type(error).__name__}: {error}")

    table = pd.DataFrame(rows)
    if write and not table.empty:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(CACHE, index=False)
        print(f"  {CACHE}")
    return table


# ---- Figure 2: what each parameter set predicts, including the swaps ---- #
def figure_2(table=None, measures=("r2", "peak_error", "fti_error"), name="figure2_prediction_swap") -> Path:
    """
    How well each parameter set predicts the testing trains, its own route's and the other's.

    Four groups: each route predicting its own testing trains, and each route's parameters run through the
    other route's step 5. The swap is the question the study asks - if the movement parameters predict the
    isometric trains as well as the isometric ones do, the two routes identified the same muscle.

    Boxplot with every participant drawn as a jittered point over it, which is what data-to-viz recommends
    at this sample size: a violin needs about thirty per group to be trustworthy, and a bare box hides how
    few points it stands on. One panel per measure, since they answer the same question in three units and
    reading them apart invites reading only one.

    Parameters
    ----------
    table: pd.DataFrame
        As `cross_predict` builds it, with columns subject, parameters_from, predicts, and the measures
    measures: tuple
        Which columns get a panel, ex: ("r2", "peak_error", "fti_error")
    """
    table = pd.read_csv(CACHE) if table is None and CACHE.exists() else table
    if table is None or table.empty:
        print(f"figure 2: no cross prediction on disk, run `python figures.py --cross` first ({CACHE})")
        return None

    groups = [
        (ISOMETRIC, ISOMETRIC, "isometric\n$\\rightarrow$ isometric", ISOMETRIC_COLOR),
        (MOVEMENT, ISOMETRIC, "movement\n$\\rightarrow$ isometric", MOVEMENT_COLOR),
        (MOVEMENT, MOVEMENT, "movement\n$\\rightarrow$ movement", MOVEMENT_COLOR),
        (ISOMETRIC, MOVEMENT, "isometric\n$\\rightarrow$ movement", ISOMETRIC_COLOR),
    ]
    subjects = sorted(table["subject"].astype(str).str.zfill(2).unique())
    grey = shades(subjects)
    pretty = {"r2": "$R^2$ on the testing trains", "peak_error": "peak error (%)",
              "fti_error": "integral error (%)"}

    figure, axes = plt.subplots(1, len(measures), figsize=(5.4 * len(measures), 6.2), squeeze=False)
    rng = np.random.default_rng(0)

    for column, measure in enumerate(measures):
        ax = axes[0][column]
        for position, (source, target, _, _) in enumerate(groups):
            part = table[(table["parameters_from"] == source) & (table["predicts"] == target)]
            values = part[measure].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            if len(finite):
                ax.boxplot([finite], positions=[position], widths=0.5, showfliers=False,
                           medianprops={"color": "black", "linewidth": 1.6},
                           boxprops={"color": "0.35"}, whiskerprops={"color": "0.35"},
                           capprops={"color": "0.35"})

            for subject, value in zip(part["subject"].astype(str).str.zfill(2), part[measure]):
                if not np.isfinite(value):
                    continue
                ax.plot(position + rng.uniform(-0.17, 0.17), value, "o", markersize=6,
                        color=grey.get(subject, "0.5"), markeredgecolor="0.25", markeredgewidth=0.5,
                        zorder=3)

        # The two swaps sit beside the route they are compared against
        ax.axvline(1.5, color="0.7", linewidth=1)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([label for _, _, label, _ in groups], fontsize=9)
        ax.set_xlabel("parameters identified on  $\\rightarrow$  trains predicted", fontsize=9)
        for tick, (_, _, _, colour) in zip(ax.get_xticklabels(), groups):
            tick.set_color(colour)
        ax.set_ylabel(pretty.get(measure, measure))
        ax.set_title(pretty.get(measure, measure), fontsize=11)
        ax.grid(True, axis="y", alpha=0.25)
        if measure == "r2":
            ax.axhline(0.0, color="0.4", linestyle=":", linewidth=1)

    figure.suptitle("Predicting the testing trains with each route's parameters, and with the other's\n"
                    f"{len(subjects)} participant(s), one point per participant",
                    fontsize=13, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    shade_bar(figure, subjects, ax=axes.ravel().tolist())
    return _save(figure, name)


# ---- The article's own figures 4, 5 and 6 ---- #
# What a peak and an integral are in each route's own units. The article reports force, so the isometric
# route is read straight off; the flexion measures an angle, and its counterparts are the excursion and the
# angle-time integral above the starting angle, see the scoring in each route.
MEASURES = {
    ISOMETRIC: {"peak": ("PK", "N", 1.0), "fti": ("FTI", "N s", 1.0)},
    MOVEMENT: {"peak": ("PK", "deg", 180 / np.pi), "fti": ("FTI", "deg s", 180 / np.pi)},
}

# Every train recorded here is a constant frequency train; the article also had doublet trains and reports
# its figures 5 and 6 on the CFTs alone. The filter is kept so the restriction is explicit rather than
# implied by the protocol, and so it still holds if doublets are ever added.
CFT_FREQUENCIES = (20, 33, 50)


def scored_trains(method, testing_only=True, cft_only=True) -> pd.DataFrame:
    """The per train scores of one route, in that route's units, optionally the testing CFTs alone."""
    table = train_frame(method)
    if table.empty:
        return table
    if testing_only and "set" in table:
        table = table[table["set"] == "testing"]
    if cft_only:
        table = table[table["frequency"].round().isin(CFT_FREQUENCIES)]
    for measure, (_, _, scale) in MEASURES[method].items():
        for side in ("measured", "predicted"):
            table = table.assign(**{f"{measure}_{side}": table[f"{measure}_{side}"] * scale})
    return table


def figure_4(name="figure4_agreement") -> Path:
    """
    The article's figure 4 on one canvas: measured against predicted, both routes.

    Integral on the left, peak on the right, as the article lays it out; one row per route, because the two
    observables have no common unit and a shared axis would only pretend they do. Each panel carries the
    identity line, the article's trendline through the origin, and the slope with the ICC.

    A caution the panel titles cannot carry: these points pool every testing train of every participant, so
    the between participant spread does most of the work in the ICC. It is the most flattering view of the
    identification in this folder, and it answers a different question from the per train R2.
    """
    import analysis

    tables = {method: scored_trains(method) for method in (ISOMETRIC, MOVEMENT)}
    tables = {method: table for method, table in tables.items() if table is not None and not table.empty}
    if not tables:
        print("figure 4: nothing scored yet")
        return None

    figure, axes = plt.subplots(len(tables), 2, figsize=(11.5, 5.4 * len(tables)), squeeze=False)
    every_subject = sorted({s for table in tables.values() for s in table["subject"].unique()})
    grey = shades(every_subject)

    for row, (method, table) in enumerate(tables.items()):
        colour = ISOMETRIC_COLOR if method == ISOMETRIC else MOVEMENT_COLOR
        for column, measure in enumerate(("fti", "peak")):
            label, unit, _ = MEASURES[method][measure]
            ax = axes[row][column]
            x = table[f"{measure}_measured"].to_numpy(dtype=float)
            y = table[f"{measure}_predicted"].to_numpy(dtype=float)
            scored = analysis.agreement(x, y)

            for subject in sorted(table["subject"].unique()):
                part = table[table["subject"] == subject]
                ax.plot(part[f"{measure}_measured"], part[f"{measure}_predicted"], "o", markersize=5,
                        color=grey[subject], markeredgecolor="0.25", markeredgewidth=0.4, zorder=3)

            top = float(np.nanmax([np.nanmax(x), np.nanmax(y)])) * 1.05
            ax.plot([0, top], [0, top], color="0.45", linestyle=":", linewidth=1.2, label="identity",
                    zorder=2)
            if np.isfinite(scored["slope"]):
                ax.plot([0, top], [0, scored["slope"] * top], color=colour, linewidth=1.8, zorder=4,
                        label=f"through origin, slope {scored['slope']:.3f}")

            low, high = scored["icc_ci"]
            ax.set_title(f"{ROUTE_LABEL[method]} - {label}   ICC {scored['icc']:.3f} "
                         f"[{low:.3f}, {high:.3f}]   n = {scored['n']}", fontsize=11, color=colour)
            ax.set_xlabel(f"measured {label} ({unit})")
            ax.set_ylabel(f"predicted {label} ({unit})")
            ax.set_xlim(0, top)
            ax.set_ylim(0, top)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=9, loc="upper left")

    figure.suptitle(f"Figure 4: measured against predicted, testing trains, "
                    f"{len(every_subject)} participant(s)", fontsize=13, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    shade_bar(figure, every_subject, ax=axes.ravel().tolist())
    return _save(figure, name)


def _relationship_figure(measure, number, only_testing=False, name=None) -> Path:
    """
    The article's figures 5 and 6: measured against predicted at each frequency, one row per pulse duration.

    The article's own layout - filled diamonds measured, open squares predicted, error bars one SD over the
    participants, a star where the paired t-test separates them - with its CFT and DFT columns replaced by
    the two routes, since this protocol recorded no doublet trains and does record two observables.

    Every train is drawn, identification and testing alike, as the article draws every condition it tested.
    Restricting to the testing set would leave one point per panel: the five testing trains sit at five
    different pulse widths, so no panel would hold a relationship at all.

    Parameters
    ----------
    measure: str
        "fti" or "peak"
    number: int
        5 or 6, the article's figure number, used in the title and the file name
    only_testing: bool
        Draw the testing trains alone. Off by default, see above.
    """
    import analysis

    tables = {method: scored_trains(method, testing_only=only_testing) for method in (ISOMETRIC, MOVEMENT)}
    tables = {method: table for method, table in tables.items() if table is not None and not table.empty}
    if not tables:
        print(f"figure {number}: nothing scored yet")
        return None

    pulse_widths = sorted({int(round(pw)) for table in tables.values()
                           for pw in table["pulse_width_us"].unique()})
    figure, axes = plt.subplots(len(pulse_widths), len(tables), squeeze=False,
                               figsize=(5.2 * len(tables), 2.9 * len(pulse_widths)))

    for row, pulse_width in enumerate(pulse_widths):
        for column, (method, table) in enumerate(tables.items()):
            ax = axes[row][column]
            label, unit, _ = MEASURES[method][measure]
            at_width = table[table["pulse_width_us"].round() == pulse_width]
            frequencies = sorted(at_width["frequency"].round().unique())

            drawn = False
            for frequency in frequencies:
                part = at_width[at_width["frequency"].round() == frequency]
                measured = part[f"{measure}_measured"].to_numpy(dtype=float)
                predicted = part[f"{measure}_predicted"].to_numpy(dtype=float)
                if not len(measured):
                    continue
                drawn = True
                # The two markers would sit on top of each other at the same frequency
                for offset, values, marker, face in ((-0.7, measured, "D", "black"),
                                                     (0.7, predicted, "s", "white")):
                    ax.errorbar(frequency + offset, np.mean(values), yerr=np.std(values, ddof=1)
                                if len(values) > 1 else 0.0,
                                fmt=marker, markersize=6, color="black", markerfacecolor=face,
                                capsize=3, linewidth=1, zorder=3)

                scored = analysis.agreement(measured, predicted)
                if np.isfinite(scored["p"]) and scored["p"] < 0.05:
                    height = max(np.mean(measured) + np.std(measured, ddof=1) if len(measured) > 1 else 0,
                                 np.mean(predicted) + np.std(predicted, ddof=1) if len(predicted) > 1 else 0)
                    ax.text(frequency, height * 1.08, "*", ha="center", va="bottom", fontsize=13)

            if not drawn:
                ax.set_axis_off()
                continue
            ax.set_xlim(0, 60)
            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.2)
            ax.set_ylabel(f"{label} ({unit})", fontsize=9)
            if row == 0:
                ax.set_title(ROUTE_LABEL[method], fontsize=12, fontweight="bold",
                             color=ISOMETRIC_COLOR if method == ISOMETRIC else MOVEMENT_COLOR)
            if row == len(pulse_widths) - 1:
                ax.set_xlabel("Frequency (Hz)", fontsize=10)
            if column == len(tables) - 1:
                ax.text(1.04, 0.5, f"{pulse_width} $\\mu$s", transform=ax.transAxes, rotation=270,
                        va="center", ha="left", fontsize=10, fontweight="bold")

    handles = [
        Line2D([], [], color="black", marker="D", markerfacecolor="black", linestyle="none",
               label="measured"),
        Line2D([], [], color="black", marker="s", markerfacecolor="white", linestyle="none",
               label="predicted"),
        Line2D([], [], color="none", marker="", linestyle="none", label="*  paired t-test p < 0.05"),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=3, fontsize=10, frameon=False)
    label = MEASURES[ISOMETRIC][measure][0]
    subjects = max(table["subject"].nunique() for table in tables.values())
    figure.suptitle(f"Figure {number}: {label} against stimulation frequency, one row per pulse duration\n"
                    f"constant frequency trains, mean $\\pm$ SD over {subjects} participant(s)",
                    fontsize=13, fontweight="bold")
    figure.tight_layout(rect=(0, 0.03, 1, 0.95))
    return _save(figure, name or f"figure{number}_{measure}")


def figure_5(only_testing=False, name=None) -> Path:
    """The article's figure 5: the force-time integral against frequency, at each pulse duration."""
    return _relationship_figure("fti", 5, only_testing=only_testing, name=name)


def figure_6(only_testing=False, name=None) -> Path:
    """The article's figure 6: the peak against frequency, at each pulse duration."""
    return _relationship_figure("peak", 6, only_testing=only_testing, name=name)


# ---- Supplementary: what each step costs ---- #
STEPS = ("step1", "step2", "step3", "step4", "step5")
STEP_LABELS = {
    "step1": r"step 1  $\tau_1$ off the relaxation",
    "step2": r"step 2  $A$, $K_m$, $\tau_2$",
    "step3": r"step 3  $A$ per intensity",
    "step4": r"step 4  eq. 3 fit",
    "step5": r"step 5  prediction",
}


def timing_frame() -> pd.DataFrame:
    """One row per (subject, route) with the seconds each step took, from the stored `step_seconds`."""
    rows = []
    for method in (ISOMETRIC, MOVEMENT):
        for subject, result in load_route(method).items():
            seconds = result["extra"].get("step_seconds", {})
            if not seconds:
                continue
            rows.append({"subject": subject, "route": method,
                         **{step: float(seconds.get(step, np.nan)) for step in STEPS}})
    return pd.DataFrame(rows)


def figure_step_times(table=None, name="figureS1_step_times") -> Path:
    """
    What each step of the identification costs, isometric on the left and movement on the right.

    One stacked bar per participant, step 1 at the bottom and step 5 on top, so both the total and where
    the total goes are readable at once. The two panels do not share a y axis on purpose: the movement
    route solves the same five steps on the full musculoskeletal model, and forcing one scale would flatten
    the isometric panel into nothing rather than making the comparison easier - the axis labels carry it.

    Parameters
    ----------
    table: pd.DataFrame
        As `timing_frame` builds it. Read from `results/` when not given.
    """
    table = timing_frame() if table is None else table
    if table.empty:
        print("figure S1: no result carries step_seconds yet, rerun the two routes")
        return None

    subjects = sorted(table["subject"].unique())
    colours = plt.cm.viridis(np.linspace(0.12, 0.92, len(STEPS)))

    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), squeeze=False)
    for column, (method, colour) in enumerate(((ISOMETRIC, ISOMETRIC_COLOR), (MOVEMENT, MOVEMENT_COLOR))):
        ax = axes[0][column]
        part = table[table["route"] == method].set_index("subject")
        positions = np.arange(len(subjects))
        bottom = np.zeros(len(subjects))

        for step, shade in zip(STEPS, colours):
            values = np.array([float(part[step].get(subject, np.nan)) if subject in part.index else np.nan
                               for subject in subjects])
            values = np.nan_to_num(values, nan=0.0)
            ax.bar(positions, values, bottom=bottom, width=0.72, color=shade,
                   edgecolor="white", linewidth=0.4, label=STEP_LABELS[step])
            bottom += values

        ax.set_xticks(positions)
        ax.set_xticklabels([f"P{subject}" for subject in subjects], rotation=90, fontsize=8)
        ax.set_ylabel("time (s)")
        ax.set_title(f"{ROUTE_LABEL[method]}   median total {np.median(bottom[bottom > 0]):.0f} s"
                     if (bottom > 0).any() else ROUTE_LABEL[method],
                     fontsize=12, color=colour, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_axisbelow(True)

    axes[0][0].legend(loc="upper left", fontsize=8.5, frameon=True, framealpha=0.9)
    figure.suptitle("Time per identification step, by participant",
                    fontsize=14, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(figure, name)


# ---- Figure 3: the pipeline itself ---- #
def figure_3(name="figure3_pipeline") -> Path:
    """
    The identification pipeline, in the manner of Marion, Wexler & Hull (2013), figure 1.

    The two routes run the same five steps on two observables, so the steps are drawn once down a shared
    spine and only what actually differs is split into an isometric column and a movement column. What is
    collapsed is as informative as what is split: steps 1 and 4 are literally the same code for both.
    """
    figure, ax = plt.subplots(figsize=(11.5, 12.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 15.4)
    ax.set_axis_off()

    def box(x, y, width, height, text, face, edge, fontsize=9, weight="normal"):
        ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.06,rounding_size=0.12",
                                    linewidth=1.3, facecolor=face, edgecolor=edge, zorder=2))
        ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=fontsize,
                zorder=3, fontweight=weight, linespacing=1.35)

    def arrow(x0, y0, x1, y1, style="-|>", color="0.35", dashed=False):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=13,
                                     linewidth=1.2, color=color, zorder=1,
                                     linestyle="--" if dashed else "-"))

    SHARED, ISO, MOV = "#eeeeee", "#dceaf7", "#f9dcdc"
    ISO_EDGE, MOV_EDGE = ISOMETRIC_COLOR, MOVEMENT_COLOR

    # --- Inputs --- #
    box(0.5, 14.2, 4.2, 0.95, "Isometric force trains\n18 conditions, 3 x 6 grid", ISO, ISO_EDGE, 9.5, "bold")
    box(5.3, 14.2, 4.2, 0.95, "Elbow flexions\nsame 18 conditions", MOV, MOV_EDGE, 9.5, "bold")
    box(5.3, 13.1, 4.2, 0.72, "Passive articular torque\n(identified beforehand, held)", MOV, MOV_EDGE, 8)

    # --- The split, shared --- #
    box(0.5, 11.9, 9.0, 0.85,
        "protocol.py  -  reference intensity 400 us  |  13 identification  |  5 testing",
        SHARED, "0.4", 9.5, "bold")
    arrow(2.6, 14.2, 2.6, 12.78)
    arrow(7.4, 13.1, 7.4, 12.78)

    # --- Step 1, shared code --- #
    box(0.5, 10.5, 9.0, 1.05,
        "STEP 1   $\\tau_1$ off the relaxation, 50-25 % window, no solver   (relaxation.py, shared)\n"
        "isometric: on the measured force        movement: on the elbow excursion",
        SHARED, "0.4", 9.5)
    arrow(5.0, 11.9, 5.0, 11.6)

    # --- Step 2 --- #
    box(0.5, 9.0, 9.0, 1.15,
        "STEP 2   identify  $A$,  $K_m$,  $\\tau_2$   at the reference intensity, $\\tau_1$ held\n"
        "one ocp, one phase per train  -  DingModelSingleIntensity  -  free = (A, km_rest, tau2)",
        SHARED, "0.4", 9.5)
    arrow(5.0, 10.5, 5.0, 10.2)

    # --- Step 3 --- #
    box(0.5, 7.5, 9.0, 1.15,
        "STEP 3   identify  $A$  alone at each of the 5 other intensities, everything else held\n"
        "one ocp per pulse width  -  continuation: each starts where the last one landed",
        SHARED, "0.4", 9.5)
    arrow(5.0, 9.0, 5.0, 8.7)

    # --- Step 4 --- #
    box(0.5, 6.0, 9.0, 1.15,
        "STEP 4   fit eq. 3 through the six (pulse width, $A$) points  ->  $a'$,  $pd_0$,  $pd_t$\n"
        "no solver  -  flags: A not rising  |  $pd_0$ on its floor  |  $pd_0$ profile flat",
        SHARED, "0.4", 9.5)
    arrow(5.0, 7.5, 5.0, 7.2)

    # --- The full parameter set --- #
    box(2.0, 4.9, 6.0, 0.8,
        "$a'$,  $pd_0$,  $pd_t$,  $\\tau_1$,  $\\tau_2$,  $K_m$      ($R_0$ = 2, $\\tau_c$ = 20 ms fixed)",
        "#fff3cd", "0.4", 9.5, "bold")
    arrow(5.0, 6.0, 5.0, 5.75)

    # --- Step 5, the one place the two routes genuinely differ --- #
    box(0.4, 3.1, 4.4, 1.35,
        "STEP 5  isometric\nintegrate forward (IvpFes)\nover the 5 testing trains\n"
        "nothing is optimised",
        ISO, ISO_EDGE, 9)
    box(5.2, 3.1, 4.4, 1.35,
        "STEP 5  movement\nocp with free = ( ), so the\ndynamics alone determine it\nthe passive stop is held too",
        MOV, MOV_EDGE, 9)
    arrow(4.0, 4.9, 2.6, 4.5)
    arrow(6.0, 4.9, 7.4, 4.5)

    # --- Scoring, shared again --- #
    box(0.5, 1.8, 9.0, 1.0,
        "SCORING (analysis.py, shared)   peak, integral and $R^2$ on every testing train\n"
        "isometric: above 0 N          movement: above the starting angle",
        SHARED, "0.4", 9.5)
    arrow(2.6, 3.1, 2.6, 2.85)
    arrow(7.4, 3.1, 7.4, 2.85)

    box(1.5, 0.5, 7.0, 0.85,
        "Compared on  $A \\cdot \\tau_1$  (the plateau both routes constrain), mean and median",
        "#fff3cd", "0.4", 9.5, "bold")
    arrow(5.0, 1.8, 5.0, 1.4)

    handles = [
        Line2D([], [], color="0.4", marker="s", markersize=11, linestyle="none",
               markerfacecolor=SHARED, label="identical for both routes"),
        Line2D([], [], color=ISO_EDGE, marker="s", markersize=11, linestyle="none",
               markerfacecolor=ISO, label="isometric only"),
        Line2D([], [], color=MOV_EDGE, marker="s", markersize=11, linestyle="none",
               markerfacecolor=MOV, label="movement only"),
    ]
    ax.legend(handles=handles, loc="lower center", ncol=3, fontsize=9.5, frameon=False,
              bbox_to_anchor=(0.5, -0.03))
    ax.set_title("Ding (2007) identification pipeline, isometric and movement",
                 fontsize=14, fontweight="bold", pad=14)
    figure.tight_layout()
    return _save(figure, name)


def main(cross=False, subjects=None):
    """Draw every figure. `cross` recomputes the swap figure 2 needs, which is one ocp per subject."""
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
    if cross:
        print("cross prediction:")
        cross_predict(subjects=subjects)
    print("figures:")
    figure_3()
    figure_1()
    figure_2()
    figure_4()
    figure_5()
    figure_6()
    figure_step_times()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cross", action="store_true",
                        help="recompute the cross prediction (slow: one ocp per subject)")
    parser.add_argument("--subjects", nargs="+", help="only these subjects, ex: 01 02")
    arguments = parser.parse_args()
    main(cross=arguments.cross, subjects=arguments.subjects)
