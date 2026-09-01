"""
Final result per participant for the used identification method.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import store

import ding_model
import plots
import protocol
import relaxation

A4_PORTRAIT = (210 / 25.4, 297 / 25.4)
MEASURED, PREDICTED = plots.MEASURED_COLOR, plots.PREDICTED_COLOR
GOOD, BAD, MUTED = "tab:green", "tab:red", "0.45"
TOLERATED_AT_BOUND = ("pd0", "tau2")

ROLES = {
    "reference": ("#1f77b4", "R", "white"),
    "identification": ("#aec7e8", "o", "black"),
    "testing": ("#d62728", "x", "white"),
    "dropped": ("0.85", "-", "0.35"),
}

LABELS = {
    "a_scale": ("a'", 1.0, "N/s", ".1f"),
    "pd0": ("pd0", 1e6, "us", ".1f"),
    "pdt": ("pdt", 1e6, "us", ".1f"),
    "tau1_rest": ("tau1", 1e3, "ms", ".1f"),
    "tau2": ("tau2", 1e3, "ms", ".2f"),
    "km_rest": ("Km", 1.0, "", ".4f"),
}

TITLE, LABEL, TICK, TEXT, CELL = 7.5, 6.5, 5.5, 6.5, 5.0


def plot_subject(route, subject, story, folder=None):
    """
    The whole identification of one participant on one route, on one A4 sheet.
    """
    with plt.rc_context({"xtick.labelsize": TICK, "ytick.labelsize": TICK, "axes.labelsize": LABEL,
                         "legend.fontsize": TICK, "axes.titlesize": TITLE}):
        figure = plt.figure(figsize=A4_PORTRAIT)
        grid = figure.add_gridspec(
            7, 6,
            height_ratios=[0.72, 0.80, 0.90, 0.90, 0.90, 0.62, 0.84],
            left=0.085, right=0.975, top=0.950, bottom=0.030, hspace=0.80, wspace=0.90,
        )

        _header(figure.add_subplot(grid[0, :]), route, story)
        _conditions(figure.add_subplot(grid[1, 0:3]), story)
        _step1(figure.add_subplot(grid[1, 3:6]), route, story)
        _step2(figure.add_subplot(grid[2, :]), route, story)
        _step5(figure.add_subplot(grid[3, :]), route, story)
        _step4(figure.add_subplot(grid[4, 0:2]), story)
        _agreement(figure.add_subplot(grid[4, 2:4]), route, story, "peak")
        _agreement(figure.add_subplot(grid[4, 4:6]), route, story, "fti")
        _per_condition(figure.add_subplot(grid[5, :]), story)
        _diagnostics(figure.add_subplot(grid[6, :]), story)

        figure.suptitle(f"P{subject}   {route.METHOD}", fontsize=11, fontweight="bold", y=0.985)
        _save(figure, folder)


def _save(figure, folder=None):
    """
    Write the sheet where the route's figures go, keeping the page at A4.
    """
    folder = folder if folder is not None else plots._output_root
    if folder is not None:
        folder.mkdir(parents=True, exist_ok=True)
        figure.savefig(folder / "summary.svg", format="svg")
    if plots._show:
        plt.show()
    else:
        plt.close(figure)


# ---- The panels ---- #
def _header(axes, route, story):
    """The configuration, the parameters, the prediction score and the solver's verdict, as text."""
    axes.axis("off")
    testing = story["table"][story["table"]["set"] == "testing"]
    at_bound = story["at_bound"]
    unexpected = [name for name in at_bound if name not in TOLERATED_AT_BOUND]

    identified = "   ".join(
        f"{LABELS[name][0]} = {format(LABELS[name][1] * value, LABELS[name][3])}"
        f"{' ' + LABELS[name][2] if LABELS[name][2] else ''}"
        for name, value in story["parameters"].items() if name in LABELS
    )
    lines = [
        (f"observable {story['observable']}   kinematics {story['kinematics']}   cost {route.COST}   "
         f"passive torque {story['passive_torque']}   tau1 {story['tau1_policy']}   "
         f"tau2 {'identified' if story['tau2_identified'] else 'held'}", "black", "normal"),
        (identified, "black", "bold"),
        (f"plateau A*tau1 at {protocol.REFERENCE_PULSE_WIDTH_US:.0f} us = {story['plateau']:.1f} "
         f"N   eq. 3 R2 = {story['relationship']['r2']:.3f}   "
         f"A rises with the pulse width: {story['relationship']['monotonic']}", "black", "normal"),
        (f"testing: peak error {testing['peak_error'].mean():.1f} %   "
         f"integral error {testing['fti_error'].mean():.1f} %   R2 {testing['r2'].mean():.3f}   "
         f"rmse {story['rmse']:.3f} {route.UNIT}", "black", "normal"),
        (f"every step converged: {story['converged']}"
         + (f"   at bound: {', '.join(at_bound)}" if at_bound else "   nothing on a bound")
         + f"   solved in {story['seconds']:.0f} s ("
         + ", ".join(f"{key} {value:.0f}" for key, value in story["seconds_per_step"].items()) + ")",
         GOOD if story["converged"] and not unexpected else BAD, "normal"),
    ]
    for i, (line, colour, weight) in enumerate(lines):
        axes.text(0.0, 1.0 - 0.20 * i, line, transform=axes.transAxes, fontsize=TEXT, va="top",
                  family="monospace", color=colour, fontweight=weight)


def _conditions(axes, story):
    """
    The stimulation grid of this participant, and what each of its trains was used for.
    """
    axes.set_title("The trains, and what each was used for", fontweight="bold")
    axes.axis("off")

    # The frequency is a first column rather than a rowLabel, which matplotlib draws outside the table box
    roles = story["conditions"]
    cells, colours, inks = [], [], []
    for frequency in protocol.FREQUENCIES:
        row = [ROLES[roles.get((frequency, pulse_width), "dropped")] for pulse_width in protocol.PULSE_WIDTHS_US]
        cells.append([f"{frequency} Hz"] + [letter for _, letter, _ in row])
        colours.append(["white"] + [fill for fill, _, _ in row])
        inks.append(["black"] + [ink for _, _, ink in row])

    table = axes.table(
        cellText=cells, cellColours=colours,
        colLabels=[""] + [f"{pulse_width:.0f}" for pulse_width in protocol.PULSE_WIDTHS_US],
        colWidths=[0.20] + [0.133] * len(protocol.PULSE_WIDTHS_US),
        cellLoc="center", loc="center", bbox=[0.03, 0.18, 0.94, 0.64],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(CELL + 1)
    for (row, column), cell in table.get_celld().items():
        cell.set_linewidth(0.4)
        if row > 0:
            cell.get_text().set_color(inks[row - 1][column])
            cell.get_text().set_fontweight("bold")

    counts = {name: sum(1 for role in roles.values() if role == name)
              for name in ("reference", "identification", "testing")}
    axes.text(0.5, 0.13, "pulse width (us)", transform=axes.transAxes, ha="center", va="top",
              fontsize=CELL, color=MUTED)
    legend = (f"R reference x{counts['reference']} (steps 1, 2)      "
              f"o identification x{counts['identification']} (steps 3, 4)\n"
              f"x testing x{counts['testing']} (step 5)      - dropped by the processing")
    axes.text(0.5, 0.01, legend, transform=axes.transAxes, ha="center", va="top", fontsize=CELL,
              color=MUTED, linespacing=1.5)


def _step1(axes, route, story):
    """What tau1 was read off, all reference phases on one axes, the fitted window marked."""
    axes.set_title("Step 1: tau1 off the relaxation", fontweight="bold")
    if story["tau1_policy"] != "relaxation":
        axes.axis("off")
        axes.text(0.5, 0.5, f"no relaxation in this observable\ntau1 {story['tau1_policy']} at step 2",
                  transform=axes.transAxes, ha="center", va="center", fontsize=TEXT, color=BAD)
        return

    for time, signal, estimate in story["relaxations"]:
        axes.plot(time - time[0], signal, color=MEASURED, alpha=0.7, linewidth=0.9)
        fit = relaxation.relaxation_fit(time, signal)
        if fit["window"] is not None:
            window = fit["window"]
            axes.plot(fit["tail_time"][window] - time[0], fit["tail_force"][window], color=PREDICTED,
                      linewidth=1.8, label=f"{1e3 * estimate:.0f} ms" if np.isfinite(estimate) else None)
    axes.set_xlabel("Time (s)")
    axes.set_ylabel(route.YLABEL)
    axes.legend(title=f"mean {1e3 * story['tau1']:.1f} ms", title_fontsize=TICK, loc="lower left",
                framealpha=0.85)
    axes.grid(alpha=0.3)


def _step2(axes, route, story):
    """The reference phases step 2 identified its parameters on, measured against the model."""
    axes.set_title(f"Step 2: {', '.join(story['step2_free'])} on the reference intensity", fontweight="bold")
    _phases(axes, route, story["step2"], story["reference_labels"])
    axes.set_ylabel(route.YLABEL)


def _step5(axes, route, story):
    """The testing phases, which nothing was fitted to, measured against the prediction."""
    axes.set_title("Step 5: the testing phases, every parameter fixed", fontweight="bold")
    _phases(axes, route, story["prediction"], story["testing_labels"])
    axes.set_ylabel(route.YLABEL)


def _phases(axes, route, step, labels):
    """One step's phases laid end to end, measured against what the model produced."""
    for i, (time, measured, produced) in enumerate(zip(step["times"], step["targets"], step["identified"])):
        offset = step["offsets"][i]
        axes.plot(offset + time, route.SCALE * np.asarray(measured), color=MEASURED, linewidth=1.0,
                  label="measured" if i == 0 else None)
        axes.plot(offset + time, route.SCALE * np.asarray(produced), color=PREDICTED, linewidth=1.0,
                  linestyle="--", label="model" if i == 0 else None)
        if i:
            axes.axvline(offset, color="grey", linewidth=0.5, alpha=0.5)
        axes.annotate(labels[i], (offset + 0.5 * time[-1], 0.02), xycoords=("data", "axes fraction"),
                      ha="center", va="bottom", fontsize=CELL, color=MUTED)
    axes.set_xlabel("Time (s)")
    axes.legend(loc="upper right", framealpha=0.85, ncol=2)
    axes.margins(y=0.20)
    axes.grid(alpha=0.3)


def _step4(axes, story):
    """The A of every intensity and the eq. 3 curve fitted through them."""
    axes.set_title("Steps 3, 4: A vs pulse width", fontweight="bold")
    pulse_widths, a_values = story["pulse_widths"], story["a_values"]
    relationship = story["relationship"]

    axes.scatter(1e6 * pulse_widths, a_values, s=14, color=MEASURED, zorder=3, label="identified")
    if np.isfinite(relationship["a_scale"]):
        curve = np.linspace(0.9 * pulse_widths.min(), 1.1 * pulse_widths.max(), 200)
        axes.plot(1e6 * curve, ding_model.a_calculation(curve, relationship["a_scale"], relationship["pd0"],
                                                   relationship["pdt"]),
                  color=PREDICTED, linewidth=1.2, label=f"eq. 3, R2 {relationship['r2']:.2f}")
    axes.axvline(protocol.REFERENCE_PULSE_WIDTH_US, color="grey", linestyle=":", linewidth=0.8)
    axes.set_xlabel("Pulse width (us)")
    axes.set_ylabel("A (N/s)")
    axes.legend(loc="lower right", framealpha=0.85)
    axes.grid(alpha=0.3)


def _agreement(axes, route, story, measure):
    """One of the two scalars the article scores, measured on one axis and predicted on the other."""
    label, unit = ("Peak", route.UNIT) if measure == "peak" else ("Integral", f"{route.UNIT}.s")
    axes.set_title(f"{label}, measured vs predicted", fontweight="bold")

    table = story["table"]
    for name, marker, colour in (("identification", "o", MUTED), ("testing", "D", PREDICTED)):
        rows = table[table["set"] == name]
        axes.scatter(rows[f"{measure}_measured"], rows[f"{measure}_predicted"],
                     s=12, marker=marker, color=colour, alpha=0.85, label=name)

    limits = np.array([
        min(table[f"{measure}_measured"].min(), table[f"{measure}_predicted"].min()),
        max(table[f"{measure}_measured"].max(), table[f"{measure}_predicted"].max()),
    ])
    axes.plot(limits, limits, color="black", linewidth=0.7, linestyle=":")
    axes.set_xlabel(f"measured ({unit})")
    axes.set_ylabel(f"predicted ({unit})")
    axes.legend(loc="upper left", framealpha=0.85)
    axes.grid(alpha=0.3)


def _per_condition(axes, story):
    """The peak error of every testing condition, so one bad train is not hidden in a mean."""
    axes.set_title("Peak error of every testing condition", fontweight="bold")
    testing = story["table"][story["table"]["set"] == "testing"]
    errors = testing["peak_error"].to_numpy(dtype=float)

    positions = np.arange(len(errors))
    axes.barh(positions, errors, height=0.6,
              color=[GOOD if value <= 25 else BAD for value in errors], alpha=0.85)
    for position, value in zip(positions, errors):
        axes.annotate(f"{value:.0f} %", (value, position), xytext=(3, 0), textcoords="offset points",
                      va="center", fontsize=CELL, color=MUTED)
    axes.set_yticks(positions)
    axes.set_yticklabels(testing["condition"])
    axes.axvline(25, color="grey", linestyle=":", linewidth=0.8)
    axes.set_xlabel("Peak error (%)")
    axes.margins(x=0.12)
    axes.invert_yaxis()
    axes.grid(alpha=0.3, axis="x")


_SHORT_EXIT = {"Solved_To_Acceptable_Level": "acceptable", "Maximum_Iterations_Exceeded": "max iter",
               "Restoration_Failed": "restoration", "Infeasible_Problem_Detected": "infeasible"}


def _exit_of(block) -> str:
    """converged, and IPOPT's exit whenever it was not a clean optimum: acceptable level is not one."""
    verdict = str(block.get("converged", "-"))
    status = block.get("status", "")
    if status in ("", "Solve_Succeeded"):
        return verdict
    return f"{verdict} ({_SHORT_EXIT.get(status, status.replace('_', ' ').lower())})"


def _diagnostics(axes, story):
    """What the solver did at every step and how long it took: a fit on a bound is still a warning."""
    axes.axis("off")
    axes.set_title("What the solver did", fontweight="bold", loc="left", pad=10)

    rows = []
    for name, block in story["diagnostics"].items():
        rows.append([
            name,
            ", ".join(block.get("free", [])) or "-",
            str(block.get("iterations", "-")),
            f"{block['cost']:.2e}" if "cost" in block else "-",
            f"{block['seconds']:.1f}" if np.isfinite(block.get("seconds", np.nan)) else "-",
            _exit_of(block),
            ", ".join(block.get("at_bound", [])) or "-",
        ])
    solved = sum(block["seconds"] for block in story["diagnostics"].values()
                 if np.isfinite(block.get("seconds", np.nan)))
    rows.append(["total", f"{len(rows)} solves", "-", "-", f"{solved:.1f}", str(story["converged"]),
                 ", ".join(story["at_bound"]) or "-"])

    table = axes.table(
        cellText=rows, colLabels=["step", "identified", "iter", "cost", "s", "converged", "at bound"],
        loc="upper center", cellLoc="left", bbox=[0.0, -0.10, 1.0, 1.05],
        colWidths=[0.13, 0.20, 0.07, 0.11, 0.06, 0.17, 0.26],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(CELL)
    for (row, column), cell in table.get_celld().items():
        cell.set_linewidth(0.4)
        if row == 0:
            cell.get_text().set_fontweight("bold")
    for row in range(1, len(rows) + 1):
        on_bound = [name for name in rows[row - 1][6].split(", ") if name not in TOLERATED_AT_BOUND + ("-",)]
        if rows[row - 1][5] not in ("True", "-") or on_bound:
            table[row, 5].set_facecolor("#ffe1e1")
            table[row, 6].set_facecolor("#ffe1e1")
    for column in range(7):
        table[len(rows), column].set_text_props(fontweight="bold")
