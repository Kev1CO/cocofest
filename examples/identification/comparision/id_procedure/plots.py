"""
The per subject figures of one identification: the five steps, eq. 3, and the article's agreement plot.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import store

import protocol
from helper import debug_plots

MEASURED_COLOR = "tab:blue"
PREDICTED_COLOR = "tab:red"

# How each step 2 parameter is written in the figure's title
PARAMETER_LABELS = {
    "A": (r"$A$", 1.0, "N/s", ".1f"),
    "km_rest": (r"$K_m$", 1.0, "", ".4f"),
    "tau2": (r"$\tau_2$", 1e3, "ms", ".2f"),
    "tau1_rest": (r"$\tau_1$", 1e3, "ms", ".1f"),
}

_output_root: Path | None = None
_show: bool = False


def output_for(method, subject, debug=True, show=False):
    """Send the following figures to results/debug/<method>/P<subject>/ding, or nowhere when debug is off."""
    global _output_root, _show
    _output_root = debug_plots.DEBUG_ROOT / method / f"P{subject}" / "ding" if debug else None
    _show = show


def _layout(figure, title, fontsize=13):
    """
    Lay the axes out and reserve a band above them for the title.
    """
    header = 0.35 + 0.30 * (1 + title.count("\n"))
    fraction = min(header / figure.get_figheight(), 0.25)
    figure.tight_layout(rect=(0, 0, 1, 1 - fraction))
    figure.suptitle(title, fontsize=fontsize, fontweight="bold", y=1 - 0.5 * fraction, va="center")


def _finish(fig, name, folder=None):
    """Save the figure where `output_for` pointed, or in `folder` when it has a home of its own."""
    folder = folder if folder is not None else _output_root
    if folder is not None:
        folder.mkdir(parents=True, exist_ok=True)
        fig.savefig(folder / f"{name}.svg", format="svg", bbox_inches="tight")
    if _show:
        plt.show()
    else:
        plt.close(fig)


# ---- Eq. 3, the force-intensity relationship ---- #
def plot_force_intensity(pulse_widths, a_values, relationship, title, unit="N/s", name="force_intensity"):
    """The A identified at each intensity (s, N/s), with the eq. 3 fit_a_pulse_width result through them."""
    import ding_model

    pulse_widths = np.asarray(pulse_widths, dtype=float)
    a_values = np.asarray(a_values, dtype=float)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    fine = np.linspace(min(relationship["pd0"], pulse_widths.min()) * 0.95, pulse_widths.max() * 1.15, 400)
    curve = ding_model.a_calculation(fine, relationship["a_scale"], relationship["pd0"], relationship["pdt"])

    ax.plot(1e6 * fine, curve, color=PREDICTED_COLOR, label="eq. 3 fit")
    ax.axhline(relationship["a_scale"], color="0.6", linestyle=":", linewidth=1, label=r"$a'$ (asymptote)")
    ax.axvline(1e6 * relationship["pd0"], color="0.4", linestyle="--", linewidth=1, label=r"$pd_0$")
    ax.plot(1e6 * pulse_widths, a_values, "o", color=MEASURED_COLOR, markersize=8, label="A identified per intensity")

    ax.set_title(f"{title}\n" + rf"$a'$ = {relationship['a_scale']:.0f} {unit}, "
                 rf"$pd_0$ = {1e6 * relationship['pd0']:.0f} us, "
                 rf"$pd_t$ = {1e6 * relationship['pdt']:.0f} us, $R^2$ = {relationship['r2']:.4f}", fontsize=11)
    ax.set_xlabel("Pulse width (us)")
    ax.set_ylabel(f"A ({unit})")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    _finish(fig, name)


# ---- One panel per train ---- #
def plot_predictions(times, measured, predicted, conditions, unit, ylabel, title, scale=1.0, baselines=None,
                     name="predictions"):
    """
    Every train on its own axes, measured against predicted, annotated with how it scored.
    """
    import metrics

    if not len(times):
        return
    columns = min(3, len(times))
    rows = int(np.ceil(len(times) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4.6 * columns, 3.2 * rows), squeeze=False)

    for index in range(len(times)):
        ax = axes[index // columns][index % columns]
        offset = 0.0 if baselines is None else float(baselines[index])
        time = np.asarray(times[index], dtype=float)
        condition = conditions[index]

        ax.plot(time, (np.asarray(measured[index]) - offset) * scale, color=MEASURED_COLOR,
                linewidth=1.6, label="Measured")
        ax.plot(time, (np.asarray(predicted[index]) - offset) * scale, color=PREDICTED_COLOR,
                linestyle="--", linewidth=1.4, label="Predicted")

        scored = metrics.score_train(time, measured[index], predicted[index], baseline=offset)
        _, _, blend = debug_plots.condition_colors(
            protocol.as_microseconds(condition["pulse_width"]), condition["frequency"]
        )
        ax.set_facecolor((*blend, 0.10))
        ax.set_title(
            f"{protocol.condition_name(condition['frequency'], condition['pulse_width'])}   "
            rf"$R^2$={scored['r2']:.2f}, PK {scored['peak_error']:.0f}%, FTI {scored['fti_error']:.0f}%",
            fontsize=9,
        )
        ax.grid(True, alpha=0.2)
        if index % columns == 0:
            ax.set_ylabel(ylabel)
        if index // columns == rows - 1:
            ax.set_xlabel("Time (s)")

    for index in range(len(times), rows * columns):
        axes[index // columns][index % columns].axis("off")
    axes[0][0].legend(fontsize=8, loc="upper right")

    fig.suptitle(f"{title}  ({unit})", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _finish(fig, name)


# ---- Measured against predicted, the article's Figure 4 ---- #
def plot_agreement(table, unit, title, name="agreement"):
    """
    The two scalars the article scores, measured on one axis and predicted on the other.
    """
    import metrics

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
    for ax, measure, label, measure_unit in (
        (axes[0], "peak", "Peak", unit),
        (axes[1], "fti", "Time integral", f"{unit}.s"),
    ):
        testing = table[table["set"] == "testing"] if "set" in table else table
        measured = testing[f"{measure}_measured"].to_numpy(dtype=float)
        predicted = testing[f"{measure}_predicted"].to_numpy(dtype=float)

        if "set" in table:
            fitted = table[table["set"] == "identification"]
            ax.scatter(
                fitted[f"{measure}_measured"], fitted[f"{measure}_predicted"],
                s=45, facecolors="none", edgecolors="0.55", linewidths=1.2, label="Identification trains",
            )
        ax.scatter(measured, predicted, s=45, color=PREDICTED_COLOR, alpha=0.75, label="Testing trains")

        span = [0, 1.05 * float(np.nanmax(np.concatenate([measured, predicted])))]
        ax.plot(span, span, color="0.3", linewidth=1, label="Identity")

        scored = metrics.agreement(measured, predicted)
        ax.plot(span, [scored["slope"] * value for value in span], color=MEASURED_COLOR, linestyle="--",
                linewidth=1.3, label=f"Trendline through 0 (slope {scored['slope']:.3f})")

        ax.set_title(
            f"{label}\nICC = {scored['icc']:.3f} "
            f"[{scored['icc_ci'][0]:.2f}, {scored['icc_ci'][1]:.2f}], "
            f"error {scored['model_error']:.1f} %, p = {scored['p']:.3f}",
            fontsize=10,
        )
        ax.set_xlabel(f"Measured ({measure_unit})")
        ax.set_ylabel(f"Predicted ({measure_unit})")
        ax.set_xlim(span)
        ax.set_ylim(span)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    _finish(fig, name)


# ---- Step 1, what tau1 was read off ---- #
def plot_tau1_relaxation(fits, tau1, subject, labels=None, high=0.50, low=0.25, folder=None,
                         ylabel="Force (N)", unit="N", route="step 1"):
    """
    What step 1 read tau1 off, all of one participant's reference trains on one figure.
    """
    rows = len(fits)
    if not rows:
        return

    figure, axes = plt.subplots(rows, 2, figsize=(11.5, 3.3 * rows), squeeze=False)
    for row, fit in enumerate(fits):
        time, force = fit["time"], fit["force"]
        peak_index, window, line = fit["peak_index"], fit["window"], fit["fit"]
        tail_time, tail_force = fit["tail_time"], fit["tail_force"]
        peak = float(force[peak_index])
        label = labels[row] if labels is not None and row < len(labels) else f"train {row}"

        # --- The force, and the band the article takes the window from --- #
        ax = axes[row][0]
        ax.plot(time, force, color=MEASURED_COLOR, linewidth=1.6, label=f"Measured ({unit})")
        ax.plot(time[peak_index], peak, "o", color="black", markersize=6, label=f"Peak ({peak:.1f} {unit})")
        ax.axhspan(low * peak, high * peak, color="0.5", alpha=0.12,
                   label=f"{100 * high:.0f}-{100 * low:.0f} % of peak")
        for fraction in (high, low):
            ax.axhline(fraction * peak, color="0.5", linestyle=":", linewidth=1)

        if window is not None:
            used_time, used_force = tail_time[window], tail_force[window]
            ax.plot(used_time, used_force, "o", color=PREDICTED_COLOR, markersize=4,
                    label=f"Fitted samples ({len(used_time)})")
            if line is not None:
                span = tail_time[tail_time >= used_time[0]]
                ax.plot(span, np.exp(line[1] + line[0] * (span - used_time[0])), color=PREDICTED_COLOR,
                        linestyle="--", linewidth=1.4, label=r"$S_0\,e^{-t/\tau_1}$")

        found = "no usable window" if not np.isfinite(fit["tau1"]) else rf"$\tau_1$ = {1e3 * fit['tau1']:.1f} ms"
        ax.set_title(f"{label}   {found}", fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_ylim(min(0.0, float(np.min(force))), 1.15 * peak if peak > 0 else 1.0)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, loc="upper right")

        # --- The same relaxation in log space, which is where the slope is actually fitted --- #
        ax = axes[row][1]
        positive = tail_force > 0
        ax.plot(tail_time[positive], np.log(tail_force[positive]), color=MEASURED_COLOR, linewidth=1.2,
                label="log signal, whole relaxation")
        if window is not None:
            used_time, used_force = tail_time[window], tail_force[window]
            ax.axvspan(used_time[0], used_time[-1], color="0.5", alpha=0.12,
                       label=f"fitted over {1e3 * (used_time[-1] - used_time[0]):.0f} ms")
            for edge in (used_time[0], used_time[-1]):
                ax.axvline(edge, color="0.5", linestyle=":", linewidth=1)

            if line is not None:
                ax.plot(used_time, np.log(used_force), "o", color=PREDICTED_COLOR, markersize=4,
                        label="Fitted samples")
                ax.plot(used_time, line[1] + line[0] * (used_time - used_time[0]), color=PREDICTED_COLOR,
                        linestyle="--", linewidth=1.6, label=f"slope = {line[0]:.3g} 1/s")
        ax.set_title(f"{label}   log space", fontsize=10)
        ax.set_ylabel("log signal")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)

    for column in (0, 1):
        axes[-1][column].set_xlabel("Time (s)")

    kept = [fit["tau1"] for fit in fits if np.isfinite(fit["tau1"])]
    mean = "no train gave one" if not np.isfinite(tau1) else rf"$\tau_1$ = {1e3 * tau1:.1f} ms"
    _layout(figure, f"[debug] P{subject} {route}: {mean}, from {len(kept)}/{len(fits)} relaxation(s)")
    _finish(figure, f"P{subject}", folder=folder)


# ---- Step 2 and 3, what the solver actually fitted ---- #
def _draw_fit_row(axes_pair, time, tracked, identified, label, ylabel="Force (N)", unit="N", scale=1.0,
                  note=""):
    """
    One train on one row: the tracked signal against the identified output, and the residual beside it.
    """
    time = np.asarray(time, dtype=float).squeeze()
    measured = scale * np.asarray(tracked, dtype=float).squeeze()
    predicted = scale * np.asarray(identified, dtype=float).squeeze()
    residual = predicted - measured
    rmse = float(np.sqrt(np.mean(residual**2)))
    suffix = f", {note}" if note else ""

    # --- The tracked signal against what the identified parameters produced --- #
    ax = axes_pair[0]
    ax.plot(time, measured, color=MEASURED_COLOR, linewidth=1.6, label=f"Tracked ({unit})")
    ax.plot(time, predicted, color=PREDICTED_COLOR, linewidth=1.4, linestyle="--",
            label=f"Identified ({unit})")
    for signal, color in ((measured, MEASURED_COLOR), (predicted, PREDICTED_COLOR)):
        peak = int(np.argmax(signal))
        ax.plot(time[peak], signal[peak], "o", color=color, markersize=5)

    peak_error = 100 * (predicted.max() - measured.max()) / measured.max() if measured.max() else np.nan
    ax.set_title(f"{label}   RMSE = {rmse:.3g} {unit}, peak {peak_error:+.1f} %{suffix}", fontsize=10)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, loc="upper right")

    # --- The residual, which is what the cost is built out of --- #
    ax = axes_pair[1]
    ax.axhline(0.0, color="0.4", linestyle=":", linewidth=1)
    ax.plot(time, residual, color=PREDICTED_COLOR, linewidth=1.2, label="Identified - tracked")
    ax.fill_between(time, 0.0, residual, color=PREDICTED_COLOR, alpha=0.15)
    bias = float(np.mean(residual))
    ax.axhline(bias, color="0.4", linestyle="--", linewidth=1, label=f"mean = {bias:+.3g} {unit}")
    ax.set_title(f"{label}   residual, {100 * rmse / (measured.max() or 1):.1f} % of peak", fontsize=10)
    ax.set_ylabel(f"Residual ({unit})")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    return rmse


def _parameter_text(parameters, names=None):
    """The identified parameters as one line, in the article's symbols."""
    pieces = []
    for name in names if names is not None else parameters:
        if name not in parameters:
            continue
        symbol, scale, unit, fmt = PARAMETER_LABELS.get(name, (name, 1.0, "", ".4g"))
        value = format(scale * parameters[name], fmt)
        pieces.append(f"{symbol} = {value}{' ' + unit if unit else ''}")
    return ", ".join(pieces)


def plot_step2_fit(times, tracked, identified, parameters, subject, labels=None, free=None, at_bound=None,
                   converged=None, cost=None, folder=None, ylabel="Force (N)", unit="N", scale=1.0,
                   route="step 2", name=None):
    """
    What step 2 fitted its parameters to, all of one participant's reference trains on one figure.
    """
    rows = len(times)
    if not rows:
        return

    figure, axes = plt.subplots(rows, 2, figsize=(11.5, 3.3 * rows), squeeze=False)
    for row in range(rows):
        label = labels[row] if labels is not None and row < len(labels) else f"train {row}"
        _draw_fit_row(axes[row], times[row], tracked[row], identified[row], label,
                      ylabel=ylabel, unit=unit, scale=scale)

    for column in (0, 1):
        axes[-1][column].set_xlabel("Time (s)")

    found = _parameter_text(parameters, names=free)
    verdict = "" if converged is None else f", converged = {converged}"
    total = "" if cost is None else f", cost = {cost:.4g}"
    pinned = f", on bound: {', '.join(at_bound)}" if at_bound else ""
    _layout(figure, f"[debug] P{subject} {route}: {found}{verdict}{total}{pinned}")
    _finish(figure, name if name is not None else f"P{subject}", folder=folder)


# ---- Step 3, one A per intensity ---- #
def plot_step3_fit(groups, subject, folder=None, ylabel="Force (N)", unit="N", scale=1.0,
                   route="step 3", name=None):
    """
    What step 3 fitted A to, every remaining identification intensity of one participant on one figure.
    """
    rows = sum(len(step["times"]) for _, step, _ in groups)
    if not rows:
        return

    figure, axes = plt.subplots(rows, 2, figsize=(11.5, 3.3 * rows), squeeze=False)
    row = 0
    for pulse_width_us, step, labels in groups:
        a_value = step["parameters"]["A"]
        note = f"A = {a_value:.1f} N/s" + ("" if step["converged"] else ", NOT CONVERGED")
        for i in range(len(step["times"])):
            label = labels[i] if labels is not None and i < len(labels) else f"{pulse_width_us:.0f} us"
            _draw_fit_row(axes[row], step["times"][i], step["targets"][i], step["identified"][i], label,
                          ylabel=ylabel, unit=unit, scale=scale, note=note)
            row += 1

    for column in (0, 1):
        axes[-1][column].set_xlabel("Time (s)")

    identified = ", ".join(f"{pulse_width_us:.0f} us: {step['parameters']['A']:.0f}"
                           for pulse_width_us, step, _ in groups)
    failed = [f"{pulse_width_us:.0f} us" for pulse_width_us, step, _ in groups if not step["converged"]]
    verdict = f" - NOT CONVERGED at {', '.join(failed)}" if failed else " - all converged"
    _layout(figure, f"[debug] P{subject} {route}{verdict}\nA (N/s) at {identified}")
    _finish(figure, name if name is not None else f"P{subject}", folder=folder)


# ---- Step 5, what the identified set predicts on trains it never saw ---- #
def plot_step5_prediction(times, tracked, predicted, table, subject, folder=None,
                          ylabel="Force (N)", unit="N", scale=1.0, route="step 5", name=None):
    """
    What the whole parameter set predicts, every testing train of one participant on one figure.
    """
    rows = len(times)
    if not rows:
        return None

    figure, axes = plt.subplots(rows, 2, figsize=(11.5, 3.3 * rows), squeeze=False)
    for row in range(rows):
        scored = table.iloc[row] if table is not None and row < len(table) else None
        label = scored["condition"] if scored is not None else f"train {row}"
        note = f"$R^2$ = {scored['r2']:.3f}" if scored is not None else ""
        _draw_fit_row(axes[row], times[row], tracked[row], predicted[row], label,
                      ylabel=ylabel, unit=unit, scale=scale, note=note)

    for column in (0, 1):
        axes[-1][column].set_xlabel("Time (s)")

    headline = ""
    if table is not None and not table.empty:
        headline = (f": PK {table['peak_error'].mean():.1f} % (median {table['peak_error'].median():.1f}), "
                    f"FTI {table['fti_error'].mean():.1f} %, "
                    f"$R^2$ {table['r2'].mean():.3f} (median {table['r2'].median():.3f})")
    _layout(figure, f"[debug] P{subject} {route}, {rows} testing train(s){headline}")
    return _finish(figure, name if name is not None else f"P{subject}", folder=folder)


# ---- Step 4, eq. 3 through the A of every intensity ---- #
def plot_step4_relationship(pulse_widths, a_values, relationship, subject, reference_us=None,
                            profile=None, folder=None, unit="N/s", route="step 4", name=None):
    """
    The force-intensity relationship step 4 fitted, and where it misses.
    """
    import ding_model

    pulse_widths = np.asarray(pulse_widths, dtype=float)
    a_values = np.asarray(a_values, dtype=float)
    predicted = ding_model.a_calculation(pulse_widths, relationship["a_scale"], relationship["pd0"],
                                    relationship["pdt"])
    residual = a_values - predicted
    profile = ding_model.profile_pd0(pulse_widths, a_values) if profile is None else profile

    figure, axes = plt.subplots(1, 3, figsize=(17.5, 5))

    # --- The points and the exponential through them --- #
    ax = axes[0]
    fine = np.linspace(min(relationship["pd0"], pulse_widths.min()) * 0.95, pulse_widths.max() * 1.15, 400)
    ax.plot(1e6 * fine, ding_model.a_calculation(fine, relationship["a_scale"], relationship["pd0"],
                                            relationship["pdt"]),
            color=PREDICTED_COLOR, label="eq. 3 fit")
    ax.axhline(relationship["a_scale"], color="0.6", linestyle=":", linewidth=1, label=r"$a'$ (asymptote)")
    ax.axvline(1e6 * relationship["pd0"], color="0.4", linestyle="--", linewidth=1, label=r"$pd_0$")
    ax.plot(1e6 * pulse_widths, a_values, "o", color=MEASURED_COLOR, markersize=8,
            label="A identified (step 3)")
    if reference_us is not None:
        on_reference = np.isclose(1e6 * pulse_widths, reference_us)
        ax.plot(1e6 * pulse_widths[on_reference], a_values[on_reference], "o", color="black", markersize=8,
                fillstyle="none", markeredgewidth=2, label="reference (step 2)")
    ax.set_title(rf"$a'$ = {relationship['a_scale']:.0f} {unit}, "
                 rf"$pd_0$ = {1e6 * relationship['pd0']:.0f} us, "
                 rf"$pd_t$ = {1e6 * relationship['pdt']:.0f} us", fontsize=11)
    ax.set_xlabel("Pulse width (us)")
    ax.set_ylabel(f"A ({unit})")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)

    # --- Where the exponential misses, which is what the monotonicity flag is about --- #
    ax = axes[1]
    ax.axhline(0.0, color="0.4", linestyle=":", linewidth=1)
    ax.stem(1e6 * pulse_widths, residual, linefmt="-", markerfmt="o", basefmt=" ")
    for pulse_width, value in zip(1e6 * pulse_widths, residual):
        ax.annotate(f"{value:+.0f}", (pulse_width, value), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8)
    ax.set_title(f"residual, {100 * np.max(np.abs(residual)) / (a_values.max() or 1):.1f} % of A at worst",
                 fontsize=11)
    ax.set_xlabel("Pulse width (us)")
    ax.set_ylabel(f"A identified - eq. 3 ({unit})")
    ax.grid(True, alpha=0.25)

    # --- Whether pd0 is a measurement or a bound: the cost with pd0 held and the other two refitted --- #
    ax = axes[2]
    best = float(np.min(profile["sse"])) or 1.0
    ax.plot(1e6 * profile["pd0"], profile["sse"] / best, "o-", color=PREDICTED_COLOR, markersize=4,
            label="best cost with pd0 held")
    ax.axvline(1e6 * profile["floor"], color="0.4", linestyle=":", linewidth=1.2,
               label=f"floor ({1e6 * profile['floor']:.0f} us)")
    ax.axvline(1e6 * relationship["pd0"], color=MEASURED_COLOR, linestyle="--", linewidth=1.4,
               label=rf"fitted $pd_0$ = {1e6 * relationship['pd0']:.0f} us")
    verdict = ("minimum inside the range: the intensities carry a threshold" if profile["interior"]
               else "no interior minimum: pd0 is the bound, not a measurement")
    ax.set_title(f"pd0 profile - {verdict}", fontsize=10)
    ax.set_xlabel("pd0 held at (us)")
    ax.set_ylabel("cost / best")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    at_bound = [name for name, (low, high) in relationship["bounds"].items()
                if np.isclose(relationship[name], low) or np.isclose(relationship[name], high)]
    flags = f", R^2 = {relationship['r2']:.4f}"
    flags += "" if relationship["monotonic"] else ", A DOES NOT RISE WITH THE PULSE WIDTH"
    flags += f", on bound: {', '.join(at_bound)}" if at_bound else ""
    flags += "" if profile["interior"] else ", pd0 NOT IDENTIFIABLE"
    _layout(figure, f"[debug] P{subject} {route}: force-intensity relationship (eq. 3){flags}")
    _finish(figure, name if name is not None else f"P{subject}", folder=folder)
