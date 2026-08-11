"""
Debug plots shared by the optimization programs.
They check:
1. plot_tracking_target  every phase on one axes: the raw experimental signal, and the target the cost function
                         actually holds. Catches a bad resampling, a target attached to the wrong phase, or a unit mix-up.
2. plot_solution         the identified trajectory against the tracked data, the residual, and a condition strip
                         naming the stimulation frequency and pulse width of every train.
3. plot_passive_torque   the identified passive torque over the range it was identified on.

Figures are written as svg under `results/debug/<subject>/<stage>/` as soon as set_output() has been called.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

PULSE_WIDTH_RANGE_US = (250.0, 500.0)
FREQUENCY_RANGE_HZ = (20.0, 50.0)

# Cycled over the extracted phases so that two consecutive ones can be told apart
PHASE_COLORS = ("tab:red", "tab:purple")

PULSE_WIDTH_CMAP = LinearSegmentedColormap.from_list(
    "pulse_width", ["#bedc7f", "#89a257", "#4d8061", "#305d42", "#1e3a29"]
)
FREQUENCY_CMAP = LinearSegmentedColormap.from_list("frequency", ["#ffd4a3", "#ffaa5e", "#d08159"])

# Identified parameters
PARAMETER_LABELS = {
    "tau1_rest": r"$\tau_{1,rest}$",
    "tau2": r"$\tau_2$",
    "km_rest": r"$K_{m,rest}$",
    "a_scale": r"$A_{scale}$",
    "pd0": r"$pd_0$",
    "pdt": r"$pd_t$",
    "k1": r"$k_1$",
    "k2": r"$k_2$",
    "k3": r"$k_3$",
    "k4": r"$k_4$",
    "kc1": r"$k_{c1}$",
    "kc2": r"$k_{c2}$",
    "theta_min": r"$\theta_{min}$",
    "theta_max": r"$\theta_{max}$",
    "a1": r"$a_1$",
    "a2": r"$a_2$",
    "a3": r"$a_3$",
    "a4": r"$a_4$",
    "a5": r"$a_5$",
    "b": r"$b$",
}


def _parameter_table(ax, parameters, title="Identified parameters"):
    """
    Add the identified parameters as a boxed table on `ax`.

    matplotlib's "best" legend placement does the positioning, so the box follows the data instead of sitting at
    a fixed corner. The handles are empty on purpose: only the text matters.
    """
    if not parameters:
        return

    handles = [Line2D([], [], linestyle="none") for _ in parameters]
    labels = [f"{PARAMETER_LABELS.get(name, name)} = {value:.3g}" for name, value in parameters.items()]
    table = ax.legend(
        handles,
        labels,
        loc="best",
        title=title,
        handlelength=0,
        handletextpad=0,
        fontsize=9,
        framealpha=0.9,
        labelspacing=0.35,
    )
    table.get_title().set_fontsize(9)

DEBUG_ROOT = Path(__file__).resolve().parent.parent.parent / "results" / "debug"

_output_root: Path | None = None
_show: bool = False


def set_output(folder, show=False):
    """
    Send every following debug figure to `folder`, as svg, under a sub folder per stage.

    Parameters
    ----------
    folder: str | Path | None
        The subject's debug folder, ex: results/debug/force_id_all/P01. None disables saving.
    show: bool
        If the figures should also be opened on screen. Blocks until they are closed, so only for a single run.
    """
    global _output_root, _show
    _output_root = Path(folder) if folder is not None else None
    _show = show


def output_for(method, subject, debug=True, show=False):
    """
    Point the debug figures at one subject's folder for one stage.

    Parameters
    ----------
    method: str
        The stage name the result is stored under, ex: "force_id_all"
    subject: str
        The subject id, ex: "01"
    debug: bool
        False disables saving entirely
    show: bool
        If the figures should also be opened on screen
    """
    set_output(DEBUG_ROOT / method / f"P{subject}" if debug else None, show=show)


def _finish(fig, stage, name):
    """Save the figure under its stage folder, then show or close it."""
    if _output_root is not None:
        folder = _output_root / stage
        folder.mkdir(parents=True, exist_ok=True)
        fig.savefig(folder / f"{name}.svg", format="svg", bbox_inches="tight")
    if _show:
        plt.show()
    else:
        plt.close(fig)


# ---- Colours of the stimulation conditions ---- #
def _normalise(value, bounds):
    return float(np.clip((value - bounds[0]) / (bounds[1] - bounds[0]), 0, 1))


def condition_colors(pulse_width_us, frequency_hz):
    """The pulse width colour, the frequency colour, and the translucent blend used to band the plots above."""
    pulse_color = PULSE_WIDTH_CMAP(_normalise(pulse_width_us, PULSE_WIDTH_RANGE_US))
    frequency_color = FREQUENCY_CMAP(_normalise(frequency_hz, FREQUENCY_RANGE_HZ))
    blend = tuple(np.mean([pulse_color[:3], frequency_color[:3]], axis=0))
    return pulse_color, frequency_color, blend


def cost_function_target(ocp, key: str):
    """
    Read back, phase by phase, the target actually held by the tracking objective of a built ocp.

    Plotting this rather than the array that was passed in is what makes the check worth doing: it is the only
    way to see a target that was silently truncated, repeated or attached to the wrong phase.
    """
    targets = []
    for nlp in ocp.nlp:
        for penalty in nlp.J:
            if penalty is not None and penalty.target is not None and penalty.extra_parameters.get("key") == key:
                targets.append(np.asarray(penalty.target).squeeze())
                break
    return targets


def _as_phase_list(data):
    """Accept either one array or a list of per-phase arrays, always return a list of 1d arrays."""
    if isinstance(data, (list, tuple)):
        return [np.asarray(phase).squeeze() for phase in data]
    return [np.asarray(data).squeeze()]


def plot_tracking_target(raw_time, raw_data, target_time, target, title, unit, scale=1.0, name="tracking_target"):
    """
    Plot, on a single axes, the raw experimental signal of every phase and the target the cost function holds.

    Parameters
    ----------
    raw_time, raw_data: list | np.ndarray
        The raw experimental signal, per phase or as a single array
    target_time, target: list | np.ndarray
        The resampled signal actually tracked, per phase or as a single array
    title: str
        The figure title
    unit: str
        The unit of the plotted signal
    scale: float
        A factor applied to both signals before plotting, ex: 180/pi to show an angle in degrees
    name: str
        The svg file name
    """
    raw_time, raw_data = _as_phase_list(raw_time), _as_phase_list(raw_data)
    target_time, target = _as_phase_list(target_time), _as_phase_list(target)

    fig, ax = plt.subplots(figsize=(11, 5))
    for i in range(len(raw_data)):
        ax.plot(
            raw_time[i],
            raw_data[i] * scale,
            color="tab:blue",
            linewidth=2.5,
            alpha=0.45,
            label="Raw experimental data" if i == 0 else None,
        )
        ax.annotate(
            f"phase {i}",
            (raw_time[i][0], np.max(raw_data[i]) * scale),
            textcoords="offset points",
            xytext=(0, 8),
            fontsize=10,
            color="tab:blue",
            ha="left",
            va="bottom",
        )
    for i in range(len(target)):
        ax.plot(
            target_time[i],
            target[i] * scale,
            color="tab:red",
            linestyle="--",
            linewidth=1.2,
            label="Tracked target (cost function)" if i == 0 else None,
        )

    ax.set_title(f"[debug] {title}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"({unit})")
    ax.legend()
    _finish(fig, "pre_ocp", name)


def plot_phase_extraction(panels, title, unit, scale=1.0, name="phase_extraction"):
    """
    Plot every recording that check if the slicing kept the right part of the signal.

    Parameters
    ----------
    panels: list[dict]
        One entry per row, with "full_time", "full_data", "phase_times", "phase_data", "stim_times" and "label"
    title: str
        The figure title
    unit: str
        The unit of the plotted signal
    scale: float
        A factor applied before plotting, ex: 180/pi to show an angle in degrees
    name: str
        The svg file name
    """
    panels = [panels] if isinstance(panels, dict) else panels
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 3.2 * len(panels)), squeeze=False)

    for ax, panel in zip(axes[:, 0], panels):
        full_data = np.asarray(panel["full_data"]) * scale
        ax.plot(panel["full_time"], full_data, color="tab:blue", linewidth=1, label="Whole recording")

        for i in range(len(panel["phase_data"])):
            color = PHASE_COLORS[i % len(PHASE_COLORS)]
            phase_time = np.asarray(panel["phase_times"][i])
            phase_data = np.asarray(panel["phase_data"][i]) * scale

            ax.axvspan(phase_time[0], phase_time[-1], color=color, alpha=0.08, zorder=0)
            ax.axvline(phase_time[0], color=color, linewidth=1.0, alpha=0.7, zorder=1)
            ax.axvline(phase_time[-1], color=color, linewidth=1.0, alpha=0.7, linestyle="--", zorder=1)
            ax.plot(
                phase_time,
                phase_data,
                color=color,
                linewidth=2.5,
                label="Extracted phases" if i == 0 else None,
            )
            ax.plot(
                [phase_time[0], phase_time[-1]],
                [phase_data[0], phase_data[-1]],
                linestyle="none",
                marker="o",
                markersize=4,
                color=color,
                markeredgecolor="white",
                markeredgewidth=0.6,
                label="Phase start and end" if i == 0 else None,
            )
            ax.annotate(
                f"{i}",
                (phase_time[0], np.max(phase_data)),
                textcoords="offset points",
                xytext=(0, 8),
                fontsize=11,
                fontweight="bold",
                color=color,
            )

        stim_times = panel.get("stim_times")
        if stim_times is not None and len(stim_times):
            flat = [t for train in stim_times for t in train] if np.ndim(stim_times[0]) else list(stim_times)
            ax.scatter(flat, np.full(len(flat), full_data.min()), s=3, color="black", label="Stimulations")

        ax.set_title(panel.get("label", ""), fontsize=10)
        ax.set_ylabel(f"({unit})")

    axes[-1, 0].set_xlabel("Time (s)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"[debug] {title}")
    fig.tight_layout()
    _finish(fig, "pre_ocp", name)


def plot_solution(
    time,
    tracked,
    identified,
    title,
    unit,
    scale=1.0,
    phase_lengths=None,
    conditions=None,
    parameters=None,
    ylabel=None,
    name="solution",
):
    """
    Plot the identified trajectory against the tracked data, the residual, and the stimulation conditions.

    Parameters
    ----------
    time, tracked, identified: np.ndarray
        The time vector and the two signals to compare, all of the same length
    title: str
        The figure title
    unit: str
        The unit of the plotted signal
    scale: float
        A factor applied before plotting, ex: 180/pi to show an angle in degrees
    phase_lengths: list
        The number of samples of every phase. When given, the phases are drawn separately so that the jumps
        between two independent phases are not drawn as a trajectory.
    conditions: list[dict]
        One entry per train, with "t_start", "t_end", "pulse_width_us" and "frequency_hz". Drawn as a strip at
        the bottom and echoed as translucent bands behind the two plots above.
    parameters: dict
        The identified parameters, shown as a boxed table on the tracking plot
    ylabel: str
        The label of the tracking plot, ex: "Force (N)". Defaults to the unit alone.
    name: str
        The svg file name
    """
    time, tracked, identified = np.asarray(time), np.asarray(tracked), np.asarray(identified)
    slices = _phase_slices(len(time), phase_lengths)

    n_axes = 3 if conditions else 2
    ratios = [3, 1, 0.5] if conditions else [3, 1]
    fig, axes = plt.subplots(n_axes, 1, sharex=True, figsize=(11, 8), height_ratios=ratios)
    ax, ax_residual = axes[0], axes[1]

    for i, phase in enumerate(slices):
        ax.plot(time[phase], tracked[phase] * scale, color="tab:blue", label="Tracked" if i == 0 else None)
        ax.plot(
            time[phase],
            identified[phase] * scale,
            color="tab:red",
            linestyle="--",
            label="Identified" if i == 0 else None,
        )
        ax_residual.plot(time[phase], (identified[phase] - tracked[phase]) * scale, color="tab:grey")

    error = np.sqrt(np.mean((identified - tracked) ** 2)) * scale
    ax.set_title(f"[debug] {title} (RMSE = {error:.3g} {unit})")
    ax.set_ylabel(ylabel if ylabel else f"({unit})")

    ax.add_artist(ax.legend(loc="upper left", fontsize=9))
    _parameter_table(ax, parameters)
    ax_residual.axhline(0, color="black", linewidth=0.8)
    ax_residual.set_ylabel(f"Residual ({unit})")

    if conditions:
        _draw_condition_strip(axes[2], (ax, ax_residual), conditions)
        axes[2].set_xlabel("Time (s)")
    else:
        ax_residual.set_xlabel("Time (s)")

    fig.tight_layout()
    _finish(fig, "solution", name)


def _draw_condition_strip(ax, upper_axes, conditions):
    """
    Draw one rectangle per train, split in half: pulse width on top, stimulation frequency below. The blend of the
    two colours is echoed behind the plots above, so a condition can be told apart at a glance.
    """
    for condition in conditions:
        t0, t1 = condition["t_start"], condition["t_end"]
        pulse_width, frequency = condition["pulse_width_us"], condition["frequency_hz"]
        pulse_color, frequency_color, blend = condition_colors(pulse_width, frequency)

        ax.add_patch(Rectangle((t0, 0.5), t1 - t0, 0.5, facecolor=pulse_color, edgecolor="white", linewidth=1.0))
        ax.add_patch(Rectangle((t0, 0.0), t1 - t0, 0.5, facecolor=frequency_color, edgecolor="white", linewidth=1.0))
        ax.text(
            (t0 + t1) / 2, 0.75, f"{pulse_width:.0f} us", ha="center", va="center", fontsize=9, fontweight="bold"
        )
        ax.text(
            (t0 + t1) / 2, 0.25, f"{frequency:.0f} Hz", ha="center", va="center", fontsize=9, fontweight="bold"
        )

        for upper in upper_axes:
            upper.axvspan(t0, t1, color=blend, alpha=0.15, zorder=0)

    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel("Condition", fontsize=8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)


def _phase_slices(n_samples, phase_lengths):
    """Turn a list of per-phase sample counts into slices, or a single slice covering everything."""
    if not phase_lengths:
        return [slice(0, n_samples)]

    slices, start = [], 0
    for length in phase_lengths:
        slices.append(slice(start, min(start + length, n_samples)))
        start += length
    return slices


def plot_passive_torque(passive_torque, subject, theta_dot=0.0, parameters=None, name="passive_torque"):
    """
    Plot the passive torque against the elbow angle, over the range it was identified on.

    Parameters
    ----------
    passive_torque: PassiveTorque
        The identified passive torque model
    subject: str
        The subject id, used in the title
    theta_dot: float
        The joint angular velocity the curve is evaluated at (rad/s)
    name: str
        The svg file name
    """
    bounds = passive_torque.theta_bounds if passive_torque.theta_bounds else (0.0, np.pi)
    theta = np.linspace(bounds[0], bounds[1], 300)
    torque = np.asarray([float(passive_torque.torque(angle, theta_dot)) for angle in theta])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.rad2deg(theta), torque, color="tab:blue", label="Passive torque")
    ax.axhline(0, color="black", linewidth=0.8)

    for attribute, label, color in (
        ("theta_min", r"$\theta_{min}$", "tab:grey"),
        ("theta_max", r"$\theta_{max}$", "black"),
    ):
        angle = getattr(passive_torque, attribute, None)
        if angle is not None and bounds[0] <= angle <= bounds[1]:
            ax.axvline(np.rad2deg(angle), color=color, linestyle="--", linewidth=1, label=label)

    span = float(np.max(np.abs(torque)))
    ax.set_ylim(-1.15 * span, 1.15 * span)
    ax.set_title(f"[debug] P{subject} passive torque over the identified range")
    ax.set_xlabel("Joint angle (deg)")
    ax.set_ylabel("Passive torque (N.m)")
    ax.grid(True, alpha=0.3)
    ax.add_artist(ax.legend(loc="upper right", fontsize=9))
    _parameter_table(ax, parameters, title="Identified passive model")
    _finish(fig, "solution", name)
