"""
Identification procedure step 1.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import plots  # noqa: E402


TAU1_WINDOW = (0.50, 0.25)  # The window used by all Ding's articles (50-25%)
TAU1_BOUNDS = (5e-3, 1.0)  # Bounds to consider tau1


def relaxation_fit(time, signal, high=TAU1_WINDOW[0], low=TAU1_WINDOW[1]) -> dict:
    """
    The 50-25 % window of one relaxation and the log-linear fit through it.

    Parameters
    ----------
    time: np.ndarray
        The time vector of the train or movement (s)
    signal: np.ndarray
        What relaxes: the measured force (N), or the elbow excursion above its starting angle (deg)
    high, low: float
        The window, as fractions of the peak

    Returns
    -------
    dict
        The signal, where it peaks, the relaxation, the window, the (slope, intercept) of log signal
        against time, and the tau1 they give.
    """
    time, signal = np.asarray(time, dtype=float), np.asarray(signal, dtype=float)
    peak_index = int(np.argmax(signal))
    peak = float(signal[peak_index])

    tail_time, tail_signal = time[peak_index:], signal[peak_index:]
    inside = ((tail_signal <= high * peak) & (tail_signal >= low * peak) if peak > 0
              else np.zeros(len(tail_signal), dtype=bool))

    tau1, window, fit = np.nan, None, None
    if inside.sum() >= 3:
        first, last = int(np.argmax(inside)), len(inside) - 1 - int(np.argmax(inside[::-1]))
        window = slice(first, last + 1)
        if tail_signal[window].min() > 0:
            fit = np.polyfit(tail_time[window] - tail_time[first], np.log(tail_signal[window]), 1)
            tau1 = float(-1 / fit[0]) if fit[0] < 0 else np.nan

    return {"time": time, "force": signal, "peak_index": peak_index, "tail_time": tail_time,
            "tail_force": tail_signal, "window": window, "fit": fit, "tau1": tau1}


def tau1_from_relaxation(time, signal, high=TAU1_WINDOW[0], low=TAU1_WINDOW[1]) -> float:
    """
    tau1 off one relaxation.

    Returns
    -------
    float
        tau1 (s), or nan when the signal never relaxed far enough to hold a window
    """
    return relaxation_fit(time, signal, high=high, low=low)["tau1"]


def mean_tau1(times, signals, debug=False, subject=None, labels=None, window=TAU1_WINDOW,
              folder=None, ylabel="Force (N)", unit="N", route="step 1") -> tuple:
    """
    tau1 over the reference trains

    Parameters
    ----------
    times, signals: list[np.ndarray]
        One entry per train or movement
    debug: bool
        Write the figure showing what tau1 was read off, one row per train
    subject: str
        The subject id, which names that figure
    labels: list[str]
        The stimulation condition of every train, for the row titles
    window: tuple
        The (high, low) fractions of the peak the fit is taken between
    folder: Path
        Where the figure goes. Defaults to the isometric route's, so the movement route passes its own.
    ylabel, unit, route: str
        What the signal is and which route asked, so the two figures can be told apart

    Returns
    -------
    tuple
        The mean tau1 (s) fitted over the relaxation windows
    """
    fits = [relaxation_fit(time, signal, high=window[0], low=window[1])
            for time, signal in zip(times, signals)]
    estimates = [fit["tau1"] for fit in fits]
    inside = [value for value in estimates if TAU1_BOUNDS[0] <= value <= TAU1_BOUNDS[1]]
    tau1 = float(np.mean(inside)) if inside else np.nan

    if debug:
        plots.plot_tau1_relaxation(
            fits, tau1, subject, labels=labels,
            high=window[0], low=window[1],
            folder=folder if folder is not None else plots.STEP1_ROOT,
            ylabel=ylabel, unit=unit, route=route,
        )
    return tau1, estimates
