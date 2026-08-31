"""Step 1 of the identification: tau1 read off the muscle relaxation, with no solver."""

from __future__ import annotations

import numpy as np

import store

import plots

TAU1_WINDOW = (0.50, 0.25)  # The window used by all Ding's articles (50-25%)
TAU1_BOUNDS = (5e-3, 1.0)  # Bounds to consider tau1


def relaxation_fit(time, signal, high=TAU1_WINDOW[0], low=TAU1_WINDOW[1]) -> dict:
    """
    The 50-25 % window of one relaxation and the log-linear fit through it.

    Returns
    -------
    dict
        The signal, where it peaks, the relaxation, the window, the (slope, intercept) of log signal
        against time, and the tau1 they give
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


def mean_tau1(times, signals, debug=False, subject=None, labels=None, window=TAU1_WINDOW,
              folder=None, ylabel="Force (N)", unit="N", route="step 1") -> tuple:
    """
    tau1 over the reference trains, and the per train estimates it was averaged from.
    """
    fits = [relaxation_fit(time, signal, high=window[0], low=window[1])
            for time, signal in zip(times, signals)]
    estimates = [fit["tau1"] for fit in fits]
    inside = [value for value in estimates if TAU1_BOUNDS[0] <= value <= TAU1_BOUNDS[1]]
    tau1 = float(np.mean(inside)) if inside else np.nan

    if debug:
        plots.plot_tau1_relaxation(fits, tau1, subject, labels=labels, high=window[0], low=window[1],
                                   folder=folder, ylabel=ylabel, unit=unit, route=route)
    return tau1, estimates
