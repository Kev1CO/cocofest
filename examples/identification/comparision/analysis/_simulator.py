"""
What one parameter set would produce, on the observable the route it belongs to identifies.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

import _shared

import ding_model
import metrics
import protocol
import steps
from cocofest import IvpFes

STIMULATION_SECONDS = 1.0
TRAIN_SECONDS = 1.25

# What eq. 3 and the state equations take, in the order the sensitivity tables print them
FREE = ("a_scale", "km_rest", "tau1_rest", "tau2", "pd0", "pdt")


def stim_times(frequency, stimulation=STIMULATION_SECONDS) -> list:
    """When the pulses of one train fall, for a stimulation frequency in Hz."""
    return list(np.arange(0.0, stimulation, 1.0 / float(frequency)))


def simulate(parameters: dict, frequency, pulse_width_us, final_time=TRAIN_SECONDS) -> tuple:
    """
    One train, integrated forward: the time base (s) and the force the model produced (N).
    """
    model = ding_model.DingModelArticle(
        stim_time=stim_times(frequency),
        **{name: float(value) for name, value in parameters.items() if name in FREE},
    )
    program = IvpFes(
        fes_parameters={"model": model, "pulse_width": float(protocol.as_microseconds(pulse_width_us)) * 1e-6},
        ivp_parameters={"final_time": float(final_time)},
    )
    states, time = program.integrate()
    return np.asarray(time, float).squeeze(), np.asarray(states["F"], float).squeeze()


def outcome(parameters: dict, frequency, pulse_width_us, final_time=TRAIN_SECONDS) -> dict:
    """
    The two scalars every score in this study is built on, off the muscle model alone.
    """
    try:
        time, force = simulate(parameters, frequency, pulse_width_us, final_time)
    except Exception:
        return {"peak": np.nan, "fti": np.nan}
    if not np.all(np.isfinite(force)):
        return {"peak": np.nan, "fti": np.nan}
    return {"peak": metrics.peak(force), "fti": metrics.time_integral(time, force)}


# ---- The routes whose skeleton moves, run through their own program ---- #
@lru_cache(maxsize=4)
def _phases_of(route, subject: str) -> tuple:
    """
    One participant's phases, weighted as the steps weight them, kept between calls.
    """
    phases = route.load(subject, debug=False)
    scale = steps.scale_of(phases)
    for phase in phases:
        phase["cost_scale"] = scale
    return tuple(phases)


def _phase_at(route, subject: str, frequency, pulse_width_us):
    """The participant's own recording of one stimulation condition, or None when they never did it."""
    phases = _phases_of(route, subject)
    wanted = (protocol.nearest(float(frequency), protocol.FREQUENCIES),
              protocol.nearest(protocol.as_microseconds(pulse_width_us), protocol.PULSE_WIDTHS_US))
    for i, phase in enumerate(phases):
        recorded = (protocol.nearest(float(phase["frequency"]), protocol.FREQUENCIES),
                    protocol.nearest(protocol.as_microseconds(phase["pulse_width"]), protocol.PULSE_WIDTHS_US))
        if recorded == wanted:
            return list(phases), i
    return None, None


def simulate_route(route, subject: str, parameters: dict, frequency, pulse_width_us) -> tuple:
    """
    One condition run through the route's own program with every parameter fixed.
    """
    phases, index = _phase_at(route, subject, frequency, pulse_width_us)
    if phases is None:
        raise ValueError(f"P{subject} never recorded {protocol.condition_name(frequency, pulse_width_us)}")

    one = steps.select(phases, [index])
    settled = steps.parameter_table(parameters["tau1_rest"])
    step = steps.solve(route, subject, one, (), ding_model.DingModelArticle, settled,
                       constants={name: float(value) for name, value in parameters.items() if name in FREE},
                       max_iter=5000)
    return np.asarray(step["times"][0], float), np.asarray(step["identified"][0], float)


def outcome_of(route, subject: str, parameters: dict, frequency, pulse_width_us) -> dict:
    """
    The two scalars, on the observable `route` identifies rather than on the muscle force behind it.
    """
    if route.family == "isometric":
        return outcome(parameters, frequency, pulse_width_us)

    try:
        time, produced = simulate_route(route, subject, parameters, frequency, pulse_width_us)
    except Exception:
        return {"peak": np.nan, "fti": np.nan}
    if not produced.size or not np.all(np.isfinite(produced)):
        return {"peak": np.nan, "fti": np.nan}

    # The angle is scored above the posture the flexion started from, the way steps.score scores it
    baseline = float(produced[0])
    return {"peak": metrics.peak(produced, baseline),
            "fti": metrics.time_integral(time, produced, baseline)}


def conditions(frequencies=None, pulse_widths_us=None) -> list:
    """The (frequency, pulse width) conditions a sensitivity campaign sweeps, the study's grid by default."""
    return [(frequency, pulse_width)
            for frequency in (frequencies or protocol.FREQUENCIES)
            for pulse_width in (pulse_widths_us or protocol.PULSE_WIDTHS_US)]


def box_around(parameters: dict, span: float, limits: dict = None) -> dict:
    """
    A multiplicative band around one participant's identified values, clipped to what the model allows.
    """
    box = {}
    for name in FREE:
        if name not in parameters:
            continue
        value = float(parameters[name])
        low, high = value / span, value * span
        if limits and name in limits:
            floor, ceiling = limits[name]
            low, high = max(low, floor), min(high, ceiling)
        box[name] = (low, high) if high > low else (value * 0.999, value * 1.001)
    return box


def model_limits() -> dict:
    """The admissible range of every free parameter, as the identification steps declare it."""
    import steps

    limits = {}
    for name in FREE:
        guess, low, high, _ = steps.IDENTIFIED_PARAMETERS[name]
        limits[name] = (low, high if high is not None else 50.0 * guess)
    return limits
