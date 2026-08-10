"""
Cross validation of an identification.
"""

from __future__ import annotations

import numpy as np

DEFAULT_FOLDS = 5


def _conditions_array(conditions, key):
    return np.array([float(c[key]) for c in conditions])


def k_fold(conditions, folds=DEFAULT_FOLDS, seed=0):
    """
    Stratified k-fold split of the phases.

    Parameters
    ----------
    conditions: list[dict]
        One entry per phase, with "frequency_hz" and "pulse_width_us"
    folds: int | str
        The number of folds, or "loo" for leave one out (one fold per phase)
    seed: int
        Shuffling seed, so a run can be repeated exactly

    Returns
    -------
    list[tuple[list, list]]
        One (train indices, test indices) pair per fold
    """
    n_phases = len(conditions)
    if folds == "loo":
        return [([j for j in range(n_phases) if j != i], [i]) for i in range(n_phases)]

    folds = int(folds)
    if not 2 <= folds <= n_phases:
        raise ValueError(f"folds must be between 2 and the number of phases ({n_phases}), or 'loo'.")

    # Stratify on the stimulation frequency: every fold then holds out phases spread across the conditions
    rng = np.random.default_rng(seed)
    assignment = np.empty(n_phases, dtype=int)
    for frequency in np.unique(_conditions_array(conditions, "frequency_hz")):
        indices = np.where(_conditions_array(conditions, "frequency_hz") == frequency)[0]
        rng.shuffle(indices)
        assignment[indices] = np.arange(len(indices)) % folds

    return [
        ([j for j in range(n_phases) if assignment[j] != fold], [j for j in range(n_phases) if assignment[j] == fold])
        for fold in range(folds)
    ]


def leave_one_condition_out(conditions, by="pulse_width_us"):
    """
    Hold out one whole stimulation condition at a time, all its phases together.

    Returns
    -------
    list[tuple[list, list, float]]
        One (train indices, test indices, held out value) triple per condition
    """
    values = _conditions_array(conditions, by)
    return [
        ([j for j in range(len(conditions)) if values[j] != value], [j for j in range(len(conditions)) if values[j] == value], float(value))
        for value in np.unique(values)
    ]


def summarise(errors) -> dict:
    """
    The numbers a cross validation is reported with: the mean prediction error over the folds and its spread.

    """
    errors = np.asarray([e for e in errors if np.isfinite(e)], dtype=float)
    if errors.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "n": int(errors.size),
        "mean": float(errors.mean()),
        "std": float(errors.std(ddof=1)) if errors.size > 1 else 0.0,
        "min": float(errors.min()),
        "max": float(errors.max()),
    }
