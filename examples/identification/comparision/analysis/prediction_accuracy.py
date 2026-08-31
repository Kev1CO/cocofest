"""
How accurate a prediction is, and how wide an interval one has to draw around it to be right 95% of the time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import _shared

import matplotlib.pyplot as plt

ANALYSIS = "prediction_accuracy"


def interval(residuals) -> dict:
    """The 95% interval on one prediction, both ways, with the test that says which one to quote."""
    residuals = np.asarray(residuals, float)
    residuals = residuals[np.isfinite(residuals)]
    if len(residuals) < 3:
        return {}
    low, high = np.percentile(residuals, [2.5, 97.5])
    spread = float(residuals.std(ddof=1))
    normal = stats.shapiro(residuals) if len(residuals) <= 5000 else None
    return {
        "n": int(len(residuals)),
        "bias_pct": float(residuals.mean()),
        "sd_pct": spread,
        "pi95_empirical_low": float(low),
        "pi95_empirical_high": float(high),
        "pi95_gaussian_low": float(residuals.mean() - 1.96 * spread),
        "pi95_gaussian_high": float(residuals.mean() + 1.96 * spread),
        "shapiro_p": float(normal.pvalue) if normal is not None else np.nan,
        "quote": "empirical" if normal is not None and normal.pvalue < 0.05 else "gaussian",
    }


def accuracy(route: str) -> dict:
    """Every accuracy measure of one route, pooled over its testing trains and participants."""
    held_out = _shared.testing(route)
    if held_out.empty:
        return {}

    fits, errors = [], []
    for subject in _shared.subjects_of(route):
        _, measured, predicted = _shared.signal_of(route, subject)
        fits.append(_shared.best_fit_rate(measured, predicted))
        errors.append(_shared.normalised_rmse(measured, predicted))

    return {
        "route": route,
        "n_trains": int(len(held_out)),
        "n_subjects": len(_shared.subjects_of(route)),
        "peak_error_pct": float(held_out["peak_error"].mean()),
        "peak_error_sd_pct": float(held_out["peak_error"].std()),
        "fti_error_pct": float(held_out["fti_error"].mean()),
        "fti_error_sd_pct": float(held_out["fti_error"].std()),
        "r2": float(held_out["r2"].mean()),
        "r2_sd": float(held_out["r2"].std()),
        "bfr_pct": float(np.nanmean(fits)),
        "bfr_sd_pct": float(np.nanstd(fits, ddof=1)),
        "nrmse_pct": float(np.nanmean(errors)),
        "nrmse_sd_pct": float(np.nanstd(errors, ddof=1)),
    }


def table(routes) -> pd.DataFrame:
    return pd.DataFrame([block for block in (accuracy(route) for route in routes) if block])


def intervals(routes) -> pd.DataFrame:
    """The 95% prediction interval of every route and every scalar."""
    rows = []
    for route in routes:
        held_out = _shared.testing(route)
        if held_out.empty:
            continue
        for scalar in _shared.SCALARS:
            block = interval(held_out[f"{scalar}_signed_error"])
            if block:
                rows.append({"route": route, "scalar": scalar, **block})
    return pd.DataFrame(rows)


def per_subject(routes) -> pd.DataFrame:
    """The same accuracy one participant at a time, the frame the cohort figures box-plot."""
    rows = []
    for route in routes:
        held_out = _shared.testing(route)
        if held_out.empty:
            continue
        for subject, group in held_out.groupby("subject"):
            _, measured, predicted = _shared.signal_of(route, subject)
            rows.append({
                "route": route, "subject": subject, "n_trains": len(group),
                "peak_error_pct": float(group["peak_error"].mean()),
                "fti_error_pct": float(group["fti_error"].mean()),
                "r2": float(group["r2"].mean()),
                "bfr_pct": _shared.best_fit_rate(measured, predicted),
                "nrmse_pct": _shared.normalised_rmse(measured, predicted),
            })
    return pd.DataFrame(rows)


def figure(routes, name="prediction_accuracy") -> None:
    """The accuracy of every route side by side, and the residual distribution the interval comes from."""
    subjects = per_subject(routes)
    if subjects.empty:
        return
    order = [route for route in routes if route in set(subjects["route"])]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for axis, column, label in zip(axes[:2], ("bfr_pct", "peak_error_pct"),
                                   ("Best Fit Rate (%)", "peak %model error")):
        _shared.boxplot(axis,
                        [subjects.loc[subjects["route"] == route, column].dropna() for route in order],
                        [route.replace("_id", "") for route in order], showfliers=False)
        axis.axhline(0.0, color="0.6", lw=0.8)
        axis.set_ylabel(label, fontsize=8)
        axis.tick_params(axis="x", labelrotation=90, labelsize=6)
        axis.tick_params(axis="y", labelsize=7)

    axis = axes[2]
    for route in order:
        held_out = _shared.testing(route)
        axis.hist(held_out["peak_signed_error"], bins=25, histtype="step",
                  label=route.replace("_id", ""), density=True)
    axis.axvline(0.0, color="0.6", lw=0.8)
    axis.set_xlabel("signed peak error (%)", fontsize=8)
    axis.set_ylabel("density", fontsize=8)
    axis.legend(fontsize=5.5)
    axis.tick_params(labelsize=7)

    fig.tight_layout()
    _shared.save_figure(fig, ANALYSIS, name)


def main():
    _shared.headless()
    parser = _shared.arguments(__doc__.splitlines()[1], subjects=False)
    routes = _shared.available(parser.parse_args().routes)

    _shared.banner("prediction accuracy on the testing trains")
    cohort_table = table(routes)
    print(cohort_table.round(2).to_string(index=False))
    _shared.write(cohort_table, ANALYSIS, "cohort")

    _shared.banner("95% interval on one prediction")
    interval_table = intervals(routes)
    print(interval_table.round(1).to_string(index=False))
    _shared.write(interval_table, ANALYSIS, "prediction_intervals")

    _shared.write(per_subject(routes), ANALYSIS, "per_subject")
    figure(routes)


if __name__ == "__main__":
    main()
