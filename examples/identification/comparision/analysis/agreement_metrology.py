"""
How well a prediction agrees with the measurement, and how small a change this model could ever detect.

Four measures, on the peak and the force-time integral of the testing trains of every route:

    ICC(A,1)   absolute agreement, read out loud with the thresholds of Koo & Li (2016): below 0.50 poor,
               0.50-0.75 moderate, 0.75-0.90 good, above 0.90 excellent
    SEM        the standard error of measurement, SD sqrt(1 - ICC), Weir (2005)
    MDC95      the minimal detectable change, 1.96 sqrt(2) SEM, the smallest individual change this model
               can claim is real rather than noise
    Bland      the bias and the 95% limits of agreement of Bland & Altman (1986), with the regression of
               the difference on the mean that says whether the bias grows with the amplitude

An ICC alone flatters a spread-out cohort, since it rises with the between-subject variance; the SEM and
the MDC do not, which is why the three are reported together and never one without the others.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import _figures
import _shared

import matplotlib.pyplot as plt

import metrics

ANALYSIS = "agreement_metrology"


def metrology(measured, predicted) -> dict:
    """The agreement block of one route and one scalar, as the tables and the figure both read it."""
    measured = np.asarray(measured, float)
    predicted = np.asarray(predicted, float)
    usable = np.isfinite(measured) & np.isfinite(predicted)
    measured, predicted = measured[usable], predicted[usable]
    if len(measured) < 3:
        return {}

    block = metrics.agreement(measured, predicted)
    spread = float(np.std(np.concatenate([measured, predicted]), ddof=1))
    sem = spread * np.sqrt(max(0.0, 1 - block["icc"]))
    mdc = 1.96 * np.sqrt(2) * sem

    # Bland & Altman: the difference against the average of the two, and whether that difference drifts
    difference = predicted - measured
    average = 0.5 * (predicted + measured)
    drift = stats.linregress(average, difference)

    return {
        "n": block["n"],
        "icc": block["icc"],
        "icc_low": block["icc_ci"][0],
        "icc_high": block["icc_ci"][1],
        "icc_grade": _shared.grade_icc(block["icc"]),
        "slope_zero_intercept": block["slope"],
        "model_error_pct": block["model_error"],
        "model_error_sd_pct": block["model_error_sd"],
        "signed_error_pct": block["signed_model_error"],
        "paired_t": block["t"],
        "paired_p": block["p"],
        "sem": sem,
        "sem_pct": 100 * sem / np.mean(measured),
        "mdc95": mdc,
        "mdc95_pct": 100 * mdc / np.mean(measured),
        "bias": float(difference.mean()),
        "loa_low": float(difference.mean() - 1.96 * difference.std(ddof=1)),
        "loa_high": float(difference.mean() + 1.96 * difference.std(ddof=1)),
        "loa_halfwidth_pct": float(100 * 1.96 * difference.std(ddof=1) / np.mean(measured)),
        "proportional_bias_slope": float(drift.slope),
        "proportional_bias_p": float(drift.pvalue),
    }


def table(routes) -> pd.DataFrame:
    """One row per route and per scalar."""
    rows = []
    for route in routes:
        held_out = _shared.testing(route)
        if held_out.empty:
            continue
        unit = _shared.cohort(route)["unit"].iloc[0]
        for scalar in _shared.SCALARS:
            block = metrology(held_out[f"{scalar}_measured"], held_out[f"{scalar}_predicted"])
            if block:
                rows.append({"route": route, "scalar": scalar,
                             "unit": unit if scalar == "peak" else f"{unit}.s", **block})
    return pd.DataFrame(rows)


def per_subject(routes) -> pd.DataFrame:
    """The same block one participant at a time, which is where a single odd participant shows up."""
    rows = []
    for route in routes:
        held_out = _shared.testing(route)
        if held_out.empty:
            continue
        for subject, group in held_out.groupby("subject"):
            for scalar in _shared.SCALARS:
                block = metrology(group[f"{scalar}_measured"], group[f"{scalar}_predicted"])
                if block:
                    rows.append({"route": route, "subject": subject, "scalar": scalar, **block})
    return pd.DataFrame(rows)


def relative(measured, predicted) -> dict:
    """The same block on the relative difference, which is what a proportional bias asks for."""
    measured, predicted = np.asarray(measured, float), np.asarray(predicted, float)
    average = 0.5 * (predicted + measured)
    usable = np.isfinite(average) & (average != 0)
    difference = 100 * (predicted - measured)[usable] / average[usable]
    spread = float(difference.std(ddof=1))
    return {"average": average[usable], "difference": difference, "bias": float(difference.mean()),
            "loa_low": float(difference.mean() - 1.96 * spread),
            "loa_high": float(difference.mean() + 1.96 * spread)}


def absolute(measured, predicted) -> dict:
    """The difference in the route's own unit, the usual Bland-Altman."""
    measured, predicted = np.asarray(measured, float), np.asarray(predicted, float)
    block = metrology(measured, predicted)
    return {"average": 0.5 * (predicted + measured), "difference": predicted - measured,
            "bias": block["bias"], "loa_low": block["loa_low"], "loa_high": block["loa_high"]}


def figure(routes, name="bland_altman") -> None:
    """
    Bland-Altman on the peak and the time integral, laid out like the measured-against-predicted figure.
    """
    routes = [route for route in _figures.ordered(routes) if not _shared.testing(route).empty]
    if not routes:
        return

    fig, axes = plt.subplots(2, len(routes), figsize=(5.2 * len(routes), 8.0), squeeze=False)
    for column, route in enumerate(routes):
        held_out = _shared.testing(route)
        unit = _shared.cohort(route)["unit"].iloc[0]
        for row, scalar in enumerate(_shared.SCALARS):
            axis = axes[row][column]
            measured = held_out[f"{scalar}_measured"].to_numpy(float)
            predicted = held_out[f"{scalar}_predicted"].to_numpy(float)
            block = metrology(measured, predicted)
            proportional = block["proportional_bias_p"] < 0.05
            drawn = relative(measured, predicted) if proportional else absolute(measured, predicted)
            told = "%" if proportional else (unit if scalar == "peak" else f"{unit} s")

            for _, group in held_out.groupby("subject"):
                one = (relative if proportional else absolute)(group[f"{scalar}_measured"],
                                                               group[f"{scalar}_predicted"])
                order = np.argsort(one["average"])
                axis.plot(np.asarray(one["average"])[order], np.asarray(one["difference"])[order],
                          color="0.6", lw=0.5, alpha=0.6, zorder=1)
            axis.scatter(drawn["average"], drawn["difference"], s=13, zorder=2, edgecolor="none",
                         color=_figures.colour(route), alpha=0.75)

            # The confidence band of the bias itself, which is not the limits of agreement
            n = len(drawn["difference"])
            half = 1.96 * float(np.std(drawn["difference"], ddof=1)) / np.sqrt(max(n, 1))
            axis.axhspan(drawn["bias"] - half, drawn["bias"] + half, color="0.6", alpha=0.20, zorder=0)
            axis.axhline(0.0, color="0.6", lw=0.8)
            axis.axhline(drawn["bias"], color=_figures.colour(route), lw=1.4)
            for limit in (drawn["loa_low"], drawn["loa_high"]):
                axis.axhline(limit, color=_figures.colour(route), lw=1.0, ls="--")

            if row == 0:
                axis.set_title(_figures.label(route), fontsize=10, fontweight="bold")
            axis.set_xlabel(f"mean of measured and predicted "
                            f"({unit if scalar == 'peak' else f'{unit} s'})", fontsize=8)
            axis.set_ylabel(f"predicted - measured ({told})", fontsize=8)
            axis.tick_params(labelsize=7)
            axis.grid(alpha=0.3)
            _figures.note(axis,
                          f"{scalar.upper()}   ICC {block['icc']:.2f} ({block['icc_grade']})\n"
                          f"bias {drawn['bias']:.2f}, LoA [{drawn['loa_low']:.2f}, "
                          f"{drawn['loa_high']:.2f}] {told}\n"
                          f"proportional bias p {block['proportional_bias_p']:.3f} -> "
                          f"{'relative' if proportional else 'absolute'} scale\n"
                          f"SEM {block['sem_pct']:.0f} %   MDC95 {block['mdc95_pct']:.0f} % "
                          f"(of the mean measured)", size=6)
            _figures.count(axis, f"{n} testing trains, {held_out['subject'].nunique()} participants")

    margin = table([_figures.REFERENCE])
    quoted = (f"The MDC95 of {_figures.REFERENCE} - "
              + ", ".join(f"{row['scalar']} {row['mdc95_pct']:.0f} %" for _, row in margin.iterrows())
              + " - is the equivalence margin the paired comparison uses; it is computed here and nowhere "
                "else." if not margin.empty else "")
    fig.suptitle("Bland-Altman: Data the model identification did not train on", fontsize=13, fontweight="bold")
    _figures.caption(fig, "SEM and MDC95 are normalised by the mean of the measurement, for comparison "
                          "between identification methods.\n"
                          "Thin lines join the trains of one participant. Shaded: the confidence band of "
                          "the bias, not the limits of agreement.\n" + quoted)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    _shared.save_figure(fig, ANALYSIS, name)


def main():
    _shared.headless()
    parser = _shared.arguments(__doc__.splitlines()[1], subjects=False)
    routes = _shared.available(parser.parse_args().routes)

    _shared.banner("agreement and metrology on the testing trains")
    cohort_table = table(routes)
    print(cohort_table[["route", "scalar", "n", "icc", "icc_grade", "sem_pct", "mdc95_pct",
                        "bias", "loa_low", "loa_high", "proportional_bias_p"]].to_string(index=False))
    _shared.write(cohort_table, ANALYSIS, "cohort")
    _shared.write(per_subject(routes), ANALYSIS, "per_subject")
    figure(routes)


if __name__ == "__main__":
    main()
