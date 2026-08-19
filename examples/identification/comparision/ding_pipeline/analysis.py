"""
The outcome measures and the statistics Ding et al. (2007) report, for either route.

The article never scores a prediction on its rmse. It scores two scalars per train, the peak (PK) and the
force-time integral (FTI), and it scores the shape separately with an R2:

    "%modelerror = |predicted - measured| / measured x 100"
    "%physiological variability = |measured1 - measured2| / average(measured1, measured2) x 100"
    "To evaluate how well the model predicted the shape of the force response, R2 values (predicted R2) were
     calculated for each testing train between the predicted force and the averaged measured force."
    "Intraclass correlation coefficients (ICCs) between the predicted and measured data were calculated for
     FTIs and PKs."
    "Trendlines with intercepts of zero were used to evaluate agreement between the measured and predicted
     values. A slope of 1 and ICC value of 1 would represent a perfect prediction of the model."
    "If the percentage difference in FTI or PK between the measured and predicted forces (%model error) was
     larger than the % physiological variability, a paired t-test was used to determine whether there was a
     significant difference between the predicted and the measured forces for that testing train."

Everything here is written on a generic "signal", so the movement route can hand it elbow angle instead of
force. The peak then becomes the peak excursion from the starting angle and the FTI becomes the angle-time
integral, which are the two scalars that carry the same meaning for a movement.

The one measure this cohort cannot support is the physiological variability, and with it the article's rule
for when to run the paired t-test: each (frequency, pulse width) condition was recorded once per subject, so
there is no second occurrence to difference against. `physiological_variability` is written anyway, for data
that does hold repeats; when there are none the pipeline records nan and runs the paired t-tests
unconditionally, which is the more conservative of the two readings.
"""

from __future__ import annotations

import sys
from pathlib import Path

import time

import numpy as np
import pandas as pd
from scipy import stats

COMPARISON_ROOT = Path(__file__).resolve().parent.parent
for _root in (COMPARISON_ROOT / "graphs",):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

import stats_tools  # noqa: E402


# ---- Reading a profile of the cost against one parameter ---- #
NEGLIGIBLE_SPREAD = 0.05


def interior_minimum(curve, negligible_spread=NEGLIGIBLE_SPREAD) -> dict:
    """
    Whether a cost profile pins its parameter down, and where.

    Two things have to hold, and the second is what tells a real minimum from a drift. The cost has to move
    by more than `negligible_spread` across the grid, and its minimum has to lie *inside* it. A cost falling
    monotonically onto the lowest value tried has not found an optimum, it is sliding, and widening the grid
    only moves the answer again.

    Parameters
    ----------
    curve: pd.Series
        The cost against the parameter value, indexed by that value and normalised however the caller likes

    Returns
    -------
    dict
        best, cost_spread, interior, and identifiable, which is the two conditions together
    """
    if curve.empty:
        return {"best": np.nan, "cost_spread": np.nan, "interior": False, "identifiable": False}

    spread = float(curve.max() - curve.min())
    best = float(curve.idxmin())
    interior = bool(len(curve) > 2 and best not in (curve.index.min(), curve.index.max()))
    return {
        "best": best,
        "cost_spread": spread,
        "interior": interior,
        "identifiable": bool(spread > negligible_spread and interior),
    }


# ---- The two scalars every train is scored on ---- #
def peak(signal, baseline=0.0) -> float:
    """PK: the largest excursion of the signal above its resting value."""
    return float(np.max(np.asarray(signal, dtype=float) - baseline))


def time_integral(time, signal, baseline=0.0) -> float:
    """FTI: the area under the signal above its resting value, over the analysed window."""
    return float(np.trapezoid(np.asarray(signal, dtype=float) - baseline, np.asarray(time, dtype=float)))


def r_squared(measured, predicted) -> float:
    """
    The article's "predicted R2": how much of the measured trace's variance the prediction accounts for.

    The coefficient of determination, not a squared correlation: a prediction that is the right shape but
    the wrong size has to be penalised, and only this form does that.
    """
    measured, predicted = np.asarray(measured, dtype=float), np.asarray(predicted, dtype=float)
    spread = float(np.sum((measured - measured.mean()) ** 2))
    if spread == 0:
        return np.nan
    return float(1 - np.sum((measured - predicted) ** 2) / spread)


# ---- The two percentages ---- #
def model_error(measured, predicted) -> float:
    """%model error, the unsigned relative distance between a predicted scalar and the measured one."""
    measured, predicted = float(measured), float(predicted)
    return abs(predicted - measured) / abs(measured) * 100 if measured != 0 else np.nan


def signed_model_error(measured, predicted) -> float:
    """
    The same, keeping its sign, so a systematic over or under prediction can be told from a scattered one.

    The article reports the unsigned form; the sign is what a Bland-Altman style reading of it needs.
    """
    measured, predicted = float(measured), float(predicted)
    return (predicted - measured) / abs(measured) * 100 if measured != 0 else np.nan


def physiological_variability(first, second) -> float:
    """%physiological variability between the two occurrences of the same train."""
    first, second = float(first), float(second)
    average = 0.5 * (first + second)
    return abs(first - second) / abs(average) * 100 if average != 0 else np.nan


# ---- Agreement between the measured and the predicted scalars ---- #
def zero_intercept_slope(measured, predicted) -> float:
    """
    The slope of the article's trendline through the origin, "a slope of 1 [...] a perfect prediction".

    Least squares with the intercept held at zero, so the slope alone carries the whole bias.
    """
    measured, predicted = np.asarray(measured, dtype=float), np.asarray(predicted, dtype=float)
    denominator = float(np.sum(measured**2))
    return float(np.sum(measured * predicted) / denominator) if denominator > 0 else np.nan


def agreement(measured, predicted) -> dict:
    """
    The article's agreement block for one set of trains: ICC, zero-intercept slope and a paired t-test.

    Returns
    -------
    dict
        icc with its CI, the trendline slope, the mean unsigned and signed %model error, and the paired
        t-test of predicted against measured
    """
    measured = np.asarray(measured, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    usable = np.isfinite(measured) & np.isfinite(predicted)
    measured, predicted = measured[usable], predicted[usable]

    # Neither an ICC nor a paired t-test means anything on two points, and reporting one would be worse
    # than reporting none. A single subject's report reaches this on every intensity, so it has to hold the
    # same keys as the full return.
    if len(measured) < 3:
        return {"n": int(len(measured)), "icc": np.nan, "icc_ci": (np.nan, np.nan), "slope": np.nan,
                "model_error": np.nan, "model_error_sd": np.nan, "signed_model_error": np.nan,
                "t": np.nan, "p": np.nan}

    icc = stats_tools.icc_a1(measured, predicted)
    test = stats.ttest_rel(predicted, measured)
    errors = [model_error(m, p) for m, p in zip(measured, predicted)]
    signed = [signed_model_error(m, p) for m, p in zip(measured, predicted)]
    return {
        "n": int(len(measured)),
        "icc": icc["icc"],
        "icc_ci": (icc["ci_low"], icc["ci_high"]),
        "slope": zero_intercept_slope(measured, predicted),
        "model_error": float(np.nanmean(errors)),
        "model_error_sd": float(np.nanstd(errors, ddof=1)),
        "signed_model_error": float(np.nanmean(signed)),
        "t": float(test.statistic),
        "p": float(test.pvalue),
    }


# ---- One train, scored ---- #
def score_train(time, measured, predicted, baseline=0.0) -> dict:
    """
    Every measure the article reports for a single train.

    Parameters
    ----------
    time: array
        The analysis grid (s), shared by the two signals
    measured, predicted: array
        The measured signal and the model's, read on that same grid
    baseline: float
        The resting value the peak and the integral are taken above
    """
    measured, predicted = np.asarray(measured, dtype=float), np.asarray(predicted, dtype=float)
    peak_measured, peak_predicted = peak(measured, baseline), peak(predicted, baseline)
    fti_measured = time_integral(time, measured, baseline)
    fti_predicted = time_integral(time, predicted, baseline)
    return {
        "peak_measured": peak_measured,
        "peak_predicted": peak_predicted,
        "peak_error": model_error(peak_measured, peak_predicted),
        "peak_signed_error": signed_model_error(peak_measured, peak_predicted),
        "fti_measured": fti_measured,
        "fti_predicted": fti_predicted,
        "fti_error": model_error(fti_measured, fti_predicted),
        "fti_signed_error": signed_model_error(fti_measured, fti_predicted),
        "r2": r_squared(measured, predicted),
        "rmse": float(np.sqrt(np.mean((predicted - measured) ** 2))),
        "duration": float(time[-1] - time[0]),
    }


class Stopwatch:
    """
    Elapsed seconds between laps, one lap per step.

    A lap rather than a wrapper, so timing a step costs one line at the end of it and never changes how the
    step itself is written. Both routes lap the same five keys, so what a step costs is comparable between
    them - and the movement route pays it on the full musculoskeletal model.
    """

    def __init__(self):
        self._mark = time.perf_counter()
        self.laps = {}

    def lap(self, key) -> float:
        """Close the current step under `key` and start the next one."""
        now = time.perf_counter()
        self.laps[key] = now - self._mark
        self._mark = now
        return self.laps[key]

    def total(self) -> float:
        """Every step so far, in seconds."""
        return float(sum(self.laps.values()))


def report_prediction(table, label="step 5") -> str:
    """
    The headline line of one subject's prediction, with the median beside every mean.

    R2 is unbounded below, so over a handful of trains one the model gets badly wrong takes the mean
    anywhere while the other trains were fine. The median says what the typical train did, and the worst
    train is named outright rather than left to be inferred from the gap between the two.

    Parameters
    ----------
    table: pd.DataFrame
        The testing rows of one subject, as `score_train` builds them
    label: str
        How the line starts, ex: "step 5"
    """
    if table.empty:
        return f"  {label}: no testing train to score"

    worst = table.loc[table["r2"].idxmin()]
    line = (f"  {label}: {len(table)} testing train(s), "
            f"PK error {table['peak_error'].mean():.1f} % (median {table['peak_error'].median():.1f}), "
            f"FTI error {table['fti_error'].mean():.1f} % (median {table['fti_error'].median():.1f}), "
            f"R2 {table['r2'].mean():.3f} (median {table['r2'].median():.3f})")
    if len(table) > 1 and worst["r2"] < table["r2"].median() - 0.5:
        line += (f"\n          worst train {worst['condition']}: R2 {worst['r2']:.3f}, "
                 f"peak error {worst['peak_error']:.1f} % - it alone moves the mean")
    return line


def paired_agreement(first, second, names=None) -> pd.DataFrame:
    """
    Whether two routes rank the participants the same way, parameter by parameter.

    The question a group median cannot answer. Two routes can agree on the cohort centre and still disagree
    about every individual, and only a paired statistic separates the two: a Spearman rho near zero says the
    participant a route calls stiff is not the one the other route calls stiff, whatever the medians do.
    Rank rather than Pearson, because these parameters are skewed and a single outlier would otherwise
    carry the correlation.

    Parameters
    ----------
    first, second: dict
        Parameter name to a per participant array, the two routes, aligned on the same participants
    names: list
        Which parameters to test, all the shared ones otherwise

    Returns
    -------
    pd.DataFrame
        One row per parameter: n, the two medians, their ratio, rho, its p value, and the Holm corrected p
    """
    names = list(names if names is not None else sorted(set(first) & set(second)))

    rows = []
    for name in names:
        left = np.asarray(first[name], dtype=float)
        right = np.asarray(second[name], dtype=float)
        usable = np.isfinite(left) & np.isfinite(right)
        if usable.sum() < 3:
            continue
        rho, p = stats.spearmanr(left[usable], right[usable])
        rows.append({
            "parameter": name,
            "n": int(usable.sum()),
            "first_median": float(np.median(left[usable])),
            "second_median": float(np.median(right[usable])),
            "ratio_median": float(np.median(right[usable] / left[usable]))
            if np.all(left[usable] != 0) else np.nan,
            "spearman_rho": float(rho),
            "p": float(p),
        })

    table = pd.DataFrame(rows)
    if not table.empty:
        # One family of tests, corrected together rather than one by one
        table["p_holm"] = stats_tools.holm(table["p"].fillna(1.0).to_numpy())
        table["agrees"] = (table["p_holm"] < 0.05) & (table["spearman_rho"] > 0)
    return table


# ---- Pooling the trains, the way the article's tables do ---- #
def summarise(table: pd.DataFrame, by=("pulse_width_us",), only_testing=True) -> pd.DataFrame:
    """
    The article's Table 3 and Table 4: the errors and the agreement, pooled over a stimulation condition.

    Parameters
    ----------
    table: pd.DataFrame
        One row per train, as `score_train` builds them, with "subject", "frequency", "pulse_width_us" and
        a "set" column holding "identification" or "testing"
    by: tuple
        Which condition to pool over, ex: ("pulse_width_us",) as the article's Table 3, or ("frequency",)
    only_testing: bool
        Whether the identification trains are excluded, as they are in the article's tables

    Returns
    -------
    pd.DataFrame
        One row per level of the pooling condition
    """
    table = table[table["set"] == "testing"] if only_testing and "set" in table else table
    by = list(by)

    rows = []
    for level, group in (table.groupby(by) if by else [((), table)]):
        level = level if isinstance(level, tuple) else (level,)
        row = dict(zip(by, level))
        row["n_trains"] = len(group)
        row["n_subjects"] = int(group["subject"].nunique()) if "subject" in group else np.nan

        for measure in ("peak", "fti"):
            scored = agreement(group[f"{measure}_measured"], group[f"{measure}_predicted"])
            row[f"{measure}_error"] = scored["model_error"]
            row[f"{measure}_error_sd"] = scored["model_error_sd"]
            row[f"{measure}_signed_error"] = scored["signed_model_error"]
            row[f"{measure}_icc"] = scored["icc"]
            row[f"{measure}_slope"] = scored["slope"]
            row[f"{measure}_p"] = scored["p"]

        row["r2"] = float(group["r2"].mean())
        row["r2_sd"] = float(group["r2"].std(ddof=1)) if len(group) > 1 else np.nan
        # R2 is unbounded below, so one train the model gets badly wrong drags the mean anywhere. The median
        # is what says whether the typical train was well predicted; both are reported, neither alone.
        row["r2_median"] = float(group["r2"].median())
        # The article's headline count: how often the prediction beats the physiological reference band
        if "physiological_r2" in group and group["physiological_r2"].notna().any():
            beats = group["r2"] > group["physiological_r2"]
            row["percent_above_physiological_r2"] = 100 * float(beats.mean())
        else:
            row["percent_above_physiological_r2"] = np.nan
        rows.append(row)

    summary = pd.DataFrame(rows)
    # The tests are one family per measure, so they are corrected together rather than one by one
    for measure in ("peak", "fti"):
        column = f"{measure}_p"
        if column in summary and summary[column].notna().any():
            summary[f"{measure}_p_holm"] = stats_tools.holm(summary[column].fillna(1.0).to_numpy())
    return summary
