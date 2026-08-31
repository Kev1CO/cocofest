"""
The outcome measures given in the Ding et al. (2007) article.

The peak (PK) and the force-time integral (FTI), plus an R2 on the shape, a %model error on each scalar, and an
agreement block of ICC(A,1) with a zero-intercept trendline and a paired t-test.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pingouin as pg
from scipy import stats
from statsmodels.stats.multitest import multipletests

import store


pg.options["round.column.CI95"] = None


# ---- The two scalars every train is scored on ---- #
def peak(signal, baseline=0.0) -> float:
    """PK: the largest excursion of the signal above its resting value."""
    return float(np.max(np.asarray(signal, dtype=float) - baseline))


def time_integral(time, signal, baseline=0.0) -> float:
    """FTI: the area under the signal above its resting value, over the analysed window."""
    return float(np.trapezoid(np.asarray(signal, dtype=float) - baseline, np.asarray(time, dtype=float)))


def r_squared(measured, predicted) -> float:
    """
    predicted R2: the share of the measured trace's variance the prediction accounts for.
    The coefficient of determination, not a squared correlation, so a prediction of the right shape and the
    wrong size is penalized.
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
    """The same keeping its sign, which is what a Bland-Altman style reading of it needs."""
    measured, predicted = float(measured), float(predicted)
    return (predicted - measured) / abs(measured) * 100 if measured != 0 else np.nan


# ---- Agreement between the measured and the predicted scalars ---- #
def zero_intercept_slope(measured, predicted) -> float:
    """The slope of the article's trendline through the origin, so it alone carries the whole bias."""
    measured, predicted = np.asarray(measured, dtype=float), np.asarray(predicted, dtype=float)
    denominator = float(np.sum(measured**2))
    return float(np.sum(measured * predicted) / denominator) if denominator > 0 else np.nan


def icc_a1(measured, predicted) -> tuple:
    """
    ICC(A,1) and its 95% interval: two-way, absolute agreement, single measure (McGraw & Wong, 1996).

    The absolute agreement form rather than the consistency one, which ignores a systematic offset between
    the two ratings. `pingouin` wants the two ratings stacked long, one row per rating.
    """
    scores = np.column_stack([measured, predicted])
    n, k = scores.shape
    ratings = pd.DataFrame({
        "target": np.tile(np.arange(n), k),
        "rater": np.repeat(np.arange(k), n),
        "score": scores.ravel(order="F"),
    })
    # A perfect or a negative ICC leaves the interval undefined, which pingouin reaches by dividing by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        table = pg.intraclass_corr(data=ratings, targets="target", raters="rater", ratings="score")
    row = table.loc[table["Type"] == "ICC(A,1)"].iloc[0]
    return float(row["ICC"]), tuple(float(bound) for bound in row["CI95"])


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

    # Neither an ICC nor a paired t-test means anything on two points, and the keys have to match anyway
    if len(measured) < 3:
        return {"n": int(len(measured)), "icc": np.nan, "icc_ci": (np.nan, np.nan), "slope": np.nan,
                "model_error": np.nan, "model_error_sd": np.nan, "signed_model_error": np.nan,
                "t": np.nan, "p": np.nan}

    icc, icc_ci = icc_a1(measured, predicted)
    test = stats.ttest_rel(predicted, measured)
    errors = [model_error(m, p) for m, p in zip(measured, predicted)]
    signed = [signed_model_error(m, p) for m, p in zip(measured, predicted)]
    return {
        "n": int(len(measured)),
        "icc": icc,
        "icc_ci": icc_ci,
        "slope": zero_intercept_slope(measured, predicted),
        "model_error": float(np.nanmean(errors)),
        "model_error_sd": float(np.nanstd(errors, ddof=1)),
        "signed_model_error": float(np.nanmean(signed)),
        "t": float(test.statistic),
        "p": float(test.pvalue),
    }


# ---- One train, scored ---- #
def score_train(time, measured, predicted, baseline=0.0) -> dict:
    """Every measure the article reports for a single train, scored above `baseline`."""
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
    """Elapsed seconds between laps, one lap per step, so every route lists the same five keys."""

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
        table["p_holm"] = multipletests(table["p"].fillna(1.0).to_numpy(), method="holm")[1]
        table["agrees"] = (table["p_holm"] < 0.05) & (table["spearman_rho"] > 0)
    return table


# ---- Pooling the trains, the way the article's tables do ---- #
def summarise(table: pd.DataFrame, by=("pulse_width_us",), only_testing=True) -> pd.DataFrame:
    """
    `table` holds one row per train as `score_train` builds them, with "subject", "frequency",
    "pulse_width_us" and a "set" column; `by` is the condition to pool over, one row out per level of it.
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
        row["r2_median"] = float(group["r2"].median())
        if "physiological_r2" in group and group["physiological_r2"].notna().any():
            beats = group["r2"] > group["physiological_r2"]
            row["percent_above_physiological_r2"] = 100 * float(beats.mean())
        else:
            row["percent_above_physiological_r2"] = np.nan
        rows.append(row)

    summary = pd.DataFrame(rows)
    for measure in ("peak", "fti"):
        column = f"{measure}_p"
        if column in summary and summary[column].notna().any():
            summary[f"{measure}_p_holm"] = multipletests(summary[column].fillna(1.0).to_numpy(), method="holm")[1]
    return summary
