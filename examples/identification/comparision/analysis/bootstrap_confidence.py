"""
Confidence intervals on the identified parameters and on the predictions, by resampling the training trains.

A subsampling bootstrap, m out of n without replacement, forced by the protocol: a train is a cell of the
grid and the procedure cannot be handed the same cell twice. A fit on m trains scatters more than one on
all n by roughly sqrt(n/m), so the standard error is scaled by sqrt(m/n); both that corrected interval and
the raw percentile one are written out.

The reference column is resampled too, since step 2 settles A, Km, tau1 and tau2 there alone and dropping
only the other trains would give most of the parameter set an interval of exactly zero width.
`--keep-reference 3` turns that off, leaving an interval on the eq. 3 parameters alone.

Two intervals come out of the same runs: on the parameters, and on the predicted peak and FTI of each
testing train, propagated from the parameter uncertainty rather than read off residuals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import _campaign
import _shared

import matplotlib.pyplot as plt

import protocol

ANALYSIS = "bootstrap_confidence"

DEFAULT_DRAWS = 30
DEFAULT_KEEP = 8  # of the 10 non-reference identification trains
DEFAULT_KEEP_REFERENCE = 2  # of the 3 reference trains, the ones step 2 alone reads
CONFIDENCE = 95


def run_subject(route, subject: str, draws: int, keep: int, keep_reference: int, seed: int) -> pd.DataFrame:
    """One participant re-identified `draws` times, each on a different subset of its training trains."""
    available = _campaign.cells(protocol.IDENTIFICATION)
    reference = _campaign.cells(protocol.REFERENCE)
    keep = min(keep, len(available))
    keep_reference = min(keep_reference, len(reference))
    generator = np.random.default_rng(seed)
    rows = []

    for draw in range(draws):
        chosen = set(generator.choice(len(available), size=keep, replace=False).tolist())
        dropped = [cell for i, cell in enumerate(available) if i not in chosen]
        # Step 2 only ever sees the reference column, so it is resampled too
        held = set(generator.choice(len(reference), size=keep_reference, replace=False).tolist())
        dropped += [cell for i, cell in enumerate(reference) if i not in held]

        design = _campaign.design_with(to_dropped=dropped)
        print(f"  P{subject} draw {draw + 1}/{draws} keeps {keep}/{len(available)} + "
              f"{keep_reference}/{len(reference)} reference [{_campaign.describe(design)}]", flush=True)

        outcome = _campaign.identify(route, subject, design=design)
        row = _campaign.flat(outcome, route=route.METHOD, draw=draw,
                             kept=keep + keep_reference, available=len(available) + len(reference))

        # The predicted scalars of every testing train, which the prediction interval is built on
        table = outcome.get("trains")
        if table is not None and len(table):
            held_out = table[table["set"] == "testing"]
            for _, train in held_out.iterrows():
                for scalar in _shared.SCALARS:
                    row[f"pred_{scalar}__{train['condition']}"] = float(train[f"{scalar}_predicted"])
                    row[f"meas_{scalar}__{train['condition']}"] = float(train[f"{scalar}_measured"])
        rows.append(row)
    return pd.DataFrame(rows)


def _interval(values: np.ndarray, correction: float) -> dict:
    """A percentile interval and the sqrt(m/n) corrected one, from the same draws."""
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if len(values) < 3:
        return {}
    low, high = np.percentile(values, [(100 - CONFIDENCE) / 2, 100 - (100 - CONFIDENCE) / 2])
    centre, spread = float(np.median(values)), float(values.std(ddof=1))
    corrected = spread * correction
    return {
        "n_draws": int(len(values)), "median": centre,
        "se_subsample": spread, "se_corrected": corrected,
        "ci_low": float(centre - 1.96 * corrected), "ci_high": float(centre + 1.96 * corrected),
        "percentile_low": float(low), "percentile_high": float(high),
        "cv_pct": float(100 * corrected / centre) if centre else np.nan,
    }


def parameter_intervals(detail: pd.DataFrame) -> pd.DataFrame:
    """The interval on every identified parameter, participant by participant."""
    rows = []
    for (route, subject), group in detail.groupby(["route", "subject"], sort=False):
        usable = group[group["failed"] == ""]
        if usable.empty:
            continue
        correction = np.sqrt(float(usable["kept"].iloc[0]) / float(usable["available"].iloc[0]))
        for name in _shared.PARAMETERS:
            if name not in usable:
                continue
            block = _interval(usable[name].to_numpy(float), correction)
            if block:
                rows.append({"route": route, "subject": subject, "parameter": name,
                             "correction": correction, **block})
    return pd.DataFrame(rows)


def prediction_intervals(detail: pd.DataFrame) -> pd.DataFrame:
    """The interval on every predicted scalar of every testing train."""
    rows = []
    for (route, subject), group in detail.groupby(["route", "subject"], sort=False):
        usable = group[group["failed"] == ""]
        if usable.empty:
            continue
        correction = np.sqrt(float(usable["kept"].iloc[0]) / float(usable["available"].iloc[0]))
        for column in [name for name in usable.columns if name.startswith("pred_")]:
            scalar, condition = column[len("pred_"):].split("__", 1)
            block = _interval(usable[column].to_numpy(float), correction)
            if not block:
                continue
            measured = usable.get(f"meas_{scalar}__{condition}")
            rows.append({"route": route, "subject": subject, "condition": condition, "scalar": scalar,
                         "measured": float(measured.median()) if measured is not None else np.nan,
                         **block})
    table = pd.DataFrame(rows)
    if not table.empty:
        # Whether the measurement falls inside the interval the model draws around its own prediction
        table["covers_measurement"] = ((table["measured"] >= table["percentile_low"])
                                       & (table["measured"] <= table["percentile_high"]))
    return table


def coverage(predictions: pd.DataFrame) -> pd.DataFrame:
    """How often the interval actually contains the measurement, which is what makes it a 95% interval."""
    if predictions.empty:
        return predictions
    return (predictions.groupby(["route", "scalar"], sort=False)
            .agg(n=("covers_measurement", "count"),
                 covered=("covers_measurement", "sum"),
                 coverage_pct=("covers_measurement", lambda values: 100 * float(np.mean(values))),
                 median_ci_width_pct=("cv_pct", "median"))
            .reset_index())


def figure(parameters: pd.DataFrame, name="bootstrap_confidence") -> None:
    """The width of every parameter's interval, as a share of its own value."""
    if parameters.empty:
        return
    routes = list(dict.fromkeys(parameters["route"]))
    fig, axis = plt.subplots(figsize=(1.8 * len(routes) + 4, 4.2))

    width = 0.8 / max(len(routes), 1)
    positions = np.arange(len(_shared.PARAMETERS))
    for index, route in enumerate(routes):
        group = parameters[parameters["route"] == route]
        heights = [group.loc[group["parameter"] == name_of, "cv_pct"].median()
                   for name_of in _shared.PARAMETERS]
        axis.bar(positions + index * width, heights, width, label=route.replace("_id", ""))

    axis.set_xticks(positions + 0.4 - width / 2)
    axis.set_xticklabels(_shared.PARAMETERS, rotation=45, ha="right", fontsize=7)
    axis.set_ylabel("bootstrap CV of the identified value (%)", fontsize=8)
    axis.legend(fontsize=6)
    axis.tick_params(axis="y", labelsize=7)

    fig.tight_layout()
    _shared.save_figure(fig, ANALYSIS, name)


def main():
    _shared.headless()
    parser = _shared.arguments(__doc__.splitlines()[1], routes=("sensor_id",))
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS, help="re-identifications per participant")
    parser.add_argument("--keep", type=int, default=DEFAULT_KEEP,
                        help="how many of the ten non-reference identification trains each draw keeps")
    parser.add_argument("--keep-reference", type=int, default=DEFAULT_KEEP_REFERENCE,
                        help="how many of the three reference trains each draw keeps; 3 leaves the "
                             "reference column whole and then only a_scale, pd0 and pdt get an interval")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--refresh", action="store_true", help="re-run participants already on disk")
    arguments = parser.parse_args()

    print(f"{arguments.draws} draws keeping {arguments.keep} + {arguments.keep_reference} reference")
    detail = _campaign.per_subject(
        ANALYSIS, arguments,
        lambda route, subject: run_subject(route, subject, arguments.draws, arguments.keep,
                                           arguments.keep_reference, arguments.seed),
    )
    if detail.empty:
        return
    parameters = parameter_intervals(detail)
    predictions = prediction_intervals(detail)

    _shared.banner("confidence intervals on the identified parameters")
    print(parameters.round(5).to_string(index=False))
    _shared.banner("coverage of the 95% prediction intervals")
    print(coverage(predictions).round(1).to_string(index=False))

    _shared.write(detail, ANALYSIS, "all_draws")
    _shared.write(parameters, ANALYSIS, "parameter_intervals")
    _shared.write(predictions, ANALYSIS, "prediction_intervals")
    _shared.write(coverage(predictions), ANALYSIS, "coverage")
    figure(parameters)


if __name__ == "__main__":
    main()
