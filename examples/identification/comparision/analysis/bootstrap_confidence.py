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
import _figures
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


def figure(parameters: pd.DataFrame, covered=None, name="bootstrap_confidence") -> None:
    """
    The width of the intervals, and the number that decides whether they are intervals at all.

    The coverage panel is deliberately the large one: a 95 % interval that holds the measurement 60 % of
    the time is not a 95 % interval, however narrow it is.
    """
    if parameters.empty:
        return
    routes = _figures.ordered(list(dict.fromkeys(parameters["route"])))
    fig, axes = plt.subplots(1, 2, figsize=(8.0 + 1.6 * len(routes), 5.0),
                             gridspec_kw={"width_ratios": [1.3, 1]})

    axis = axes[0]
    width = 0.8 / max(len(routes), 1)
    positions = np.arange(len(_shared.PARAMETERS))
    for index, route in enumerate(routes):
        group = parameters[parameters["route"] == route]
        heights = [group.loc[group["parameter"] == name_of, "cv_pct"].median()
                   for name_of in _shared.PARAMETERS]
        axis.bar(positions + index * width, heights, width, color=_figures.colour(route), alpha=0.8,
                 label=_figures.label(route))
    axis.set_xticks(positions + 0.4 - width / 2)
    axis.set_xticklabels(_shared.PARAMETERS, rotation=45, ha="right", fontsize=7)
    axis.set_ylabel("bootstrap CV of the identified value (%)", fontsize=8)
    axis.set_title("(a) how wide, parameter by parameter", fontsize=10, fontweight="bold", loc="left")
    axis.legend(fontsize=6)
    axis.tick_params(axis="y", labelsize=7)
    _figures.note(axis, "drawn from ci_low/ci_high, the interval corrected by sqrt(m/n) because the\n"
                        "bootstrap is a subsampling without replacement. percentile_low/high, the raw\n"
                        "one, is on disk too and is what the coverage of panel (b) is read on.", size=5.5)

    axis = axes[1]
    covered = pd.DataFrame() if covered is None else covered
    if covered.empty:
        axis.axis("off")
        axis.text(0.5, 0.5, "no coverage on disk yet", ha="center", va="center", fontsize=9,
                  color="0.4")
    else:
        labels = []
        for position, row in enumerate(covered.itertuples()):
            axis.bar(position, row.coverage_pct, width=0.6, color=_figures.colour(row.route), alpha=0.8)
            axis.text(position, row.coverage_pct + 1, f"{row.coverage_pct:.0f} %", ha="center",
                      fontsize=8)
            labels.append(f"{_figures.label(row.route)}\n{row.scalar}")
        axis.axhline(95.0, color=_figures.THRESHOLD, ls="--", lw=1.2)
        axis.set_xticks(range(len(labels)))
        axis.set_xticklabels(labels, fontsize=6.5)
        axis.set_ylim(0, 105)
        axis.set_ylabel("share of measurements inside the 95 % interval", fontsize=9)
    axis.set_title("(b) is it an interval at all", fontsize=10, fontweight="bold", loc="left")

    fig.suptitle("Bootstrap: the coverage before the width", fontsize=13, fontweight="bold")
    _figures.caption(fig, "Dashed grey: the nominal 95 %. A bar far below it says the interval does not "
                          "hold what it claims, and no width in panel (a) repairs that.")
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
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
    figure(parameters, coverage(predictions))


if __name__ == "__main__":
    main()
