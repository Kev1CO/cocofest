"""
Global sensitivity by the Fourier Amplitude Sensitivity Test, Cukier et al. (1973) and Saltelli (1999).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from SALib.analyze import fast as salib_fast
from SALib.sample import fast_sampler

import _campaign
import _shared
import _simulator

import matplotlib.pyplot as plt

ANALYSIS = "fast_global_sensitivity"

INTERFERENCE = 4  # M, the highest harmonic the spectrum is read at
DEFAULT_SAMPLES = 65  # N, has to be at least 4 M^2 + 1
DEFAULT_SPAN = 2.0  # the band around each identified value, x2 and /2
DEFAULT_CONDITIONS = ((20, 400), (33, 400), (50, 400))


def problem_of(box: dict) -> tuple:
    """The SALib problem for the parameters of `box`, and their names in the order of the design columns."""
    names = [name for name in _simulator.FREE if name in box]
    return {"num_vars": len(names), "names": names,
            "bounds": [[float(box[name][0]), float(box[name][1])] for name in names]}, names


def filled(responses: np.ndarray, samples: int) -> np.ndarray:
    """
    A model run that failed leaves a NaN, which the spectrum cannot be read through.
    """
    whole = np.array(responses, dtype=float)
    for start in range(0, len(whole), samples):
        block = whole[start:start + samples]
        if not np.all(np.isfinite(block)):
            centre = float(np.nanmean(block)) if np.any(np.isfinite(block)) else 0.0
            whole[start:start + samples] = np.nan_to_num(block, nan=centre)
    return whole


def run_subject(route, subject: str, samples: int, span: float, box_kind: str,
                conditions, seed: int = 0) -> pd.DataFrame:
    """One participant: the design run through the model, one condition at a time."""
    from helper import results

    stored = results.load(subject, route.METHOD)
    limits = _simulator.model_limits()
    box = (limits if box_kind == "bounds"
           else _simulator.box_around(stored["parameters"], span, limits))
    box = {name: box[name] for name in _simulator.FREE if name in box}

    problem, names = problem_of(box)
    design = fast_sampler.sample(problem, samples, M=INTERFERENCE, seed=seed)
    rows = []
    for frequency, pulse_width in conditions:
        produced = {"peak": np.empty(len(design)), "fti": np.empty(len(design))}
        for i, point in enumerate(design):
            block = _simulator.outcome_of(route, subject, dict(zip(names, point)), frequency, pulse_width)
            produced["peak"][i], produced["fti"][i] = block["peak"], block["fti"]
        print(f"  P{subject} {frequency}Hz {pulse_width}us: {len(design)} runs, "
              f"{np.isnan(produced['peak']).sum()} failed", flush=True)

        for scalar in ("peak", "fti"):
            # A parameter the output cannot see at all leaves a flat block, and a flat spectrum to divide by
            with np.errstate(divide="ignore", invalid="ignore"):
                read = salib_fast.analyze(problem, filled(produced[scalar], samples), M=INTERFERENCE,
                                          print_to_console=False, seed=seed)
            for index, name in enumerate(names):
                first, total = float(read["S1"][index]), float(read["ST"][index])
                rows.append({"route": route.METHOD, "subject": subject, "scalar": scalar,
                             "frequency": frequency, "pulse_width_us": pulse_width,
                             "parameter": name, "S1": first, "ST": total,
                             "S1_conf": float(read["S1_conf"][index]),
                             "ST_conf": float(read["ST_conf"][index]),
                             "interaction": total - first if np.isfinite(total) else np.nan,
                             "range_low": box[name][0], "range_high": box[name][1]})
    return pd.DataFrame(rows)


def summarise(detail: pd.DataFrame) -> pd.DataFrame:
    """One row per route, scalar and parameter, pooled over participants and conditions."""
    if detail.empty:
        return detail
    table = (detail.groupby(["route", "scalar", "parameter"], sort=False)
             .agg(n=("subject", "count"), S1=("S1", "mean"), S1_sd=("S1", "std"),
                  ST=("ST", "mean"), ST_sd=("ST", "std"), interaction=("interaction", "mean"),
                  S1_conf=("S1_conf", "mean"), ST_conf=("ST_conf", "mean"))
             .reset_index())
    table["reading"] = np.where(table["ST"] < 0.03, "invisible to the data",
                                np.where(table["S1"] < 0.5 * table["ST"], "only through interactions",
                                         "identifiable on its own"))
    return table.sort_values(["route", "scalar", "ST"], ascending=[True, True, False])


def figure(summary: pd.DataFrame, name="fast_global_sensitivity") -> None:
    """S1 and ST side by side, on the peak, so the interaction share reads as the gap between them."""
    peak = summary[summary["scalar"] == "peak"] if "scalar" in summary else summary
    if peak.empty:
        return
    routes = list(dict.fromkeys(peak["route"]))
    fig, axes = plt.subplots(1, len(routes), figsize=(4.4 * len(routes), 4.2), squeeze=False)

    positions = np.arange(len(_simulator.FREE))
    for axis, route in zip(axes[0], routes):
        group = peak[peak["route"] == route].set_index("parameter")
        first = [group["S1"].get(name, np.nan) for name in _simulator.FREE]
        total = [group["ST"].get(name, np.nan) for name in _simulator.FREE]
        axis.bar(positions - 0.2, first, 0.4, label="S1 (own effect)")
        axis.bar(positions + 0.2, total, 0.4, label="ST (with interactions)")
        axis.set_xticks(positions)
        axis.set_xticklabels(_simulator.FREE, rotation=45, ha="right", fontsize=7)
        axis.set_ylabel("share of the variance of the peak force", fontsize=8)
        axis.set_title(route, fontsize=8)
        axis.legend(fontsize=6)
        axis.tick_params(axis="y", labelsize=7)

    fig.tight_layout()
    _shared.save_figure(fig, ANALYSIS, name)


def main():
    _shared.headless()
    parser = _shared.arguments(__doc__.splitlines()[1], routes=("sensor_id",))
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES,
                        help=f"N, at least {4 * INTERFERENCE ** 2 + 1}")
    parser.add_argument("--span", type=float, default=DEFAULT_SPAN,
                        help="the multiplicative band around each identified value")
    parser.add_argument("--box", choices=("identified", "bounds"), default="identified",
                        help="draw around each participant's values, or across the whole admissible box")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--refresh", action="store_true", help="re-run participants already on disk")
    arguments = parser.parse_args()

    minimum = 4 * INTERFERENCE ** 2 + 1
    if arguments.samples < minimum:
        raise SystemExit(f"--samples must be at least {minimum} for M={INTERFERENCE}")

    print(f"FAST, N={arguments.samples}, box {arguments.box}")
    detail = _campaign.per_subject(
        ANALYSIS, arguments,
        lambda route, subject: run_subject(route, subject, arguments.samples, arguments.span,
                                           arguments.box, DEFAULT_CONDITIONS, arguments.seed),
        tag_of=lambda route, subject: f"{route}_{arguments.box}_p{subject}",
    )
    if detail.empty:
        return
    summary = summarise(detail)
    _shared.banner("FAST sensitivity indices")
    print(summary.round(3).to_string(index=False))
    _shared.write(detail, ANALYSIS, "all_indices")
    _shared.write(summary, ANALYSIS, "summary")
    figure(summary)


if __name__ == "__main__":
    main()
