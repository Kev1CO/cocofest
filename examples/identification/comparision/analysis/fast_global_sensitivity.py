"""
Global sensitivity by the Fourier Amplitude Sensitivity Test, Cukier et al. (1973) and Saltelli (1999).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from SALib.analyze import fast as salib_fast
from SALib.sample import fast_sampler

import _campaign
import _figures
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


READING_COLOURS = {"invisible to the data": "0.75", "only through interactions": "tab:orange",
                   "identifiable on its own": "tab:green"}


def figure(summary: pd.DataFrame, box="identified", conditions=DEFAULT_CONDITIONS,
           name="fast_global_sensitivity") -> None:
    """
    What the measurement sees: the local elasticity beside the global indices.

    Panel (a) is read off local_sensitivity's own summary when that campaign has run; the two answer
    different questions and the figure carries both so neither is quoted alone.
    """
    peak = summary[summary["scalar"] == "peak"] if "scalar" in summary else summary
    if peak.empty:
        return
    routes = _figures.ordered(list(dict.fromkeys(peak["route"])))

    fig, axes = plt.subplots(1, 2, figsize=(9.0 + 1.6 * len(routes), 5.2))
    axis = axes[0]
    local = _shared.read("local_sensitivity", "summary") \
        if _shared.path_of("local_sensitivity", "summary").exists() else pd.DataFrame()
    if local.empty:
        axis.axis("off")
        axis.text(0.5, 0.5, "local_sensitivity has not run yet:\nits summary is what this panel draws",
                  ha="center", va="center", fontsize=9, color="0.4")
    else:
        import local_sensitivity

        local_sensitivity.draw_local(axis, local)
    axis.set_title("(a) local, around each participant's own values", fontsize=10, fontweight="bold",
                   loc="left")

    axis = axes[1]
    width = 0.8 / max(len(routes), 1)
    positions = np.arange(len(_simulator.FREE))
    for index, route in enumerate(routes):
        group = peak[peak["route"] == route].set_index("parameter")
        for offset, name_of in zip(positions, _simulator.FREE):
            first = float(group["S1"].get(name_of, np.nan))
            total = float(group["ST"].get(name_of, np.nan))
            told = group.get("reading", pd.Series(dtype=object)).get(name_of, "")
            tint = READING_COLOURS.get(told, "0.5")
            x = offset + index * width
            # S1 solid, carried on to ST in the same colour: the gap is the interaction share
            axis.bar(x, max(total - first, 0.0), width, bottom=max(first, 0.0), color=tint, alpha=0.30,
                     edgecolor=_figures.colour(route), linewidth=1.6)
            axis.bar(x, first, width, color=tint, edgecolor=_figures.colour(route), linewidth=1.6)
    axis.set_xticks(positions + 0.4 - width / 2)
    axis.set_xticklabels(_simulator.FREE, rotation=45, ha="right", fontsize=7)
    axis.set_ylabel("share of the variance of the peak force", fontsize=8)
    axis.set_title("(b) global, FAST indices", fontsize=10, fontweight="bold", loc="left")
    axis.tick_params(axis="y", labelsize=7)
    handles = [plt.Rectangle((0, 0), 1, 1, color=tint, label=told)
               for told, tint in READING_COLOURS.items()]
    handles += [plt.Rectangle((0, 0), 1, 1, facecolor="0.5", edgecolor=_figures.colour(route),
                              label=_figures.label(route)) for route in routes]
    axis.legend(handles=handles, fontsize=6, loc="upper right")
    _figures.note(axis, "solid: S1, the parameter's own effect. Faded on top: up to ST, so the faded "
                        "part\nis the share that only exists through interactions.\n"
                        "The colour is the reading column, already written: it is not re-thresholded "
                        "here.", size=5.5)

    fig.suptitle("What the measurement sees of each parameter", fontsize=13, fontweight="bold")
    _figures.caption(fig, f"Box: {box}. Conditions swept: "
                          + ", ".join(f"({frequency} Hz, {pulse_width} us)"
                                      for frequency, pulse_width in conditions)
                          + " - not a full sweep of the design.\n"
                          "For the isometric routes the simulated model is the muscle model alone; for a "
                          "movement route it goes through that route's own program. The indices of a "
                          "movement route are indicative, not equivalent,\nwhich is exactly the "
                          "comparison these grouped bars invite: do not make it.")
    fig.tight_layout(rect=(0, 0.09, 1, 0.94))
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
    figure(summary, box=arguments.box)


if __name__ == "__main__":
    main()
