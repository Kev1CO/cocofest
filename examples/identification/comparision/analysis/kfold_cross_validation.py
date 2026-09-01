"""
Whether the identification survives having part of its training set taken away.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

import _campaign
import _figures
import _shared

import matplotlib.pyplot as plt

import protocol

ANALYSIS = "kfold_cross_validation"


def held_out_scores(outcome: dict, fold) -> dict:
    """
    The prediction scores on the fold's own trains, told apart from the study's own testing trains.
    """
    table = outcome.get("trains")
    if table is None or len(table) == 0:
        return {}
    wanted = {protocol.condition_name(frequency, pulse_width * 1e-6) for frequency, pulse_width in fold}
    picked = table[table["condition"].isin(wanted)]
    if picked.empty:
        return {}
    return {
        "heldout_n": int(len(picked)),
        "heldout_peak_error_pct": float(picked["peak_error"].mean()),
        "heldout_fti_error_pct": float(picked["fti_error"].mean()),
        "heldout_r2": float(picked["r2"].mean()),
        "heldout_rmse": float(picked["rmse"].mean()),
    }


def run_subject(route, subject: str, scheme: str, k: int, seed: int) -> pd.DataFrame:
    """Every fold of one participant, as one row each."""
    rows = []
    groups = _campaign.folds(scheme, k=k, seed=seed)
    for index, fold in enumerate(groups):
        design = _campaign.design_with(to_testing=fold)
        names = ", ".join(protocol.condition_name(frequency, pulse_width * 1e-6)
                          for frequency, pulse_width in fold)
        print(f"  P{subject} fold {index + 1}/{len(groups)} holds out {names} "
              f"[{_campaign.describe(design)}]", flush=True)

        outcome = _campaign.identify(route, subject, design=design)
        rows.append({**_campaign.flat(outcome, route=route.METHOD, scheme=scheme, fold=index,
                                      held_out=names, n_held_out=len(fold)),
                     **held_out_scores(outcome, fold)})
    return pd.DataFrame(rows)


def summarise(detail: pd.DataFrame) -> pd.DataFrame:
    """One row per participant: the cross validated error, and how much the parameters moved between folds."""
    rows = []
    for (route, scheme, subject), group in detail.groupby(["route", "scheme", "subject"], sort=False):
        usable = group[group["failed"] == ""]
        row = {"route": route, "scheme": scheme, "subject": subject,
               "n_folds": int(len(group)), "n_usable": int(len(usable)),
               "converged": int(usable["converged"].sum())}
        for column in ("heldout_peak_error_pct", "heldout_fti_error_pct", "heldout_r2", "heldout_rmse"):
            if column in usable:
                row[f"cv_{column}"] = float(usable[column].mean())
                row[f"cv_{column}_sd"] = float(usable[column].std())
        for name in _shared.PARAMETERS:
            if name in usable and len(usable) > 1:
                values = usable[name].astype(float)
                row[f"{name}_median"] = float(values.median())
                row[f"{name}_cv_pct"] = float(100 * values.std() / values.mean()) if values.mean() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def against_study(summary: pd.DataFrame) -> pd.DataFrame:
    """The cross validated error next to the error the study's own single split reported."""
    rows = []
    for (route, scheme), group in summary.groupby(["route", "scheme"], sort=False):
        held_out = _shared.testing(route)
        rows.append({
            "route": route, "scheme": scheme, "n_subjects": int(len(group)),
            "study_peak_error_pct": float(held_out["peak_error"].mean()) if not held_out.empty else np.nan,
            "crossval_peak_error_pct": float(group["cv_heldout_peak_error_pct"].mean()),
            "study_r2": float(held_out["r2"].mean()) if not held_out.empty else np.nan,
            "crossval_r2": float(group["cv_heldout_r2"].mean()),
        })
    table = pd.DataFrame(rows)
    if not table.empty:
        table["optimism_pct"] = table["crossval_peak_error_pct"] - table["study_peak_error_pct"]
    return table


SCHEMES = ("train", "intensity", "frequency")


def held_out_label(scheme: str, held_out) -> str:
    """What a fold took away, named by the condition rather than by its index in the loop."""
    pairs = re.findall(r"(\d+)Hz (\d+)us", str(held_out))
    if not pairs:
        return str(held_out)
    if scheme == "intensity":
        return f"{pairs[0][1]} us"
    if scheme == "frequency":
        return f"{pairs[0][0]} Hz"
    return str(held_out)


def fold_labels(scheme: str, block: pd.DataFrame) -> dict:
    """What each fold is called on the axis: the condition it withdrew, or its rank in the train scheme."""
    if scheme == "train":
        return {value: f"fold {int(value) + 1}" for value in sorted(block["fold"].unique())}
    return {value: held_out_label(scheme, value) for value in block["held_out"]}


def folds_on_disk(scheme: str, detail=None) -> pd.DataFrame:
    """The folds of one scheme: the ones this run produced, or the ones an earlier one left."""
    if detail is not None and not detail.empty and (detail["scheme"] == scheme).any():
        return detail[detail["scheme"] == scheme]
    stored = _shared.stored(ANALYSIS, f"all_folds_{scheme}")
    return stored if stored is not None else pd.DataFrame()


def figure(detail=None, name="cross_validation") -> None:
    """
    The held-out error of every scheme, the three of them side by side.

    The abscissa is the condition that was taken away, not the index of the fold: dropping 250 us and
    dropping 400 us are not the same experiment, and a fold number hides which one was run.
    """
    blocks = {scheme: folds_on_disk(scheme, detail) for scheme in SCHEMES}
    blocks = {scheme: block[block["failed"] == ""] for scheme, block in blocks.items()
              if not block.empty and "heldout_peak_error_pct" in block}
    if not blocks:
        return

    fig, axes = plt.subplots(1, len(blocks), figsize=(5.6 * len(blocks), 4.8), squeeze=False)
    for axis, (scheme, block) in zip(axes[0], blocks.items()):
        routes = _figures.ordered(list(dict.fromkeys(block["route"])))
        named = fold_labels(scheme, block)
        told = block["fold" if scheme == "train" else "held_out"].map(named)
        labels = list(dict.fromkeys(told))
        for index, route in enumerate(routes):
            offset = (index - (len(routes) - 1) / 2) * 0.22
            for position, label in enumerate(labels):
                values = block.loc[(block["route"] == route) & (told == label),
                                   "heldout_peak_error_pct"].to_numpy(float)
                if not values.size:
                    continue
                axis.scatter(position + offset + _figures.jitter(len(values), 0.05), values, s=11,
                             alpha=0.55, edgecolor="none", color=_figures.colour(route))
                _figures.median_bar(axis, values, position + offset, width=0.09,
                                    colour_of=_figures.colour(route))
        axis.set_xticks(range(len(labels)))
        axis.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
        axis.set_xlabel("condition held out" if scheme != "train" else "fold", fontsize=8)
        axis.set_ylabel("held-out peak %model error", fontsize=8)
        axis.set_title(f"{scheme} scheme", fontsize=10, fontweight="bold")
        axis.tick_params(axis="y", labelsize=7)
        axis.grid(alpha=0.3, axis="y")
        _figures.count(axis, f"{block['subject'].nunique()} participants, {len(block)} folds")
        handles = [plt.Line2D([], [], ls="none", marker="o", color=_figures.colour(route),
                              label=_figures.label(route)) for route in routes]
        axis.legend(handles=handles, fontsize=6)

    fig.suptitle("Does the identification survive a different split", fontsize=13, fontweight="bold")
    _figures.caption(fig, "heldout_* is the fold that was taken away; testing_* is the study's own fixed "
                          "testing set. Both sit on the same row of the csv and are not the same number.\n"
                          "The reference column, 400 us, is never held out: a fold without it would not be "
                          "the same procedure.")
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    _shared.save_figure(fig, ANALYSIS, name)


def figure_optimism(name="optimism") -> None:
    """How much the study's own single split flatters the result, scheme by scheme."""
    frames = []
    for scheme in SCHEMES:
        stored = _shared.stored(ANALYSIS, f"against_study_{scheme}")
        if stored is not None:
            frames.append(stored)
    if not frames:
        return
    table = pd.concat(frames, ignore_index=True)

    fig, axis = plt.subplots(figsize=(4.0 + 1.4 * len(table), 4.6))
    for position, row in enumerate(table.itertuples()):
        tint = _figures.colour(row.route)
        axis.plot([position - 0.16, position + 0.16],
                  [row.study_peak_error_pct, row.crossval_peak_error_pct], color=tint, lw=3.0)
        axis.scatter([position - 0.16, position + 0.16],
                     [row.study_peak_error_pct, row.crossval_peak_error_pct], s=30, color=tint, zorder=3)
        axis.text(position - 0.2, row.study_peak_error_pct, f"{row.study_peak_error_pct:.1f}", ha="right",
                  va="center", fontsize=7)
        axis.text(position + 0.2, row.crossval_peak_error_pct, f"{row.crossval_peak_error_pct:.1f}",
                  ha="left", va="center", fontsize=7)
        axis.text(position, 0.5 * (row.study_peak_error_pct + row.crossval_peak_error_pct),
                  f"{row.optimism_pct:+.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    axis.set_xticks(range(len(table)))
    axis.set_xticklabels([f"{_figures.label(row.route)}\n{row.scheme}" for row in table.itertuples()],
                         fontsize=7)
    axis.set_ylabel("peak %model error", fontsize=9)
    axis.grid(alpha=0.3, axis="y")
    _figures.note(axis, "left point: the study's own split.  right point: cross validated.\n"
                        "the number between them is optimism_pct", size=6)

    fig.suptitle("What the single split of the study is worth", fontsize=12, fontweight="bold")
    _figures.caption(fig, "A positive optimism says the headline result is better than what a new "
                          "participant would get.\n"
                          "If it differs between routes, the paired comparison of group D is biased "
                          "towards the more optimistic route, which is why this figure exists.")
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    _shared.save_figure(fig, ANALYSIS, name)


def main():
    _shared.headless()
    parser = _shared.arguments(__doc__.splitlines()[1], routes=("movement_id",))
    parser.add_argument("--scheme", choices=("train", "intensity", "frequency"), default="intensity")
    parser.add_argument("--folds", type=int, default=5, help="folds of the train scheme, ignored otherwise")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--refresh", action="store_true", help="re-run participants already on disk")
    arguments = parser.parse_args()

    print(f"{arguments.scheme} cross validation")
    detail = _campaign.per_subject(
        ANALYSIS, arguments,
        lambda route, subject: run_subject(route, subject, arguments.scheme, arguments.folds,
                                           arguments.seed),
        tag_of=lambda route, subject: f"{route}_{arguments.scheme}_p{subject}",
    )
    if detail.empty:
        return
    summary = summarise(detail)
    _shared.banner("cross validation summary")
    print(summary.round(3).to_string(index=False))
    _shared.banner("cross validated against the study's own split")
    print(against_study(summary).round(2).to_string(index=False))

    tag = f"{arguments.scheme}"
    _shared.write(detail, ANALYSIS, f"all_folds_{tag}")
    _shared.write(summary, ANALYSIS, f"summary_{tag}")
    _shared.write(against_study(summary), ANALYSIS, f"against_study_{tag}")
    figure(detail)
    figure_optimism()


if __name__ == "__main__":
    main()
