"""
Whether the identification survives having part of its training set taken away.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import _campaign
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


def figure(detail: pd.DataFrame, name=None) -> None:
    """The held-out error fold by fold, and the parameter spread the folds produced."""
    usable = detail[detail["failed"] == ""]
    if usable.empty or "heldout_peak_error_pct" not in usable:
        return
    scheme = usable["scheme"].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    axis = axes[0]
    for fold, group in usable.groupby("fold"):
        label = group["held_out"].iloc[0]
        axis.scatter([fold] * len(group), group["heldout_peak_error_pct"], s=12, alpha=0.6,
                     edgecolor="none", label=label if fold < 8 else None)
    axis.set_xlabel("fold", fontsize=8)
    axis.set_ylabel("held-out peak %model error", fontsize=8)
    axis.legend(fontsize=5.5, ncol=2)
    axis.tick_params(labelsize=7)

    axis = axes[1]
    present = [name_of for name_of in _shared.PARAMETERS if name_of in usable]
    _shared.boxplot(axis, [usable.groupby("subject")[name_of].apply(
        lambda values: 100 * values.std() / values.mean() if values.mean() else np.nan).dropna()
        for name_of in present], present, showfliers=False)
    axis.set_ylabel("between-fold CV of the identified value (%)", fontsize=8)
    axis.tick_params(axis="x", labelrotation=90, labelsize=7)
    axis.tick_params(axis="y", labelsize=7)

    fig.suptitle(f"cross validation, {scheme} scheme", fontsize=9)
    fig.tight_layout()
    _shared.save_figure(fig, ANALYSIS, name or f"kfold_{scheme}")


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


if __name__ == "__main__":
    main()
