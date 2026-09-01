"""
Whether a reconstructed observable is as good as the reference one, participant by participant.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import _figures
import _shared

import matplotlib.pyplot as plt

import agreement_metrology
import prediction_accuracy

ANALYSIS = "route_equivalence"

CONFIDENCE = 0.90
SCALARS = (("peak_error_pct", "peak %model error", True),
           ("fti_error_pct", "integral %model error", True),
           ("r2", r"$R^2$", False),
           ("bfr_pct", "Best Fit Rate (%)", False),
           ("nrmse_pct", "NRMSE (%)", True))
SLOPE_PANELS = ("peak_error_pct", "r2", "nrmse_pct")

MARGIN_FROM = {"peak_error_pct": ("peak",), "fti_error_pct": ("fti",),
               "bfr_pct": ("peak", "fti"), "nrmse_pct": ("peak", "fti"), "r2": ()}


def scores(routes) -> pd.DataFrame:
    """The per participant scores of every route, the table every comparison here is paired on."""
    stored = _shared.path_of("prediction_accuracy", "per_subject")
    if stored.exists():
        table = _shared.read("prediction_accuracy", "per_subject")
        table["subject"] = table["subject"].map(lambda value: f"{int(value):02d}")
        return table[table["route"].isin(routes)]
    return prediction_accuracy.per_subject(routes)


def paired(table: pd.DataFrame, candidate: str, column: str) -> dict:
    """One paired difference: the candidate route minus the reference, over the shared participants."""
    wide = table[table["route"].isin([_figures.REFERENCE, candidate])].pivot_table(
        index="subject", columns="route", values=column)
    if _figures.REFERENCE not in wide or candidate not in wide:
        return {}
    wide = wide.dropna()
    left, right = wide[_figures.REFERENCE].to_numpy(float), wide[candidate].to_numpy(float)
    if len(left) < 3:
        return {}

    difference = right - left
    half = stats.t.ppf(0.5 + CONFIDENCE / 2, len(difference) - 1) * difference.std(ddof=1) / np.sqrt(len(difference))
    try:
        p = float(stats.wilcoxon(left, right).pvalue)
    except ValueError:
        p = 1.0
    return {"reference": _figures.REFERENCE, "candidate": candidate, "scalar": column,
            "n": int(len(difference)), "reference_median": float(np.median(left)),
            "candidate_median": float(np.median(right)),
            "mean_difference": float(difference.mean()),
            "ci_low": float(difference.mean() - half), "ci_high": float(difference.mean() + half),
            "wilcoxon_p": p, "candidate_lower": int((right < left).sum()),
            "candidate_higher": int((right > left).sum()),
            "left": left, "right": right, "subjects": list(wide.index)}


def differences(table: pd.DataFrame, candidates) -> pd.DataFrame:
    """Every paired difference of the study, one row per (candidate, scalar)."""
    rows = [paired(table, candidate, column) for candidate in candidates for column, _, _ in SCALARS]
    rows = [{key: value for key, value in row.items() if key not in ("left", "right", "subjects")}
            for row in rows if row]
    return pd.DataFrame(rows)


def margins(routes, override=None) -> dict:
    """The equivalence margin of every scalar: the MDC95 of the reference measurement, in %."""
    if override is not None:
        return {column: float(override) for column, _, _ in SCALARS}
    metrology = agreement_metrology.table([_figures.REFERENCE])
    if metrology.empty:
        return {column: np.nan for column, _, _ in SCALARS}
    mdc = metrology.set_index("scalar")["mdc95_pct"].to_dict()
    return {column: float(np.mean([mdc[scalar] for scalar in sources])) if sources else np.nan
            for column, sources in MARGIN_FROM.items()}


def equivalence(table: pd.DataFrame, candidates, margin: dict) -> pd.DataFrame:
    """The equivalence verdict of every (candidate, scalar), read off the interval and the margin."""
    frame = differences(table, candidates)
    if frame.empty:
        return frame
    frame["margin"] = frame["scalar"].map(margin)
    inside = (frame["ci_low"] > -frame["margin"]) & (frame["ci_high"] < frame["margin"])
    frame["verdict"] = np.where(frame["margin"].isna(), "no margin declared",
                                np.where(inside, "equivalent", "not concluded"))
    return frame


def ranks(table: pd.DataFrame, candidate: str, column: str, smaller_is_better=True) -> pd.DataFrame:
    """The participants ranked by one score on both routes, best first."""
    wide = table[table["route"].isin([_figures.REFERENCE, candidate])].pivot_table(
        index="subject", columns="route", values=column).dropna()
    if _figures.REFERENCE not in wide or candidate not in wide:
        return pd.DataFrame()
    order = wide.rank(ascending=smaller_is_better)
    order.columns = ["reference_rank" if name == _figures.REFERENCE else "candidate_rank"
                     for name in order.columns]
    return order.reset_index()


def rank_agreement(table: pd.DataFrame, candidates) -> pd.DataFrame:
    """Spearman's rho between the two rankings, with the interval a Fisher transform gives it."""
    rows = []
    for candidate in candidates:
        for column, _, smaller in SCALARS:
            order = ranks(table, candidate, column, smaller)
            if len(order) < 4:
                continue
            rho, p = stats.spearmanr(order["reference_rank"], order["candidate_rank"])
            half = 1.96 / np.sqrt(len(order) - 3)
            rows.append({"reference": _figures.REFERENCE, "candidate": candidate, "scalar": column,
                         "n": int(len(order)), "spearman_rho": float(rho), "p": float(p),
                         "rho_low": float(np.tanh(np.arctanh(rho) - half)),
                         "rho_high": float(np.tanh(np.arctanh(rho) + half)),
                         "crossings": int((order["reference_rank"] != order["candidate_rank"]).sum())})
    return pd.DataFrame(rows)


def figure_slopes(table: pd.DataFrame, candidates, name="slopes") -> None:
    """One line per participant from the reference route to the candidate, on three scores."""
    labels = {column: label for column, label, _ in SCALARS}
    fig, axes = plt.subplots(len(candidates), len(SLOPE_PANELS),
                             figsize=(4.2 * len(SLOPE_PANELS), 4.2 * len(candidates)), squeeze=False)

    for row, candidate in enumerate(candidates):
        for column, scalar in enumerate(SLOPE_PANELS):
            axis = axes[row][column]
            block = paired(table, candidate, scalar)
            if not block:
                axis.axis("off")
                continue
            smaller = dict((name_of, better) for name_of, _, better in SCALARS)[scalar]
            counts = _figures.slope_lines(axis, block["left"], block["right"], smaller_is_better=smaller,
                                          labels=[_figures.label(_figures.REFERENCE),
                                                  _figures.label(candidate)])
            axis.set_ylabel(labels[scalar], fontsize=9)
            axis.tick_params(axis="y", labelsize=7)
            axis.grid(alpha=0.3, axis="y")
            better = "lower" if smaller else "higher"
            _figures.note(axis,
                          f"{counts['better']}/{counts['n']} participants go {better} on the candidate, "
                          f"{counts['worse']} the other way\n"
                          f"median {block['reference_median']:.2f} -> {block['candidate_median']:.2f}   "
                          f"paired Wilcoxon p {block['wilcoxon_p']:.3f}", size=6)
            if column == 0:
                axis.set_title(f"{_figures.label(_figures.REFERENCE)} -> {_figures.label(candidate)}",
                               fontsize=10, fontweight="bold", loc="left")
            _figures.count(axis, f"n {counts['n']} paired")

    fig.suptitle("What changes, participant by participant, when the observable is reconstructed",
                 fontsize=13, fontweight="bold")
    _figures.caption(fig, "Green: the candidate route is better on that participant. Red: worse. "
                          "Thick rule: the median.\n" + _figures.ASYMMETRY)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    _shared.save_figure(fig, ANALYSIS, name)


def figure_equivalence(frame: pd.DataFrame, name="equivalence") -> None:
    """The paired gap against a margin fixed beforehand, which is what the word correct rests on."""
    if frame.empty:
        return
    rows = list(frame.itertuples())
    fig, axis = plt.subplots(figsize=(11, 0.62 * len(rows) + 4.0))

    for position, row in enumerate(rows):
        y = len(rows) - position - 1
        if np.isfinite(row.margin):
            axis.fill_betweenx([y - 0.45, y + 0.45], -row.margin, row.margin, color="0.55", alpha=0.30,
                               zorder=0, linewidth=0)
            for edge in (-row.margin, row.margin):
                axis.plot([edge, edge], [y - 0.45, y + 0.45], color="0.35", lw=0.9, zorder=1)
        tint = _figures.colour(row.candidate)
        axis.plot([row.ci_low, row.ci_high], [y, y], color=tint, lw=2.4, solid_capstyle="butt", zorder=3)
        axis.scatter(row.mean_difference, y, s=34, color=tint, zorder=4)
        axis.text(-0.01, y, f"{row.scalar}   {_figures.label(row.candidate)}", ha="right", va="center",
                  fontsize=8, transform=axis.get_yaxis_transform())
        axis.text(1.005, y, f"{row.verdict}   n {row.n}", ha="left", va="center", fontsize=7,
                  color="0.3", transform=axis.get_yaxis_transform())

    axis.axvline(0.0, color="0.4", lw=0.8)
    axis.set_yticks([])
    axis.set_ylim(-0.7, len(rows) - 0.3)
    axis.set_xlabel(f"paired difference, candidate route minus {_figures.REFERENCE} "
                    f"(points of the score)", fontsize=9)
    axis.grid(alpha=0.3, axis="x")

    quoted = ", ".join(f"{row.scalar} +/- {row.margin:.0f}" for row in rows
                       if np.isfinite(row.margin) and row.candidate == rows[0].candidate)
    fig.suptitle("Is the gap smaller than a gap that would count", fontsize=13, fontweight="bold")
    _figures.caption(fig,
                     f"Bar: the {CONFIDENCE:.0%} confidence interval of the mean paired difference. Grey "
                     f"band: the equivalence margin. An interval wholly inside the band concludes to "
                     f"equivalence; an interval that leaves it concludes nothing.\n"
                     f"1. A non significant p is not an equivalence. This figure exists to avoid exactly "
                     f"that reading.\n"
                     f"2. The margin was fixed before the result was looked at: the MDC95 of the "
                     f"reference measurement, in % of the mean measured ({quoted}).\n"
                     f"   A gap smaller than the smallest change the measurement itself can detect is "
                     f"not detectable. Another margin is acceptable but has to be defended in the text, "
                     f"and --margin sets it.\n"
                     f"3. Only dimensionless scores are here: a difference of RMSE between a route in N "
                     f"and a route in deg means nothing.\n"
                     f"   R2 gets no margin, so its line reports the interval and concludes nothing.\n"
                     + _figures.ASYMMETRY)
    fig.tight_layout(rect=(0, 0.20, 1, 0.95))
    _shared.save_figure(fig, ANALYSIS, name)


def figure_ranks(table: pd.DataFrame, candidates, agreement: pd.DataFrame, name="ranks") -> None:
    """Whether a candidate route puts the participants in the same order as the reference."""
    panels = ("peak_error_pct", "r2")
    fig, axes = plt.subplots(len(panels), len(candidates),
                             figsize=(4.6 * len(candidates), 4.6 * len(panels)), squeeze=False)

    for row, scalar in enumerate(panels):
        smaller = dict((name_of, better) for name_of, _, better in SCALARS)[scalar]
        for column, candidate in enumerate(candidates):
            axis = axes[row][column]
            order = ranks(table, candidate, scalar, smaller)
            if order.empty:
                axis.axis("off")
                continue
            for _, entry in order.iterrows():
                crossed = entry["reference_rank"] != entry["candidate_rank"]
                axis.plot([0, 1], [entry["reference_rank"], entry["candidate_rank"]],
                          color=_figures.colour(candidate) if crossed else "0.6",
                          lw=1.1 if crossed else 0.7, alpha=0.85 if crossed else 0.5)
                axis.scatter([0, 1], [entry["reference_rank"], entry["candidate_rank"]], s=10,
                             color="0.3", zorder=3)
                axis.text(-0.03, entry["reference_rank"], f"P{entry['subject']}", ha="right", va="center",
                          fontsize=5.5, color="0.35")
            axis.invert_yaxis()
            axis.set_xticks([0, 1])
            axis.set_xticklabels([_figures.label(_figures.REFERENCE), _figures.label(candidate)],
                                 fontsize=7)
            axis.set_xlim(-0.25, 1.25)
            axis.set_ylabel(f"rank on {scalar} (1 = best)", fontsize=8)
            axis.tick_params(axis="y", labelsize=7)
            block = agreement[(agreement["candidate"] == candidate) & (agreement["scalar"] == scalar)]
            if not block.empty:
                entry = block.iloc[0]
                axis.set_title(f"Spearman rho {entry['spearman_rho']:.2f} "
                               f"[{entry['rho_low']:.2f}, {entry['rho_high']:.2f}], p {entry['p']:.3f}   -   "
                               f"{int(entry['crossings'])} of {int(entry['n'])} participants change rank\n"
                               f"{int(entry['n'])} ranks make rho unstable: the crossings are the result, "
                               f"the coefficient only a summary", fontsize=6.5, loc="left")

    fig.suptitle("Does a reconstructed observable rank the participants the same way",
                 fontsize=13, fontweight="bold")
    _figures.caption(fig, "Coloured lines: participants whose rank changes.\n" + _figures.ASYMMETRY)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    _shared.save_figure(fig, ANALYSIS, name)


def main():
    _shared.headless()
    parser = _shared.arguments(__doc__.splitlines()[1], subjects=False)
    parser.add_argument("--margin", type=float, default=None,
                        help="equivalence margin in points of the score, the MDC95 of the reference "
                             "route otherwise")
    arguments = parser.parse_args()
    routes = _shared.available(arguments.routes)
    candidates = [route for route in routes if route != _figures.REFERENCE]
    if _figures.REFERENCE not in routes or not candidates:
        print(f"  {_figures.REFERENCE} and at least one candidate route are needed")
        return

    table = scores(routes)
    _shared.banner(f"paired against {_figures.REFERENCE}, one participant at a time")
    frame = equivalence(table, candidates, margins(routes, arguments.margin))
    print(frame.drop(columns=["reference"]).round(3).to_string(index=False))
    _shared.write(frame, ANALYSIS, "equivalence")

    _shared.banner("rank agreement")
    agreement = rank_agreement(table, candidates)
    print(agreement.drop(columns=["reference"]).round(3).to_string(index=False))
    _shared.write(agreement, ANALYSIS, "rank_agreement")

    figure_slopes(table, candidates)
    figure_equivalence(frame)
    figure_ranks(table, candidates, agreement)


if __name__ == "__main__":
    main()
