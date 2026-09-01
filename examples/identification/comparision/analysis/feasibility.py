"""
What each route resolves, read off the five campaigns at once.

One cell per route, parameter and source of evidence. The table concludes nothing on its own: it says
where the five sources agree, which are the only conclusions worth defending, and where a candidate route
parts from the reference, which is what the comparison is about.

    profile        profile_likelihood/summary.csv            verdict, the cohort's mode
    global         fast_global_sensitivity/summary.csv       reading
    bound          parameter_bounds/bounds.csv               share of the cohort on a bound
    multistart     multistart_robustness/summary.csv         between-start CV
    bootstrap      bootstrap_confidence/parameter_intervals  CV of the interval

Every threshold below is a choice, and the figure prints them rather than hiding them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import _figures
import _shared

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ANALYSIS = "feasibility"

GOOD, DOUBTFUL, BAD = "resolved", "doubtful", "not resolved"
LEVELS = (GOOD, DOUBTFUL, BAD)
TINTS = {GOOD: "tab:green", DOUBTFUL: "0.78", BAD: "tab:purple", "": "white"}

PINNED_SHARE = 0.7  # the same threshold parameter_bounds' shortlist uses
FLAT_CV_PCT = 5.0
LOOSE_CV_PCT = 25.0

SOURCES = ("profile", "global", "bound", "multistart", "bootstrap")
THRESHOLDS = {
    "profile": "identifiable / one-sided / not identifiable, as the profile wrote it",
    "global": "the reading column, already written (ST < 0.03 invisible, S1 < 0.5 ST interactions)",
    "bound": f"share on a bound below 0.1, then below {PINNED_SHARE}, then above",
    "multistart": f"between-start CV below {FLAT_CV_PCT} %, then below {LOOSE_CV_PCT} %, then above",
    "bootstrap": f"bootstrap CV below {FLAT_CV_PCT} %, then below {LOOSE_CV_PCT} %, then above",
}


def _read(program: str, name: str) -> pd.DataFrame:
    """One campaign's summary, or an empty frame when that campaign has not run."""
    stored = _shared.stored(program, name)
    return stored if stored is not None else pd.DataFrame()


def _from_cv(value) -> str:
    """A coefficient of variation read as one of the three levels."""
    if not np.isfinite(value):
        return ""
    return GOOD if value <= FLAT_CV_PCT else DOUBTFUL if value <= LOOSE_CV_PCT else BAD


def levels(routes) -> pd.DataFrame:
    """One row per route, parameter and source: the level that source puts it at."""
    profiles = _read("profile_likelihood", "summary")
    globals_ = _read("fast_global_sensitivity", "summary")
    bounds = _read("parameter_bounds", "bounds")
    starts = _read("multistart_robustness", "summary")
    draws = _read("bootstrap_confidence", "parameter_intervals")

    verdicts = {"identifiable": GOOD, "one-sided": DOUBTFUL, "not identifiable": BAD}
    readings = {"identifiable on its own": GOOD, "only through interactions": DOUBTFUL,
                "invisible to the data": BAD}

    rows = []
    for route in _figures.ordered(routes):
        for parameter in _shared.PARAMETERS:
            told = {source: "" for source in SOURCES}

            block = profiles[(profiles["route"] == route) & (profiles["parameter"] == parameter)] \
                if not profiles.empty else pd.DataFrame()
            if not block.empty:
                told["profile"] = verdicts.get(block["verdict"].mode().iloc[0], "")

            block = globals_[(globals_["route"] == route) & (globals_["parameter"] == parameter)
                             & (globals_["scalar"] == "peak")] if not globals_.empty else pd.DataFrame()
            if not block.empty:
                told["global"] = readings.get(block["reading"].mode().iloc[0], "")

            block = bounds[(bounds["route"] == route) & (bounds["parameter"] == parameter)] \
                if not bounds.empty else pd.DataFrame()
            if not block.empty:
                share = float(block["at_bound"].iloc[0]) / max(float(block["n"].iloc[0]), 1)
                told["bound"] = GOOD if share <= 0.1 else DOUBTFUL if share < PINNED_SHARE else BAD

            block = starts[starts["route"] == route] if not starts.empty else pd.DataFrame()
            if not block.empty and f"{parameter}_cv_pct" in block:
                told["multistart"] = _from_cv(float(block[f"{parameter}_cv_pct"].median()))

            block = draws[(draws["route"] == route) & (draws["parameter"] == parameter)] \
                if not draws.empty else pd.DataFrame()
            if not block.empty:
                told["bootstrap"] = _from_cv(float(block["cv_pct"].median()))

            rows.append({"route": route, "parameter": parameter, **told})
    table = pd.DataFrame(rows)
    if not table.empty:
        drawn = table[list(SOURCES)]
        table["sources"] = (drawn != "").sum(axis=1)
        table["agree"] = [row.nunique() == 1 and row.iloc[0] != "" for _, row in drawn.iterrows()]
    return table


def figure(table: pd.DataFrame, name="feasibility") -> None:
    """The five sources of evidence in one grid, route by route."""
    if table.empty:
        return
    routes = _figures.ordered(list(dict.fromkeys(table["route"])))
    reference = table[table["route"] == _figures.REFERENCE].set_index("parameter")

    fig, axes = plt.subplots(1, len(routes), figsize=(4.6 * len(routes), 5.4), squeeze=False)
    for axis, route in zip(axes[0], routes):
        block = table[table["route"] == route].set_index("parameter")
        for row, parameter in enumerate(_shared.PARAMETERS):
            y = len(_shared.PARAMETERS) - row - 1
            if parameter not in block.index:
                continue
            entry = block.loc[parameter]
            for column, source in enumerate(SOURCES):
                axis.add_patch(Rectangle((column, y), 1, 1, facecolor=TINTS.get(entry[source], "white"),
                                         alpha=0.75, edgecolor="0.8", linewidth=0.6))
                if not entry[source]:
                    axis.text(column + 0.5, y + 0.5, "-", ha="center", va="center", fontsize=8,
                              color="0.6")
                elif (route != _figures.REFERENCE and parameter in reference.index
                      and reference.loc[parameter, source] and
                      reference.loc[parameter, source] != entry[source]):
                    axis.text(column + 0.92, y + 0.08, "!=", ha="right", va="bottom", fontsize=7,
                              fontweight="bold")
            if bool(entry.get("agree", False)):
                axis.add_patch(Rectangle((0, y), len(SOURCES), 1, facecolor="none", edgecolor="black",
                                         linewidth=2.2))

        axis.set_xlim(0, len(SOURCES))
        axis.set_ylim(0, len(_shared.PARAMETERS))
        axis.set_xticks(np.arange(len(SOURCES)) + 0.5)
        axis.set_xticklabels(SOURCES, fontsize=7.5, rotation=30, ha="right")
        axis.set_yticks(np.arange(len(_shared.PARAMETERS)) + 0.5)
        axis.set_yticklabels(list(_shared.PARAMETERS)[::-1], fontsize=8)
        axis.set_title(_figures.label(route), fontsize=10, fontweight="bold")
        axis.tick_params(length=0)

    handles = [Rectangle((0, 0), 1, 1, facecolor=TINTS[level], alpha=0.75, label=level)
               for level in LEVELS]
    handles += [Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="black", linewidth=2.2,
                          label="the five sources agree"),
                Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="0.8",
                          label="!= differs from the reference route")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.93), fontsize=7, ncol=5)
    fig.suptitle("What each route resolves, and where the evidence agrees", fontsize=13,
                 fontweight="bold", y=0.995)
    _figures.caption(fig, "A dash is a campaign that has not run yet, not a negative result.\n"
                          + "\n".join(f"{source}: {told}" for source, told in THRESHOLDS.items())
                          + f"\nThese thresholds are choices. Two of them are already written in "
                            f"shortlist.csv (PINNED_SHARE {PINNED_SHARE}, FLAT_CV_PCT {FLAT_CV_PCT}); if "
                            f"this table becomes one of the manuscript, they need a justification rather "
                            f"than a mention.")
    fig.tight_layout(rect=(0, 0.22, 1, 0.93))
    _shared.save_figure(fig, ANALYSIS, name)


def main():
    _shared.headless()
    parser = _shared.arguments(__doc__.splitlines()[1], subjects=False)
    routes = _shared.available(parser.parse_args().routes)

    _shared.banner("what each route resolves, from the five campaigns")
    table = levels(routes)
    if table.empty:
        print("  nothing on disk yet")
        return
    print(table.to_string(index=False))
    if int(table["sources"].max()) == 0:
        print("\n  no campaign has run: every cell is a dash")
    _shared.write(table, ANALYSIS, "feasibility")
    figure(table)


if __name__ == "__main__":
    main()
