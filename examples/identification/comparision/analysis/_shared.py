"""
Common ground for the analysis programs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True

COMPARISON_ROOT = Path(__file__).resolve().parent.parent
for _root in (COMPARISON_ROOT / "id_procedure", COMPARISON_ROOT, COMPARISON_ROOT / "optimization"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import routes as route_table
import store
from helper import results

ROUTES = route_table.NAMES
ISOMETRIC_ROUTES = tuple(name for name in ROUTES if route_table.of(name).family == "isometric")

# The order the tables print them
PARAMETERS = ("a_scale", "km_rest", "tau1_rest", "tau2", "pd0", "pdt")
SCALARS = ("peak", "fti")

ANALYSIS_ROOT = store.RESULTS_ROOT / "analysis"


# ---- Where a program writes ---- #
def folder_of(program: str) -> Path:
    """The folder one program owns, created on first use."""
    folder = ANALYSIS_ROOT / program
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def path_of(program: str, name: str) -> Path:
    """The csv one participant of one campaign is stored as, the unit a resumed run checks for."""
    return ANALYSIS_ROOT / program / f"{name}.csv"


def write(frame: pd.DataFrame, program: str, name: str) -> Path:
    """Write one table of a program, and echo where it went."""
    path = folder_of(program) / f"{name}.csv"
    frame.to_csv(path, index=False)
    print(f"  -> {path}")
    return path


def stored(program: str, name: str):
    """
    What a campaign already holds for one participant, or None when it still has to be run.
    """
    if not path_of(program, name).exists():
        return None
    try:
        frame = read(program, name)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        print(f"  {name}: stored csv is unreadable, running it again")
        return None
    return None if frame.empty else frame


def read(program: str, name: str):
    """
    Read back one table a program wrote.
    """
    frame = pd.read_csv(path_of(program, name))
    for column in ("failed", "at_bound"):
        if column in frame:
            frame[column] = frame[column].fillna("")
    return frame


def save_figure(figure, program: str, name: str) -> Path:
    """Write one figure of a program as svg, close it, and echo where it went."""
    path = folder_of(program) / f"{name}.svg"
    figure.savefig(path)
    plt.close(figure)
    print(f"  -> {path}")
    return path


def boxplot(axis, series, labels, **kwargs) -> None:
    """
    A boxplot with its tick labels, spelled the way any matplotlib accepts.
    """
    axis.boxplot(list(series), **kwargs)
    axis.set_xticks(range(1, len(labels) + 1))
    axis.set_xticklabels(list(labels))


# ---- Reading what the routes wrote ---- #
def cohort(route: str) -> pd.DataFrame:
    """One row per participant of one route: the scalars, the parameters and their bounds."""
    rows = []
    for subject, result in results.load_method(route).items():
        extra = result["extra"]
        row = {
            "subject": subject,
            "route": route,
            "rmse": result["rmse"],
            "converged": result["converged"],
            "unit": result["unit"],
            "at_bound": result["parameters_at_bound"],
            "n_phases": result["n_phases"],
            "seconds": float(sum(extra.get("step_seconds", {}).values())),
        }
        row.update(result["parameters"])
        for name, low, high in zip(result["parameter_names"], result["parameter_min"], result["parameter_max"]):
            row[f"{name}__min"], row[f"{name}__max"] = float(low), float(high)
        row["extra"] = extra
        rows.append(row)
    return pd.DataFrame(rows)


def trains(route: str) -> pd.DataFrame:
    """One row per train of one route, identification and testing, already scored by the routes."""
    rows = []
    for subject, result in results.load_method(route).items():
        table = result["extra"].get("trains")
        if not table:
            continue
        for i in range(len(table["condition"])):
            rows.append({"subject": subject, "route": route,
                         **{key: (value[i] if isinstance(value, list) else value)
                            for key, value in table.items()}})
    return pd.DataFrame(rows)


def testing(route: str) -> pd.DataFrame:
    """The trains of one route the identification never saw, the set every prediction score is read on."""
    table = trains(route)
    return table[table["set"] == "testing"] if not table.empty else table


def subjects_of(route: str) -> list:
    """The participants one route actually holds a result for."""
    return sorted(results.load_method(route))


def available(names=ROUTES) -> list:
    """The routes that have at least one result on disk, so a partial study still reports."""
    return [route for route in names if (store.RESULTS_ROOT / route).exists() and subjects_of(route)]


# ---- The measures every analysis shares ---- #
def grade_icc(value) -> str:
    """Koo & Li (2016): how an ICC is read out loud."""
    if not np.isfinite(value):
        return "-"
    return "poor" if value < 0.5 else "moderate" if value < 0.75 else "good" if value < 0.9 else "excellent"


def best_fit_rate(measured, predicted) -> float:
    """
    The system identification fit criterion, in %.
    """
    measured, predicted = np.asarray(measured, float), np.asarray(predicted, float)
    usable = np.isfinite(measured) & np.isfinite(predicted)
    measured, predicted = measured[usable], predicted[usable]
    spread = np.linalg.norm(measured - measured.mean())
    return np.nan if spread == 0 else float(100.0 * (1 - np.linalg.norm(measured - predicted) / spread))


def normalised_rmse(measured, predicted) -> float:
    """The rmse as a % of the measured range, so routes in N, Nm and degrees can sit in one column."""
    measured, predicted = np.asarray(measured, float), np.asarray(predicted, float)
    usable = np.isfinite(measured) & np.isfinite(predicted)
    measured, predicted = measured[usable], predicted[usable]
    extent = measured.max() - measured.min()
    return np.nan if extent == 0 else float(100.0 * np.sqrt(np.mean((measured - predicted) ** 2)) / extent)


def signal_of(route: str, subject: str) -> tuple:
    """The step 5 prediction of one participant, measured against predicted, on its own time base."""
    result = results.load(subject, route)
    return result["time"], result["tracked"], result["identified"]


# ---- The command line every analysis answers to ---- #
def arguments(description: str, routes=ROUTES, subjects=True) -> argparse.ArgumentParser:
    """The flags shared by the whole package, so no two analyses spell the same option differently."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--routes", nargs="+", default=list(routes), choices=list(ROUTES),
                        help="routes to analyse")
    if subjects:
        parser.add_argument("--subjects", type=int, nargs="+", default=None,
                            help="participants to run, every one found otherwise")
    return parser


def wanted(argument, route: str) -> list:
    """The participants a campaign was asked for, intersected with the ones that exist."""
    present = subjects_of(route)
    if argument is None:
        return present
    asked = {f"{int(i):02d}" for i in argument}
    return [subject for subject in present if subject in asked]


def banner(title: str) -> None:
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")


def headless() -> None:
    """
    Force the Agg backend, after the project imports rather than before them.

    Importing cocofest switches matplotlib to TkAgg, so a `matplotlib.use("Agg")` written at the top of a
    module is undone by the very next import and the run then blocks on a window no compute node has.
    Every analysis calls this as the first line of its main().
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
