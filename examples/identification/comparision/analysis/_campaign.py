"""
What the analyses share across all the campaign.
"""

from __future__ import annotations

import time
import traceback

import numpy as np
import pandas as pd

import _shared

import protocol
import routes as route_table
import steps


# ---- The loop every solving campaign runs ---- #
def per_subject(program: str, arguments, run, tag_of=None) -> pd.DataFrame:
    """
    Run `run(route, subject)` over every asked route and participant, skipping what is already on disk.
    """
    tag_of = tag_of or (lambda route, subject: f"{route}_p{subject}")

    frames = []
    for name in _shared.available(arguments.routes):
        route = route_table.of(name)
        for subject in _shared.wanted(arguments.subjects, name):
            tag = tag_of(name, subject)
            done = None if arguments.refresh else _shared.stored(program, tag)
            if done is not None:
                print(f"{tag}: already on disk")
                frames.append(done)
                continue

            _shared.banner(f"{name} P{subject}")
            try:
                detail = run(route, subject)
            except Exception as error:
                print(f"  {tag}: {type(error).__name__}: {error}", flush=True)
                continue

            if detail.empty:
                print(f"  {tag}: produced no row, left for a later run", flush=True)
                continue
            _shared.write(detail, program, tag)
            frames.append(detail)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---- Running one identification ---- #
def identify(route, subject: str, **kwargs) -> dict:
    """
    One identification, with nothing written to the routes' results and no figure drawn.
    """
    started = time.perf_counter()
    try:
        outcome = steps.identify(route, subject, save=False, debug=False, **kwargs)
        outcome["failed"] = ""
        return outcome
    except Exception as error:
        print(f"  P{subject}: {type(error).__name__}: {error}", flush=True)
        return {"subject": subject, "failed": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(), "seconds": time.perf_counter() - started,
                "parameters": {}, "trains": pd.DataFrame(), "converged": False}


def scores(outcome: dict, which: str = "testing") -> dict:
    """The prediction scores of one identification, averaged over the trains of one set."""
    table = outcome.get("trains")
    if table is None or len(table) == 0:
        return {}
    picked = table[table["set"] == which]
    if picked.empty:
        return {}
    return {
        f"{which}_n": int(len(picked)),
        f"{which}_peak_error_pct": float(picked["peak_error"].mean()),
        f"{which}_fti_error_pct": float(picked["fti_error"].mean()),
        f"{which}_r2": float(picked["r2"].mean()),
        f"{which}_rmse": float(picked["rmse"].mean()),
    }


def flat(outcome: dict, **extra) -> dict:
    """One identification squeezed into a row: what it identified, what it scored, what it cost."""
    row = {"subject": outcome.get("subject", ""), **extra}
    row.update({name: float(value) for name, value in (outcome.get("parameters") or {}).items()})
    row.update(scores(outcome, "testing"))
    row.update(scores(outcome, "identification"))
    row["converged"] = bool(outcome.get("converged", False))
    row["at_bound"] = "|".join(outcome.get("at_bound", []))
    row["seconds"] = float(outcome.get("seconds", np.nan))
    row["rmse"] = float(outcome.get("rmse", np.nan))
    row["step2_cost"] = float(outcome.get("step2_cost", np.nan))
    row["eq3_r2"] = float((outcome.get("relationship") or {}).get("r2", np.nan))
    row["failed"] = outcome.get("failed", "")
    return row


# ---- Reshaping the protocol table ---- #
def cells(role: str = protocol.IDENTIFICATION) -> list:
    """The (frequency, pulse width) cells of the study's design that carry one role, in a stable order."""
    return [(frequency, pulse_width)
            for frequency, row in protocol.DESIGN.items()
            for pulse_width, cell in row.items() if cell == role]


def design_with(to_testing=(), to_dropped=()) -> dict:
    """
    A copy of the study's design with some cells moved.
    """
    design = {frequency: dict(row) for frequency, row in protocol.DESIGN.items()}
    for frequency, pulse_width in to_testing:
        design[frequency][pulse_width] = protocol.TESTING
    for frequency, pulse_width in to_dropped:
        design[frequency][pulse_width] = protocol.DROPPED
    return design


def describe(design: dict) -> str:
    """The design of one fold or one curve point, on one line, so a log says what was run."""
    counts = {}
    for row in design.values():
        for cell in row.values():
            counts[cell] = counts.get(cell, 0) + 1
    return " ".join(f"{cell}:{counts[cell]}" for cell in sorted(counts))


def folds(scheme: str, k: int = 5, seed: int = 0) -> list:
    """
    The held-out groups of a cross validation, as lists of (frequency, pulse width) cells.

        train      the 10 non-reference identification trains split into k folds
        intensity  both identification trains of one pulse width held out at a time, 5 folds
        frequency  every identification train of one frequency held out at a time, 3 folds
    """
    available = cells(protocol.IDENTIFICATION)

    if scheme in ("intensity", "frequency"):
        index = 1 if scheme == "intensity" else 0
        groups = {}
        for cell in available:
            groups.setdefault(cell[index], []).append(cell)
        return [groups[key] for key in sorted(groups)]

    if scheme == "train":
        order = np.random.default_rng(seed).permutation(len(available))
        return [[available[i] for i in chunk] for chunk in np.array_split(order, min(k, len(available)))]

    raise ValueError(f"unknown scheme {scheme!r}, expected train, intensity or frequency")
