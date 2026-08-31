"""
Every analysis, over every participant and every route, in one code run.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from statistics import median

import _shared

import routes as route_table


ROUTES = ("sensor_id", "muscle_force_isometric_id", "movement_id")

SKIP_EXISTING = False
MOVEMENT_SIMULATION_FACTOR = 19


@dataclass(frozen=True)
class Measured:
    """What one route's own stored results say it costs, and how many participants it holds."""

    identification: float
    step2: float
    participants: int


@lru_cache(maxsize=None)
def measured(route: str) -> Measured:
    """
    Median seconds of one identification of `route` and of its step 2, read off the npz it wrote.

    Off the results rather than off numbers typed in here, so an estimate describes the machine it is
    printed on. A route with nothing on disk comes back as nan and no participant.
    """
    table = _shared.cohort(route)
    if table.empty:
        return Measured(float("nan"), float("nan"), 0)
    step2 = [entry.get("step_seconds", {}).get("step2", float("nan")) for entry in table["extra"]]
    return Measured(float(median(table["seconds"])), float(median(step2)), len(table))


def participants_of(route: str, asked) -> int:
    """How many participants a campaign will run on: the ones asked for, or the ones the route holds."""
    return len(asked) if asked else measured(route).participants


@dataclass(frozen=True)
class Program:
    """
    One analysis program and everything this runner has to know about it.

    `routes` says which route names it is handed: the analysed set by default, `()` for a program that
    takes no `--routes` at all, and the whole list for one that reads a route the campaigns leave out.
    A program with no cost declared is a reading one: seconds, no `--subjects`, no `--refresh`.
    """

    answers: str
    arguments: tuple = ()
    routes: tuple | None = None
    identifications: float = 0.0
    step2_solves: float = 0.0
    simulation_seconds: float = 0.0

    @property
    def solves(self) -> bool:
        """Whether it re-identifies or re-simulates, which is what earns it --subjects and --refresh."""
        return bool(self.identifications or self.step2_solves or self.simulation_seconds)

    def seconds(self, route: str, subjects: int) -> float:
        """What it costs on one route, from the times the route itself recorded."""
        if self.simulation_seconds:
            # A movement route is simulated through its own program, which is what makes it the slow one
            heavier = MOVEMENT_SIMULATION_FACTOR if route_table.of(route).family == "movement" else 1
            return self.simulation_seconds * heavier * subjects
        if self.step2_solves:
            return self.step2_solves * measured(route).step2 * subjects
        return self.identifications * measured(route).identification * subjects


# Reading first, then the solving ones cheapest first so a campaign killed early still leaves something
PROGRAMS = {
    "agreement_metrology": Program("ICC graded, SEM, MDC95, Bland-Altman"),
    "prediction_accuracy": Program("%model error, R2, NRMSE, BFR, 95% prediction interval"),
    "parameter_bounds": Program("bounds, dispersion, A between step 2 and step 3"),
    # it reads movement_riener_id's stored results, which costs a second and re-identifies nothing
    # "passive_torque_comparison": Program("movement_id against movement_riener_id",
    #                                      routes=route_table.NAMES),
    "cohort_figures": Program("the cohort figures of every route", routes=()),

    "local_sensitivity": Program("elasticity of the output to each parameter", simulation_seconds=90.0),
    "profile_likelihood": Program("practical identifiability of a pinned parameter",
                                  arguments=("--points", "9"), step2_solves=30),
    "kfold_cross_validation": Program("does the identification survive losing trains?",
                                      arguments=("--scheme", "intensity"), identifications=5),
    "learning_curve": Program("how many trains the protocol needs",
                              arguments=("--repeats", "3"), identifications=13),
    "fast_global_sensitivity": Program("first order and total sensitivity indices",
                                       simulation_seconds=600.0),
    "multistart_robustness": Program("is the optimum reachable from anywhere?",
                                     arguments=("--starts", "20"), identifications=20),
    "bootstrap_confidence": Program("intervals on parameters and predictions",
                                    arguments=("--draws", "30"), identifications=30),
}

READING = tuple(name for name, program in PROGRAMS.items() if not program.solves)
SOLVING = tuple(name for name, program in PROGRAMS.items() if program.solves)


def listing(routes, asked) -> None:
    """What each program answers and what it costs, so a campaign can be planned before it is launched."""
    print("\nreading, nothing re-solved (seconds each)")
    print("-" * 68)
    for name in READING:
        print(f"  {name:38s} {PROGRAMS[name].answers}")

    print(f"\n{'solving, for the compute node':40s} {'per route (h)':>14s}  {'total (h)':>10s}")
    print("-" * 68)
    counts = {route: participants_of(route, asked) for route in routes}
    running = [route for route in routes if counts[route]]
    for route in routes:
        if not counts[route]:
            print(f"  {route} holds no stored result, so it is left out of the estimate")

    grand = 0.0
    for name in SOLVING:
        program = PROGRAMS[name]
        per_route = [program.seconds(route, counts[route]) / 3600 for route in running]
        if not per_route:
            continue
        grand += sum(per_route)
        print(f"  {name:38s} {f'{min(per_route):.1f}-{max(per_route):.1f}':>14s}  {sum(per_route):>10.1f}")
        print(f"  {'':38s} {program.answers}")
    print("-" * 68)
    print(f"  {'total, one process':38s} {'':>14s}  {grand:>10.1f}")

    span = sorted({counts[route] for route in running}) or [0]
    heaviest = max(span)
    people = f"{span[0]}" if len(span) == 1 else f"{span[0]}-{heaviest}"
    print(f"\n{len(running)} routes x {people} participants. Split it with --subjects across jobs: "
          f"{grand / max(heaviest, 1):.1f} h per participant.")


def command(name: str, routes, subjects, refresh: bool) -> list:
    """The command line one program is launched with, as its own interpreter."""
    program = PROGRAMS[name]
    line = [sys.executable, "-u", "-m", f"analysis.{name}"]

    given = routes if program.routes is None else program.routes
    if given:
        line += ["--routes", *given]
    line += list(program.arguments)

    if program.solves:
        if subjects:
            line += ["--subjects", *[str(subject) for subject in subjects]]
        if refresh:
            line += ["--refresh"]
    return line


def run(name: str, routes, subjects, refresh: bool, dry_run: bool) -> bool:
    """One program, in its own process, so neither a leak nor a crash reaches the next one."""
    line = command(name, routes, subjects, refresh)
    print(f"\n{'=' * 100}\n{name}\n{' '.join(line[3:])}\n{'=' * 100}", flush=True)
    if dry_run:
        return True

    started = time.perf_counter()
    finished = subprocess.run(line, cwd=str(_shared.COMPARISON_ROOT))
    elapsed = time.perf_counter() - started
    if finished.returncode != 0:
        print(f"{name}: exited {finished.returncode} after {elapsed:.0f} s", flush=True)
        return False
    print(f"{name}: {elapsed:.0f} s", flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    # The default is the three representatives, but any route the cost tables cover may be asked for
    parser.add_argument("--routes", nargs="+", default=list(ROUTES), choices=list(route_table.ANALYSED),
                        help=f"routes to analyse, {', '.join(ROUTES)} by default")
    parser.add_argument("--subjects", type=int, nargs="+", default=None,
                        help="participants to run, every one found otherwise")
    parser.add_argument("--reading", action="store_true",
                        help="only the programs that re-solve nothing, which take seconds")
    parser.add_argument("--only", nargs="+", default=None, choices=list(PROGRAMS),
                        help="run these programs and nothing else")
    parser.add_argument("--skip", nargs="+", default=[], choices=list(PROGRAMS),
                        help="run everything but these")
    parser.add_argument("--list", action="store_true", help="print what each one costs and run nothing")
    parser.add_argument("--dry-run", action="store_true", help="print the commands and run nothing")
    parser.add_argument("--refresh", action="store_true",
                        help=f"re-run participants already on disk instead of skipping them; "
                             f"SKIP_EXISTING is {SKIP_EXISTING} in this file")
    arguments = parser.parse_args()

    if arguments.list:
        listing(arguments.routes, arguments.subjects)
        return

    wanted = ([name for name in PROGRAMS if name in arguments.only] if arguments.only
              else list(READING) if arguments.reading
              else list(PROGRAMS))
    wanted = [name for name in wanted if name not in arguments.skip]

    # The flag only ever turns skipping off, so a command line cannot silently re-enable it
    refresh = arguments.refresh or not SKIP_EXISTING

    print(f"routes    : {', '.join(arguments.routes)}")
    print(f"subjects  : {arguments.subjects if arguments.subjects else 'every one found'}")
    print(f"analyses  : {', '.join(wanted)}")
    print(f"on disk   : {'recomputed' if refresh else 'skipped, so an interrupted run resumes'}")

    started = time.perf_counter()
    failed = [name for name in wanted
              if not run(name, arguments.routes, arguments.subjects, refresh, arguments.dry_run)]

    print(f"\n{'=' * 100}\ndone in {(time.perf_counter() - started) / 3600:.2f} h\n{'=' * 100}")
    print(f"  {len(wanted) - len(failed)}/{len(wanted)} programs ran")
    if failed:
        print(f"  failed: {', '.join(failed)}")
        print("  re-run this program to resume: every campaign skips the participants already on disk")


if __name__ == "__main__":
    main()
