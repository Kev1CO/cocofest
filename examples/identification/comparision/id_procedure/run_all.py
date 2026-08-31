"""
The whole identification in one command: the routes over the cohort, then the figures that read them.

    python run_all.py                        every route on every participant, then the cohort figures
    python run_all.py --subjects 1 2         the same on a subset
    python run_all.py --routes sensor_id     one route on its own
    python run_all.py --figures              redraw the figures alone, solving nothing
    python run_all.py --check 1              one participant, nothing written, to see the package runs

Each route writes results/<route>/p<subject>_<route>.npz and its figures under results/debug/<route>/.
"""

from __future__ import annotations

import argparse
import time

import store

import routes as route_table


def run(names=route_table.NAMES, subjects=range(1, 21), save=True, debug=True, **kwargs):
    """
    Run the given routes over the given participants, one route at a time.

    One bad participant never takes the rest down: steps.main catches per subject, and a route that raises
    outright is reported and skipped rather than ending the batch.
    """
    import steps

    for name in names:
        print(f"\n{'=' * 70}\n{name}\n{'=' * 70}", flush=True)
        started = time.perf_counter()
        try:
            steps.main(route_table.of(name), subjects=subjects, save=save, debug=debug, **kwargs)
        except Exception as error:
            print(f"{name}: {type(error).__name__}: {error}", flush=True)
            continue
        print(f"{name}: {time.perf_counter() - started:.0f} s", flush=True)


def figures():
    """Redraw the cohort figures from what the routes stored, solving nothing."""
    from analysis import cohort_figures

    print(f"\n{'=' * 70}\ncohort figures\n{'=' * 70}", flush=True)
    cohort_figures.main()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 21)),
                        help="participants to run, 1 to 20 by default")
    parser.add_argument("--routes", nargs="+", default=list(route_table.NAMES),
                        choices=list(route_table.NAMES), help="routes to run, all seven by default")
    parser.add_argument("--figures", action="store_true", help="redraw the figures alone")
    parser.add_argument("--check", type=int, metavar="SUBJECT",
                        help="run one participant through every route without writing anything")
    arguments = parser.parse_args()

    if arguments.figures:
        figures()
        return
    if arguments.check is not None:
        run(names=arguments.routes, subjects=[arguments.check], save=False, debug=False)
        return

    run(names=arguments.routes, subjects=arguments.subjects)
    figures()


if __name__ == "__main__":
    main()
