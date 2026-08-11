"""
Identify the Ding 2007 model on ALL the measured elbow movements of a subject.
"""

import sys
from pathlib import Path

OPTIMIZATION_ROOT = Path(__file__).resolve().parent.parent
if str(OPTIMIZATION_ROOT) not in sys.path:
    sys.path.insert(0, str(OPTIMIZATION_ROOT))
import movement_ocp
from helper import debug_plots

METHOD = "movement_id_all"


def main(
    subjects=range(1, 21),
    frequencies=None,
    passive_method="passive_torque_id_all_riener",
    plot=False,
    save=True,
    debug=True,
    show_debug=False,
    max_iter=1000,
):
    """
    Parameters
    ----------
    frequencies: list[int]
        The recordings to read. Defaults to the three unweighted ones (20, 33 and 50 Hz).
    passive_method: str
        The stored passive torque identification the articular torque is taken from
    plot: bool
        The solver's own penalty plot. Off by default so a whole cohort runs unattended.
    debug: bool
        Write the debug figures under results/debug/<method>/P<subject>. On by default, and independent of
        whether they are shown.
    show_debug: bool
        Open the debug figures on screen too, which blocks the batch on every one of them until it is closed.
    """
    for i in subjects:
        subject = f"{int(i):02d}"
        # Before collecting, because the extraction figure is drawn while the phases are being read
        debug_plots.output_for(METHOD, subject, debug=debug, show=show_debug)

        phases = movement_ocp.collect_movement_phases(subject, frequencies=frequencies, debug=debug)
        if not phases:
            print(f"P{subject}: no usable movement found, skipping.")
            continue

        movement_ocp.identify(
            subject=subject,
            phases=phases,
            method=METHOD,
            passive_method=passive_method,
            plot=plot,
            save=save,
            debug=debug,
            show_debug=show_debug,
            max_iter=max_iter,
        )


if __name__ == "__main__":
    main()
