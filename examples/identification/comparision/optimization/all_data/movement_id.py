"""
Identify the Ding 2007 model on ALL the measured elbow movements of a subject.
"""

import sys
from pathlib import Path

OPTIMIZATION_ROOT = Path(__file__).resolve().parent.parent
if str(OPTIMIZATION_ROOT) not in sys.path:
    sys.path.insert(0, str(OPTIMIZATION_ROOT))
import movement_ocp

METHOD = "movement_id_all"


def main(
    subjects=range(1, 21),
    frequencies=None,
    passive_method="passive_torque_id_all_riener",
    plot=True,
    save=True,
    debug=True,
    max_iter=1000,
):
    """
    Parameters
    ----------
    frequencies: list[int]
        The recordings to read. Defaults to the three unweighted ones (20, 33 and 50 Hz).
    passive_method: str
        The stored passive torque identification the articular torque is taken from
    """
    for i in subjects:
        subject = f"{int(i):02d}"
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
            max_iter=max_iter,
        )


if __name__ == "__main__":
    main()
