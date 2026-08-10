"""
Identify the passive joint torque on ALL the relaxation curves of a subject.
"""

import sys
from pathlib import Path

OPTIMIZATION_ROOT = Path(__file__).resolve().parent.parent
if str(OPTIMIZATION_ROOT) not in sys.path:
    sys.path.insert(0, str(OPTIMIZATION_ROOT))
import passive_ocp

METHOD = "passive_torque_id_all"


def main(
    subjects=range(1, 21),
    conditions=None,
    formulation="riener",
    plot=True,
    save=True,
    debug=True,
    max_iter=1000,
):
    """
    Parameters
    ----------
    conditions: list[tuple[int, bool]]
        The (frequency, weight) recordings to read. Defaults to every recording of the subject.
    formulation: str
        "double_exponential" or "riener"
    """
    conditions = passive_ocp.ALL_CONDITIONS if conditions is None else conditions

    for i in subjects:
        subject = f"{int(i):02d}"
        global_q, global_final_time, global_time = passive_ocp.collect_relaxation_phases(
            subject, conditions=conditions, debug=debug
        )
        if not global_q:
            print(f"P{subject}: no usable relaxation phase found, skipping.")
            continue

        passive_ocp.identify(
            subject=subject,
            global_q=global_q,
            global_final_time=global_final_time,
            global_time=global_time,
            method=f"{METHOD}_{formulation}",
            formulation=formulation,
            plot=plot,
            save=save,
            debug=debug,
            max_iter=max_iter,
        )


if __name__ == "__main__":
    main()
