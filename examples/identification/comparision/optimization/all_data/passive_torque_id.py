"""
Identify the passive joint torque on ALL the relaxation curves of a subject.
"""

import sys
from pathlib import Path

OPTIMIZATION_ROOT = Path(__file__).resolve().parent.parent
if str(OPTIMIZATION_ROOT) not in sys.path:
    sys.path.insert(0, str(OPTIMIZATION_ROOT))
import passive_ocp
from helper import debug_plots

METHOD = "passive_torque_id_all"


def main(
    subjects=range(1, 21),
    conditions=None,
    formulation="riener",
    plot=False,
    save=True,
    debug=True,
    show_debug=False,
    max_iter=1000,
):
    """
    Parameters
    ----------
    conditions: list[tuple[int, bool]]
        The (frequency, weight) recordings to read. Defaults to every recording of the subject.
    formulation: str
        "double_exponential" or "riener"
    plot: bool
        The solver's own penalty plot. Off by default so a whole cohort runs unattended.
    debug: bool
        Write the debug figures under results/debug/<method>/P<subject>. On by default, and independent of
        whether they are shown.
    show_debug: bool
        Open the debug figures on screen too, which blocks the batch on every one of them until it is closed.
    """
    conditions = passive_ocp.ALL_CONDITIONS if conditions is None else conditions
    method = f"{METHOD}_{formulation}"

    for i in subjects:
        subject = f"{int(i):02d}"
        # Before collecting, because the extraction figure is drawn while the phases are being read
        debug_plots.output_for(method, subject, debug=debug, show=show_debug)

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
            method=method,
            formulation=formulation,
            plot=plot,
            save=save,
            debug=debug,
            show_debug=show_debug,
            max_iter=max_iter,
        )


if __name__ == "__main__":
    main()
