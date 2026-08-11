"""
Identify the Ding 2007 model on ALL the isometric force curves of a subject at once.
"""

import sys
from pathlib import Path

OPTIMIZATION_ROOT = Path(__file__).resolve().parent.parent
if str(OPTIMIZATION_ROOT) not in sys.path:
    sys.path.insert(0, str(OPTIMIZATION_ROOT))
import force_ocp
from helper import debug_plots

METHOD = "force_id_all"


def select(ocp_data, only_decayed=True, only_complete=True, subject=None):
    """
    Keep every train, optionally dropping the ones that carry no usable force response.

    Parameters
    ----------
    ocp_data: dict
        Every train of the subject
    only_decayed: bool
        Drop the trains that are dominated by drift rather than by a stimulation response. Those carry no
        information about the force decay and pull the fit, see C3dToForce.force_production_end.
    only_complete: bool
        Drop the trains whose force recording stops before the last pulse of the train was delivered. The model
        is then asked to produce a full train inside a window where the force never developed, which no parameter
        set satisfies, and the ocp stops converging. See DataToOCP.get_data_for_ocp.
    subject: str
        Only used to name the dropped trains in the printed report

    Returns
    -------
    dict
        The ocp data restricted to the kept trains
    """
    keep, dropped = [], []
    for i in range(len(ocp_data["force"])):
        if only_decayed and not ocp_data.get("force_decayed", [True] * len(ocp_data["force"]))[i]:
            dropped.append((i, "force never decayed"))
        elif only_complete and not ocp_data.get("covers_stimulation", [True] * len(ocp_data["force"]))[i]:
            dropped.append((i, "recording stops before the last pulse"))
        else:
            keep.append(i)

    for i, reason in dropped:
        print(f"  {'P' + subject + ' ' if subject else ''}train {i} "
              f"({ocp_data['frequency'][i]}Hz, {ocp_data['pulse_width'][i] * 1e6:.0f}us) dropped: {reason}")

    return force_ocp.select_trains(ocp_data, keep)


def main(
    subjects=range(1, 21),
    only_decayed=True,
    only_complete=True,
    data_folder="force",
    plot=False,
    save=True,
    debug=True,
    show_debug=False,
    max_iter=1000,
):
    """
    Parameters
    ----------
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
        debug_plots.output_for(METHOD, subject, debug=debug, show=show_debug)

        ocp_data = force_ocp.load_force_data(subject, data_folder=data_folder)
        if ocp_data is None:
            print(f"P{subject}: no processed force data in data/{data_folder}, skipping.")
            continue

        force_ocp.identify(
            subject=subject,
            ocp_data=select(ocp_data, only_decayed=only_decayed, only_complete=only_complete, subject=subject),
            method=METHOD,
            plot=plot,
            save=save,
            debug=debug,
            show_debug=show_debug,
            max_iter=max_iter,
        )


if __name__ == "__main__":
    main()
