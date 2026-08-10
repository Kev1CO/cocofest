"""
Identify the Ding 2007 model on ALL the isometric force curves of a subject at once.
"""

import sys
from pathlib import Path

OPTIMIZATION_ROOT = Path(__file__).resolve().parent.parent
if str(OPTIMIZATION_ROOT) not in sys.path:
    sys.path.insert(0, str(OPTIMIZATION_ROOT))
import force_ocp

METHOD = "force_id_all"


def select(ocp_data, only_decayed=True):
    """
    Keep every train, optionally dropping the ones whose force never came back near zero.

    Parameters
    ----------
    ocp_data: dict
        Every train of the subject
    only_decayed: bool
        Drop the trains that are dominated by drift rather than by a stimulation response. Those carry no
        information about the force decay and pull the fit, see C3dToForce.force_production_end.

    Returns
    -------
    dict
        The ocp data restricted to the kept trains
    """
    keep = list(range(len(ocp_data["force"])))
    if only_decayed and "force_decayed" in ocp_data:
        keep = [i for i in keep if ocp_data["force_decayed"][i]]
    return force_ocp.select_trains(ocp_data, keep)


def main(
    subjects=range(1, 21),
    only_decayed=True,
    data_folder="force",
    plot=True,
    save=True,
    debug=True,
    max_iter=1000,
):
    for i in subjects:
        subject = f"{int(i):02d}"
        ocp_data = force_ocp.load_force_data(subject, data_folder=data_folder)
        if ocp_data is None:
            print(f"P{subject}: no processed force data in data/{data_folder}, skipping.")
            continue

        force_ocp.identify(
            subject=subject,
            ocp_data=select(ocp_data, only_decayed=only_decayed),
            method=METHOD,
            plot=plot,
            save=save,
            debug=debug,
            max_iter=max_iter,
        )


if __name__ == "__main__":
    main()
