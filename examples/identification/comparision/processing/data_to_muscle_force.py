"""
Converts every isometric c3d recording of the experiment into the muscle force pickle files used by
optimization/force_id.py.

The c3d files of data/exp are the only source of truth, everything under data/force is rebuilt by this script.
"""

import sys
from pathlib import Path

import pandas as pd

COMPARISON_ROOT = Path(__file__).resolve().parent.parent
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from helper.force_processing import C3dToForce  # noqa: E402

CALIBRATION_MATRIX_PATH = str(Path(__file__).resolve().parent / "helper" / "matrix.txt")
OUTPUT_FOLDER = COMPARISON_ROOT / "data" / "force"
ELBOW_ANGLE = 90
MUSCLE_NAME = "BIClong"


def main(subjects=range(1, 21), plot=False):
    participants = pd.read_excel(COMPARISON_ROOT / "data" / "exp" / "data_participants.xlsx")

    for i in subjects:
        subject = f"P{int(i):02d}"
        exp_folder = COMPARISON_ROOT / "data" / "exp" / subject
        saving_folder = OUTPUT_FOLDER / subject
        saving_folder.mkdir(parents=True, exist_ok=True)

        # Distance from the elbow to the handle, where the 6D sensor load is applied
        handle_distance = float(participants.loc[participants["participant"] == int(i), "handle_elbow_dist"].values[0])

        for path in sorted(exp_folder.glob("*force*.c3d")):
            force_converter = C3dToForce(
                c3d_path=str(path),
                calibration_matrix_path=CALIBRATION_MATRIX_PATH,
                saving_pickle_path=str(saving_folder / f"{path.stem}.pkl"),
                frequency_stimulation=int(path.name[10:12]),
                rest_time=1,
                model_path=str(COMPARISON_ROOT / "model" / f"p{int(i):02d}_scaling_scaled.bioMod"),
                elbow_angle=ELBOW_ANGLE,
                muscle_name=MUSCLE_NAME,
                transfer_force=True,
                handle_distance=handle_distance / 100.0,  # the sheet reports centimetres
            )
            force_converter.get_force(save=True, plot=plot)
            print(f"saved: {subject}/{path.stem}.pkl")


if __name__ == "__main__":
    main()
