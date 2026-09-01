"""
Second pass over the isometric c3d recordings, for the observable data_to_muscle_force.py does not keep.

    data/sensor/   the norm of the 3-axis force vector measured at the handle sensor, no biomechanical
                   transport at all

The trains are cut over the same windows as data/force, so that the two observables answer on the same data.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

COMPARISON_ROOT = Path(__file__).resolve().parent.parent
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from helper.force_processing import C3dToForce

CALIBRATION_MATRIX_PATH = str(Path(__file__).resolve().parent / "helper" / "matrix.txt")
OUTPUT_FOLDER = COMPARISON_ROOT / "data" / "sensor"
ELBOW_ANGLE = 90
MUSCLE_NAME = "BIClong"


def sensor_force_of(converter) -> dict:
    """
    The sensor force norm of every train of one recording, cut where the muscle force stops being produced.
    """
    converter.get_data_at_handle()
    converter.select_muscle_and_dof()
    handle = converter.handle_dictionary

    sensor, times, decayed = [], [], []
    for i in range(len(handle["x"])):
        force = np.array([handle["x"][i], handle["y"][i], handle["z"][i]])
        moment = np.array([handle["mx"][i], handle["my"][i], handle["mz"][i]])
        converter.get_muscle_force(
            local_data=np.concatenate(
                (converter.local_sensor_to_local_hand(force), converter.local_sensor_to_local_hand(moment))
            )
        )

        muscle = np.array(converter.muscle_force_vector) - converter.muscle_force_vector[0]
        muscle = np.maximum(muscle, 0)
        end = converter.force_production_end(muscle)
        decayed.append(bool(end < muscle.size))

        cut = np.linalg.norm(force, axis=0)[:end].copy()
        if cut.size:
            cut[-1] = 0.0
        sensor.append(cut)
        times.append(np.asarray(handle["time"][i])[:end])

    return {"sensor": sensor, "time": times, "force_decayed": decayed}


def main(subjects=range(1, 21), overwrite=False):
    participants = pd.read_excel(COMPARISON_ROOT / "data" / "exp" / "data_participants.xlsx")

    for i in subjects:
        subject = f"P{int(i):02d}"
        exp_folder = COMPARISON_ROOT / "data" / "exp" / subject
        saving_folder = OUTPUT_FOLDER / subject
        saving_folder.mkdir(parents=True, exist_ok=True)

        handle_distance = float(participants.loc[participants["participant"] == int(i), "handle_elbow_dist"].values[0])

        for path in sorted(exp_folder.glob("*force*.c3d")):
            written = saving_folder / f"{path.stem}.pkl"
            if not overwrite and written.exists():
                print(f"kept: {subject}/{path.stem}.pkl")
                continue

            converter = C3dToForce(
                c3d_path=str(path),
                calibration_matrix_path=CALIBRATION_MATRIX_PATH,
                saving_pickle_path=str(written),
                frequency_stimulation=int(path.name[10:12]),
                rest_time=1,
                model_path=str(COMPARISON_ROOT / "model" / f"p{int(i):02d}_scaling_scaled.bioMod"),
                elbow_angle=ELBOW_ANGLE,
                muscle_name=MUSCLE_NAME,
                transfer_force=True,
                handle_distance=handle_distance / 100.0,  # the sheet reports centimetres
            )
            extracted = sensor_force_of(converter)

            converter.save_in_pkl(
                {
                    "force": extracted["sensor"],
                    "time": extracted["time"],
                    "frequency": converter.frequency_stimulation,
                    "pulse_width": converter.pulse_width,
                    "stim_time": converter.handle_dictionary["stim_time"],
                    "muscle_name": MUSCLE_NAME,
                    "force_decayed": extracted["force_decayed"],
                    "observable": "sensor",
                    "unit": "N",
                },
                str(written),
            )
            print(f"saved: {subject}/{path.stem}.pkl")


if __name__ == "__main__":
    main()
