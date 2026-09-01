"""
Converts every motion c3d recording of the experiment into the flexion profiles the movement route tracks,
the way data_to_muscle_force.py does for the isometric trains.

    data/movement/   one file per motion recording, holding every usable flexion of it: the measured elbow
                     angle cut at the return, the velocity and acceleration differentiated over the whole
                     recording, and the train that drove each of them

The c3d files of data/exp are the only source of truth, everything under data/movement is rebuilt by this
script. The whole recording is saved along with the flexions, so that the extraction figure of the
identification can be redrawn without reading the c3d again.
"""

import sys
from pathlib import Path

COMPARISON_ROOT = Path(__file__).resolve().parent.parent
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from helper import movement_processing  # noqa: E402

OUTPUT_FOLDER = movement_processing.MOVEMENT_FOLDER


def main(subjects=range(1, 21), frequencies=None, overwrite=False):
    frequencies = movement_processing.NON_WEIGHTED_FREQUENCIES if frequencies is None else frequencies

    for i in subjects:
        subject = f"{int(i):02d}"
        saving_folder = OUTPUT_FOLDER / f"P{subject}"
        saving_folder.mkdir(parents=True, exist_ok=True)

        for frequency in frequencies:
            try:
                path = movement_processing.find_motion_file(subject, frequency)
            except FileNotFoundError as error:
                print(f"  P{subject} {frequency}Hz: unusable ({error}), skipped.")
                continue

            written = saving_folder / f"{path.stem}.pkl"
            if not overwrite and written.exists():
                print(f"kept: P{subject}/{written.name}")
                continue

            try:
                recording, phases = movement_processing.flexions_of(subject, frequency)
            except (FileNotFoundError, ValueError, OSError, KeyError) as error:
                print(f"  P{subject} {frequency}Hz: unusable ({error}), skipped.")
                continue

            movement_processing.save(recording, phases, frequency, written)
            print(f"saved: P{subject}/{written.name} ({len(phases)} flexion(s))")


if __name__ == "__main__":
    main()
