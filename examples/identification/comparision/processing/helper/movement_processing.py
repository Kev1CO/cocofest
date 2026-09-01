"""
Cuts the motion c3d recordings of the experiment into the elbow flexions the movement route tracks.

One flexion runs from the first pulse of a train to the elbow settling back down: `movement_slicing` cuts it,
`recording_derivatives` differentiates the whole recording once, and `trim_to_return` drops the tail where the
muscle is producing nothing. `data_to_movement.py` writes what comes out to data/movement, `load` reads it back.
"""

from __future__ import annotations

import pickle
import re
import sys
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

COMPARISON_ROOT = Path(__file__).resolve().parents[2]
if str(COMPARISON_ROOT) not in sys.path:
    sys.path.insert(0, str(COMPARISON_ROOT))
from processing.helper.c3d_to_q import C3dToQ  # noqa: E402

SEEDS_PULSE_WIDTH_PATH = Path(__file__).resolve().parent / "seeds_pulse_width.pkl"
MOVEMENT_FOLDER = COMPARISON_ROOT / "data" / "movement"
NON_WEIGHTED_FREQUENCIES = [20, 33, 50]
# A train holding less than this fraction of the recording's median pulse count was cut short by the stimulator
TRUNCATED_TRAIN_FRACTION = 0.5
SETTLED_SPEED_FRACTION = 0.05
SETTLED_QUIET_DURATION = 0.15
DERIVATIVE_WINDOW = 21
DERIVATIVE_ORDER = 3


def find_motion_file(subject, frequency) -> Path:
    """The motion recording of a subject at a given frequency, without the added weight."""
    folder = COMPARISON_ROOT / "data" / "exp" / f"P{subject}"
    matching = [
        path for path in sorted(folder.glob(f"p{subject}_motion*{frequency}Hz*.c3d")) if "weight" not in path.stem
    ]
    if not matching:
        raise FileNotFoundError(f"No unweighted {frequency}Hz motion recording for P{subject} in {folder}.")
    return matching[0]


def pulse_widths_of(file) -> list:
    """
    The pulse width applied to each train of a recording, in seconds.
    """
    with open(SEEDS_PULSE_WIDTH_PATH, "rb") as handle:
        seeds = pickle.load(handle)
    seed = int(re.search(r"_(\d+)\.c3d$", file.name).group(1))
    return [float(value) / 1e6 for value in seeds[seed]]


def settled_end(q, time, speed_fraction=SETTLED_SPEED_FRACTION, quiet_duration=SETTLED_QUIET_DURATION, search_from=0):
    """
    Index at which the joint has settled: the end of the first stretch where its speed stays below
    `speed_fraction` of the phase's speed peak for `quiet_duration` seconds.

    Parameters
    ----------
    search_from: int
        Ignore any quiet stretch starting before this index. A relaxation begins at the top of the motion where
        the joint is momentarily still, and a full rise-and-fall phase begins before the stimulation has built
        any force: without this the search would settle on that opening stillness and cut the phase immediately.
    """
    q, time = np.asarray(q, dtype=float), np.asarray(time, dtype=float)
    if len(q) < 5:
        return len(q)

    speed = np.abs(np.gradient(q, time))
    peak = float(speed.max())
    if peak <= 0:
        return len(q)

    dt = float(np.median(np.diff(time)))
    window = max(3, int(round(quiet_duration / dt)))
    if window >= len(speed):
        return len(q)

    quiet = (speed < speed_fraction * peak).astype(float)
    sustained = np.convolve(quiet, np.ones(window), mode="valid")
    settled = np.where(sustained >= window)[0]
    settled = settled[settled >= search_from]
    return len(q) if settled.size == 0 else min(len(q), int(settled[0]) + window)


def movement_slicing(q, time, stim_time):
    """
    Cut the whole movement driven by each stimulation train.

    Parameters
    ----------
    q: np.ndarray
        The elbow angle of the whole recording (rad)
    time: np.ndarray
        The time vector of the whole recording (s)
    stim_time: list
        The stimulation times, grouped per train

    Returns
    -------
    tuple
        The elbow angle, the time vector and the train stimulation times (from the phase start) of every movement
    """
    pulses = [len(train) for train in stim_time if len(train)]
    complete_train = TRUNCATED_TRAIN_FRACTION * float(np.median(pulses)) if pulses else 0

    phases_q, phases_time, phases_stim, kept_trains = [], [], [], []
    for train_index, train in enumerate(stim_time):
        if len(train) == 0:
            continue
        if len(train) < complete_train:
            print(f"  train {train_index} dropped: {len(train)} pulses against a median of {np.median(pulses):.0f}")
            continue

        start = int(np.searchsorted(time, train[0]))
        next_train = stim_time[train_index + 1] if train_index + 1 < len(stim_time) else []
        stop = int(np.searchsorted(time, next_train[0])) if len(next_train) else len(time)
        if stop - start < 5:
            continue

        segment = slice(start, stop)
        segment_q = q[segment]

        peak_index = int(np.argmax(segment_q))
        half_way = segment_q[peak_index] - 0.5 * (segment_q[peak_index] - segment_q[0])
        descended = np.where(segment_q[peak_index:] <= half_way)[0]
        search_from = peak_index + int(descended[0]) if descended.size else peak_index

        end = settled_end(segment_q, time[segment], search_from=search_from)
        if end < 5:
            continue

        phases_q.append(q[segment][:end])
        phases_time.append(time[segment][:end])
        phases_stim.append(np.asarray(train) - time[start])
        kept_trains.append(train_index)

    return phases_q, phases_time, phases_stim, kept_trains


def recording_derivatives(phases) -> list:
    """
    The elbow velocity and acceleration of every flexion, differentiated over its whole recording.
    """
    time = np.concatenate([np.asarray(phase["time"], dtype=float) for phase in phases])
    q = np.concatenate([np.asarray(phase["q"], dtype=float) for phase in phases])

    dt = float(np.median(np.diff(time)))
    window = min(DERIVATIVE_WINDOW, len(q) - (1 - len(q) % 2))
    order = min(DERIVATIVE_ORDER, window - 1)
    qdot = savgol_filter(q, window, order, deriv=1, delta=dt)
    qddot = savgol_filter(q, window, order, deriv=2, delta=dt)

    start = 0
    for phase in phases:
        stop = start + len(phase["q"])
        phase["full_time"] = np.asarray(phase["time"], dtype=float)
        phase["full_qdot"], phase["full_qddot"] = qdot[start:stop], qddot[start:stop]
        start = stop

    return phases


def trim_to_return(phases) -> list:
    """
    Cut each flexion where the elbow reaches its lowest angle again, not where the next train starts.

    movement_slicing slices from one first pulse to the next, so the window carries the tail of the return
    where the muscle is producing nothing: weight in the cost with no information in it.
    """
    for phase in phases:
        q = np.asarray(phase["q"], dtype=float)
        peak = int(np.argmax(q))
        end = peak + int(np.argmin(q[peak:])) + 1
        # nothing to cut when the movement is still descending at the window edge
        if end < 5 or end >= len(q):
            continue
        phase["q"] = q[:end]
        phase["time"] = np.asarray(phase["time"], dtype=float)[:end]

    for phase in phases:
        phase["final_time"] = float(phase["time"][-1] - phase["time"][0])
    return phases


def flexions_of(subject, frequency) -> tuple:
    """
    The flexions of one motion recording, rise and fall, with the pulse width that drove each of them.

    Returns
    -------
    tuple
        The whole recording as debug_plots.plot_phase_extraction takes it, and one dict per flexion holding
        "q", "time", "full_time", "full_qdot", "full_qddot", "stim_time", "pulse_width", "frequency" and
        the index of the train that drove it
    """
    file = find_motion_file(subject, frequency)
    print(f"Loading file: {file}")

    converter = C3dToQ(str(file))
    converter.frequency_stimulation = frequency
    stim_time = converter.get_sliced_stim_time()["stim_time"]
    q_rad, time, phase_stim, kept = movement_slicing(converter.get_q_rad(), converter.get_time(), stim_time)
    pulse_widths = pulse_widths_of(file)

    phases = [
        {
            "q": q_rad[i],
            "time": time[i],
            "stim_time": phase_stim[i],
            "pulse_width": pulse_widths[kept[i] % len(pulse_widths)],
            "frequency": frequency,
            "train": kept[i],
        }
        for i in range(len(q_rad))
    ]
    if phases:
        trim_to_return(recording_derivatives(phases))

    recording = {
        "full_time": converter.get_time(),
        "full_data": converter.get_q_rad(),
        "phase_times": [phase["time"] for phase in phases],
        "phase_data": [phase["q"] for phase in phases],
        "stim_times": stim_time,
        "label": f"{frequency} Hz",
    }
    return recording, phases


def saved_path(subject, frequency) -> Path:
    """Where the flexions of one motion recording are stored, the c3d name with a pkl extension."""
    folder = MOVEMENT_FOLDER / f"P{subject}"
    matching = [
        path for path in sorted(folder.glob(f"p{subject}_motion*{frequency}Hz*.pkl")) if "weight" not in path.stem
    ]
    if not matching:
        raise FileNotFoundError(f"No saved {frequency}Hz flexion for P{subject} in {folder}, "
                                f"run processing/data_to_movement.py")
    return matching[0]


def save(recording, phases, frequency, path) -> dict:
    """Write the flexions of one recording to data/movement, one list per entry as data/force holds them."""
    data = {
        "frequency": frequency,
        "q": [phase["q"] for phase in phases],
        "time": [phase["time"] for phase in phases],
        # The window the derivatives span, before trim_to_return cut the return off "q" and "time"
        "full_time": [phase["full_time"] for phase in phases],
        "full_qdot": [phase["full_qdot"] for phase in phases],
        "full_qddot": [phase["full_qddot"] for phase in phases],
        "stim_time": [phase["stim_time"] for phase in phases],
        "pulse_width": [phase["pulse_width"] for phase in phases],
        "train": [phase["train"] for phase in phases],
        "recording_time": np.asarray(recording["full_time"], dtype=float),
        "recording_q": np.asarray(recording["full_data"], dtype=float),
        "recording_stim_time": recording["stim_times"],
        "observable": "movement",
        "unit": "rad",
    }
    C3dToQ.save_in_pkl(data, str(path))
    return data


def load(subject, frequencies=None) -> tuple:
    """
    The saved flexions of one subject, laid out on one continuous time base.

    Returns
    -------
    tuple
        One dict per flexion, in the order the recordings are listed, and one panel per recording for the
        extraction figure.
    """
    frequencies = NON_WEIGHTED_FREQUENCIES if frequencies is None else frequencies

    phases, panels = [], []
    for frequency in frequencies:
        try:
            path = saved_path(subject, frequency)
            with open(path, "rb") as handle:
                data = pickle.load(handle)
        except (FileNotFoundError, OSError) as error:
            print(f"  P{subject} {frequency}Hz: unusable ({error}), skipped.")
            continue

        recording_phases = [
            {
                "q": np.asarray(data["q"][i], dtype=float),
                "time": np.asarray(data["time"][i], dtype=float),
                "full_time": np.asarray(data["full_time"][i], dtype=float),
                "full_qdot": np.asarray(data["full_qdot"][i], dtype=float),
                "full_qddot": np.asarray(data["full_qddot"][i], dtype=float),
                "stim_time": np.asarray(data["stim_time"][i], dtype=float),
                "pulse_width": float(data["pulse_width"][i]),
                "frequency": data["frequency"],
            }
            for i in range(len(data["q"]))
        ]
        phases += recording_phases
        panels.append(
            {
                "full_time": data["recording_time"],
                "full_data": data["recording_q"],
                "phase_times": [phase["time"] for phase in recording_phases],
                "phase_data": [phase["q"] for phase in recording_phases],
                "stim_times": data["recording_stim_time"],
                "label": f"{data['frequency']} Hz",
            }
        )

    offset = 0.0
    for phase in phases:
        phase["final_time"] = float(phase["time"][-1] - phase["time"][0])
        phase["global_time"] = phase["time"] - phase["time"][0] + offset
        phase["time_offset"] = offset
        offset += phase["final_time"]
    return phases, panels
