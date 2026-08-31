"""
The measured quantity every method identifies on.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize
from scipy.signal import savgol_filter

import store

import biorbd

import force_ocp
import movement_ocp
import protocol


ISOMETRIC_ELBOW_ANGLE = np.deg2rad(90.0)  # The elbow angle the isometric trials were recorded at
ISOMETRIC_FOLDERS = {"muscle": "force", "sensor": "sensor", "torque": "torque"}
N_SHOOTING_PER_PHASE = 90
ID_BASELINE = "none"
DERIVATIVE_WINDOW = 21
DERIVATIVE_ORDER = 3


def model_path(subject) -> str:
    """The subject's scaled bioMod, the model both the isometric transport and the movement route use."""
    return str(store.MODEL_ROOT / f"p{subject}_scaling_scaled.bioMod")


def muscle_index(model) -> int:
    """Where the biceps sits in the muscle list of a biorbd model."""
    return [model.muscleNames()[i].to_string() for i in range(model.nbMuscles())].index(movement_ocp.MUSCLE_NAME)


# ---- The isometric trains ---- #
def isometric_phases(subject, observable="muscle") -> list:
    """
    Every usable stimulation train of one subject, with the chosen observable as the tracked target.

    "muscle" is the force carried into the biceps, "sensor" the norm of the handle force vector, "torque"
    the elbow torque that load produces. The three are cut on the same trains over the same windows, see
    processing/data_to_observables.py. One phase per train, the elbow held at the recording angle.
    """
    ocp_data = force_ocp.load_force_data(subject, data_folder=ISOMETRIC_FOLDERS[observable])
    if ocp_data is None:
        raise FileNotFoundError(f"No {observable} recording for P{subject}, run processing/data_to_observables.py")

    unit = "Nm" if observable == "torque" else "N"
    model = biorbd.Model(model_path(subject))
    arm = float(model.musclesLengthJacobian(np.array([ISOMETRIC_ELBOW_ANGLE])).to_array()[muscle_index(model)][0])
    phases = []
    for i in range(len(ocp_data["force"])):
        if not ocp_data.get("force_decayed", [True] * len(ocp_data["force"]))[i]:
            reason = "force never decayed"
        elif not ocp_data.get("covers_stimulation", [True] * len(ocp_data["force"]))[i]:
            reason = "recording stops before the last pulse"
        else:
            n_shooting, final_time = ocp_data["n_shooting"][i], ocp_data["final_time"][i]
            phases.append(
                {
                    "frequency": ocp_data["frequency"][i],
                    "pulse_width": ocp_data["pulse_width"][i],
                    "stim_time": ocp_data["stim_time"][i],
                    "n_shooting": n_shooting,
                    "final_time": final_time,
                    "time": np.linspace(0, final_time, n_shooting + 1),
                    "target": np.asarray(ocp_data["force"][i][0], dtype=float),
                    "q": np.full(n_shooting + 1, ISOMETRIC_ELBOW_ANGLE),
                    "qdot0": 0.0,
                    "moment_arm": np.full(n_shooting + 1, arm),
                    "unit": unit,
                }
            )
            continue
        print(f"  P{subject} {protocol.condition_name(ocp_data['frequency'][i], ocp_data['pulse_width'][i])} "
              f"dropped: {reason}")

    return _lay_end_to_end(phases)


# ---- The recorded flexions ---- #
def movement_phases(subject, frequencies=None, observable="angle", debug=False) -> list:
    """
    Every usable elbow flexion of one subject, on a node grid aligned on its stimulation period.

    "angle" tracks the measured elbow angle and lets the skeleton move; "force" and "torque" track what the
    inverse dynamics of the measured movement produces and hold the skeleton on that very movement.
    """
    raw = movement_ocp.collect_movement_phases(subject, frequencies=frequencies, debug=debug)
    _recording_derivatives(raw)
    _trim_to_return(raw)
    model = biorbd.Model(model_path(subject))
    index = muscle_index(model)

    phases = []
    for phase in raw:
        n_shooting, final_time = movement_ocp.aligned_shooting_grid(
            phase["final_time"], phase["stim_time"], N_SHOOTING_PER_PHASE
        )
        time = np.linspace(0, final_time, n_shooting + 1)
        measured_time = np.asarray(phase["time"], dtype=float) - phase["time"][0]
        q = np.interp(time, measured_time, np.asarray(phase["q"], dtype=float))
        full_time = phase["full_time"] - phase["full_time"][0]
        qdot = np.interp(time, full_time, phase["full_qdot"])
        qddot = np.interp(time, full_time, phase["full_qddot"])

        torque, moment_arm = _inverse_dynamics(model, index, q, qdot, qddot)
        if ID_BASELINE == "static":
            torque = torque - _static_torque(model, q[0])
        # A stimulated muscle cannot pull the joint backwards, the same rule the muscle force chain applies
        torque = np.maximum(torque, 0)
        force = _distribute(model, index, q, torque)

        target, unit = {
            "angle": (q, "rad"),
            "force": (force, "N"),
            "torque": (torque, "Nm"),
        }[observable]

        if observable != "angle":
            end = _end_of_production(target, int(np.searchsorted(time, phase["stim_time"][-1])))
            n_shooting, final_time = end - 1, float(time[end - 1])
            time, target, q, moment_arm = time[:end], target[:end], q[:end], moment_arm[:end]

        phases.append(
            {
                "frequency": phase["frequency"],
                "pulse_width": phase["pulse_width"],
                "stim_time": phase["stim_time"],
                "n_shooting": n_shooting,
                "final_time": final_time,
                "time": time,
                "target": np.asarray(target, dtype=float),
                "q": None if observable == "angle" else q,
                "qdot0": 0.0 if observable == "angle" else float(qdot[0]),
                "measured_q": q,
                "moment_arm": moment_arm,
                "unit": unit,
            }
        )

    return _lay_end_to_end(phases)


def _trim_to_return(phases) -> list:
    """
    Cut each flexion where the elbow reaches its lowest angle again, not where the next train starts.

    movement_ocp slices from one first pulse to the next, so the window carries the tail of the return
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


def _recording_derivatives(phases) -> list:
    """
    The elbow velocity and acceleration of every flexion, differentiated over its whole recording.
    """
    for frequency in sorted({phase["frequency"] for phase in phases}):
        recording = [phase for phase in phases if phase["frequency"] == frequency]
        time = np.concatenate([np.asarray(phase["time"], dtype=float) for phase in recording])
        q = np.concatenate([np.asarray(phase["q"], dtype=float) for phase in recording])

        dt = float(np.median(np.diff(time)))
        window = min(DERIVATIVE_WINDOW, len(q) - (1 - len(q) % 2))
        order = min(DERIVATIVE_ORDER, window - 1)
        qdot = savgol_filter(q, window, order, deriv=1, delta=dt)
        qddot = savgol_filter(q, window, order, deriv=2, delta=dt)

        start = 0
        for phase in recording:
            stop = start + len(phase["q"])
            phase["full_time"] = np.asarray(phase["time"], dtype=float)
            phase["full_qdot"], phase["full_qddot"] = qdot[start:stop], qddot[start:stop]
            start = stop

    return phases


# The fewest nodes a cut phase may keep, below which the trim is refused rather than trusted
MINIMUM_NODES = 20

# How far above the bottom of the relaxation the target has to climb to count as rising again, as a
# fraction of the phase's own range
RISE_TOLERANCE = 0.02


def _end_of_production(target, last_pulse) -> int:
    """
    Where a flexion stops being about the stimulation, on the inverse dynamics targets.
    """
    after = target[last_pulse:]
    if after.size == 0:
        return len(target)

    floor, bottom = float(after.min()), int(np.argmin(after))
    rising = np.where(after[bottom:] > floor + RISE_TOLERANCE * max(float(target.max()) - floor, 1e-9))[0]
    end = len(target) if rising.size == 0 else last_pulse + bottom + int(rising[0])
    return end if end >= MINIMUM_NODES else len(target)


def _distribute(model, muscle_index, q, torque) -> np.ndarray:
    """
    The share of the joint torque the biceps carries at each node, by static optimisation.
    """
    ceiling = np.array([model.muscle(i).characteristics().forceIsoMax() for i in range(model.nbMuscles())])
    shares, start = [], np.zeros(len(ceiling))

    for i in range(len(q)):
        arms = model.musclesLengthJacobian(np.array([q[i]])).to_array()[:, 0]
        answer = optimize.minimize(
            lambda forces: float(np.sum((forces / ceiling) ** 2)),
            start,
            jac=lambda forces: 2 * forces / ceiling ** 2,
            constraints=[{"type": "eq",
                          "fun": lambda forces, a=arms, t=torque[i]: float(-a @ forces - t),
                          "jac": lambda forces, a=arms: -a}],
            bounds=[(0.0, high) for high in ceiling],
            method="SLSQP",
            options={"ftol": 1e-12, "maxiter": 200},
        )
        residual = abs(-arms @ answer.x - torque[i])
        if residual > 1e-6 * max(abs(torque[i]), 1.0):
            raise ValueError(f"static optimisation cannot deliver {torque[i]:.4f} Nm at "
                             f"{np.rad2deg(q[i]):.1f} deg: {residual:.2e} Nm short of it")
        # the next node starts where this one landed, which is what keeps 1800 solves per subject cheap
        start = answer.x
        shares.append(float(answer.x[muscle_index]))

    return np.array(shares)


def _static_torque(model, angle) -> float:
    """The joint torque holding the arm at one angle, motionless: the load the stimulation adds to."""
    return float(model.InverseDynamics(np.array([angle]), np.zeros(1), np.zeros(1)).to_array()[0])


def _inverse_dynamics(model, muscle_index, q, qdot, qddot) -> tuple:
    """The elbow torque of the measured movement and the biceps moment arm, node by node."""
    torque, moment_arm = [], []
    for i in range(len(q)):
        state = np.array([q[i]])
        torque.append(float(model.InverseDynamics(state, np.array([qdot[i]]), np.array([qddot[i]])).to_array()[0]))
        moment_arm.append(float(model.musclesLengthJacobian(state).to_array()[muscle_index][0]))
    return np.array(torque), np.array(moment_arm)


def _lay_end_to_end(phases) -> list:
    """The start time of every phase once they are laid out on one continuous time base."""
    offset = 0.0
    for phase in phases:
        phase["time_offset"] = offset
        offset += phase["final_time"]
    return phases


def conditions_of(phases) -> tuple:
    """The two stimulation conditions of every phase, as protocol.py takes them."""
    return [phase["frequency"] for phase in phases], [phase["pulse_width"] for phase in phases]


def labels_of(phases) -> list:
    """How every phase is named in the tables and the figures, ex: "50Hz 400us"."""
    return [protocol.condition_name(phase["frequency"], phase["pulse_width"]) for phase in phases]
