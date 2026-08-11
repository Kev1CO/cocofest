"""
This program identifies the passive joint torque of the elbow from the experimental motion recordings.

Every stimulation train of a recording is followed by a phase where the elbow falls back freely, without any
stimulation: those relaxation phases are extracted, declared as the phases of a single multiphase ocp and tracked
with a torque driven model whose torque is bounded to zero, so that the only thing driving the motion is the
passive torque. The identified parameter set is saved and later applied to the FES driven model.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from bioptim import (
    BoundsList,
    ControlType,
    CostType,
    DynamicsOptions,
    DynamicsOptionsList,
    InitialGuessList,
    InterpolationType,
    Node,
    ObjectiveFcn,
    ObjectiveList,
    OdeSolver,
    OptimalControlProgram,
    PhaseDynamics,
    PhaseTransitionFcn,
    PhaseTransitionList,
    Solver,
)

from cocofest import OcpFesId

COMPARISON_ROOT = Path(__file__).resolve().parent.parent
for _root in (COMPARISON_ROOT, Path(__file__).resolve().parent):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
from processing.helper.c3d_to_q import C3dToQ
from helper import debug_plots, results
from helper.passive_torque_model import PassiveTorque, PassiveTorqueBiorbdModel
from helper.passive_torque_riener import RienerPassiveTorque

# The two passive torque formulations that can be identified :
# - "double_exponential" is the sigmoid gated form of helper/passive_torque_model.py.
# - "riener" is the Riener and Edrich form of helper/passive_torque_riener.py
FORMULATIONS = {
    "double_exponential": (PassiveTorque, ["k1", "k2", "k3", "k4", "theta_max", "theta_min"]),
    "riener": (RienerPassiveTorque, ["a1", "a2", "stop_torque", "stop_rate", "a5", "b"]),
}

MAX_PHASE_DURATION = 3.5
MIN_PHASE_DURATION = 0.3
SETTLED_SPEED_FRACTION = 0.05
SETTLED_QUIET_DURATION = 0.15
INITIAL_VELOCITY_TOLERANCE = 0  # rad/s


def find_motion_file(subject_folder, subject, frequency, weight):
    """
    Find the motion recording of a subject at a given stimulation frequency, with or without the added weight.
    """
    candidates = sorted(subject_folder.glob(f"p{subject}_motion*{frequency}Hz*.c3d"))
    matching = [path for path in candidates if ("weight" in path.stem) == weight]
    if not matching:
        raise FileNotFoundError(
            f"No {'weighted' if weight else 'unweighted'} {frequency}Hz motion recording for subject {subject} "
            f"in {subject_folder}."
        )
    return matching[0]


def slicing(q, time, stim_time):
    """
    Extract the passive relaxation phase of every stimulation train, i.e. the part between the instant the elbow
    stops rising (zero velocity, right after the last pulse of the train) and the first pulse of the next train.

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
        The elbow angle and the time vector of every relaxation phase
    """
    sliced_q, sliced_time = [], []
    for i in range(len(stim_time)):
        if len(stim_time[i]) == 0:
            continue
        next_stim = stim_time[i + 1][0] if i + 1 < len(stim_time) and len(stim_time[i + 1]) else time[-1]
        window = np.where((time >= stim_time[i][-1] - 0.02) & (time <= stim_time[i][-1] + 0.2))[0]
        if window.size < 3:
            continue

        # Velocity from central finite differences, to find where the elbow stops rising
        velocity = (q[window][2:] - q[window][:-2]) / 2
        time_velocity_equal_0 = time[window[np.argmin(np.abs(velocity)) + 1]]

        relaxation = np.where((time >= time_velocity_equal_0) & (time <= next_stim))[0]
        if relaxation.size < 3:
            continue

        relaxation_q = q[relaxation]
        half_way = relaxation_q[0] - 0.5 * (relaxation_q[0] - relaxation_q.min())
        descended = np.where(relaxation_q <= half_way)[0]

        end = settled_end(
            relaxation_q, time[relaxation], search_from=int(descended[0]) if descended.size else 0
        )
        relaxation = relaxation[:end]
        if relaxation.size < 3:
            continue

        sliced_q.append(q[relaxation])
        sliced_time.append(time[relaxation])

    return sliced_q, sliced_time


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


# The motion recordings used for the passive identification: three stimulation frequencies, no weight
ALL_CONDITIONS = [(20, False), (33, False), (50, False)]


def relaxation_phases(
    subject, frequency, weight, debug=False, max_duration=MAX_PHASE_DURATION, min_duration=MIN_PHASE_DURATION
):
    """
    The relaxation phases of ONE motion recording.

    Parameters
    ----------
    subject: str
        The subject id, ex: "01"
    frequency: int
        The stimulation frequency of the recording (Hz)
    weight: bool
        If the recording with the added weight should be read
    debug: bool
        If the extraction should be plotted
    max_duration: float
        Phases longer than this are dropped, see MAX_PHASE_DURATION
    min_duration: float
        Phases shorter than this are dropped, see MIN_PHASE_DURATION

    Returns
    -------
    tuple
        The elbow angle and the time vector of every kept phase
    """
    file = find_motion_file(COMPARISON_ROOT / "data" / "exp" / f"P{subject}", subject, frequency, weight)
    print(f"Loading file: {file}")

    converter = C3dToQ(str(file))
    converter.frequency_stimulation = frequency
    stim_time = converter.get_sliced_stim_time()["stim_time"]
    q_rad, time = slicing(converter.get_q_rad(), converter.get_time(), stim_time)

    kept = []
    for i in range(len(time)):
        duration = time[i][-1] - time[i][0]
        if duration > max_duration or duration < min_duration:
            print(f"  P{subject} {frequency}Hz relaxation {i} dropped: lasts {duration:.3f}s")
            continue
        kept.append(i)

    panel = {
        "full_time": converter.get_time(),
        "full_data": converter.get_q_rad(),
        "phase_times": [time[i] for i in kept],
        "phase_data": [q_rad[i] for i in kept],
        "stim_times": stim_time,
        "label": f"{frequency} Hz{' with weight' if weight else ''}",
    }
    return [q_rad[i] for i in kept], [time[i] for i in kept], panel


def collect_relaxation_phases(subject, conditions, debug=False):
    """
    Gather the relaxation phases of several recordings and lay them out on one continuous time base.

    Parameters
    ----------
    subject: str
        The subject id, ex: "01"
    conditions: list[tuple[int, bool]]
        The (frequency, weight) recordings to read. One entry identifies on a single recording, ALL_CONDITIONS
        identifies on everything the subject did.
    debug: bool
        If the extraction of each recording should be plotted

    Returns
    -------
    tuple
        The elbow angle, the duration and the global time vector of every phase
    """
    phases_q, phases_time, panels = [], [], []
    for frequency, weight in conditions:
        try:
            q_rad, time, panel = relaxation_phases(subject, frequency, weight, debug=debug)
        except (FileNotFoundError, ValueError, OSError) as error:
            print(f"  P{subject} {frequency}Hz{' weighted' if weight else ''}: unusable ({error}), skipped.")
            continue
        phases_q += q_rad
        phases_time += time
        panels.append(panel)

    # --- Debug, pre ocp: every recording of the subject --- #
    if debug and panels:
        debug_plots.plot_phase_extraction(
            panels=panels,
            title=f"P{subject} relaxation phase extraction",
            unit="deg",
            scale=180 / np.pi,
            name="relaxation_extraction",
        )

    # --- Make the phases follow one another on a single time base --- #
    global_time, global_final_time, global_q = [], [], []
    for i in range(len(phases_time)):
        previous_ending_time = 0 if len(global_time) == 0 else global_time[-1][-1]
        phase_time = phases_time[i] - phases_time[i][0]
        global_time.append(phase_time + previous_ending_time)
        global_final_time.append(phase_time[-1])
        global_q.append(phases_q[i])

    return global_q, global_final_time, global_time


def prepare_ocp(
    model_path,
    final_time,
    q_target,
    max_elbow_position,
    formulation="double_exponential",
    fixed_parameters=None,
    initial_guess=None,
    initial_velocity_tolerance=INITIAL_VELOCITY_TOLERANCE,
    use_sx=False,
):
    """
    Build the multiphase ocp identifying the passive torque parameters over every relaxation phase at once.

    Parameters
    ----------
    model_path: str
        The path to the subject's scaled bioMod
    final_time: list
        The duration of every relaxation phase (s)
    q_target: list
        The measured elbow angle of every relaxation phase (rad)
    max_elbow_position: float
        The measured maximal elbow flexion of the subject (rad)
    formulation: str
        Which passive torque formulation to identify, a key of FORMULATIONS
    initial_velocity_tolerance: float
        How far the first velocity of a phase may sit from the one measured there (rad/s). 0 pins it to the
        measurement, see INITIAL_VELOCITY_TOLERANCE.
    use_sx: bool
        If the ocp should use SX instead of MX variables
    """
    n_shooting_list = [len(target) - 1 for target in q_target]
    n_phases = len(n_shooting_list)

    # --- Declare the passive torque parameters to identify --- #
    passive_torque_class, key_parameter_to_identify = FORMULATIONS[formulation]
    reached = float(np.max(np.concatenate([np.asarray(target) for target in q_target])))
    flexion_limit = max(float(max_elbow_position), reached)
    passive_torque = passive_torque_class(flexion_limit=flexion_limit)
    settings = passive_torque.default_parameter_settings(max_elbow_position=flexion_limit)

    # Only moves where the search starts from, contrary to fixed_parameters which also pins the bounds
    for name, value in (initial_guess or {}).items():
        if name in settings:
            settings[name] = {**settings[name], "initial_guess": value}

    for name, value in (fixed_parameters or {}).items():
        if name in settings:
            settings[name] = {**settings[name], "initial_guess": value, "min_bound": value, "max_bound": value}

    parameters, parameters_bounds, parameters_init = OcpFesId.set_parameters(
        parameter_to_identify=key_parameter_to_identify,
        parameter_setting=settings,
        use_sx=use_sx,
    )

    # --- One model per phase, all sharing the same passive torque parameters --- #
    models = [
        PassiveTorqueBiorbdModel(model_path, passive_torque=passive_torque, parameters=parameters)
        for _ in range(n_phases)
    ]

    dynamics = DynamicsOptionsList()
    x_bounds, x_init = BoundsList(), InitialGuessList()
    u_bounds, u_init = BoundsList(), InitialGuessList()
    objective_functions = ObjectiveList()

    for i in range(n_phases):
        dynamics.add(
            DynamicsOptions(
                expand_dynamics=True,
                phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
                ode_solver=OdeSolver.COLLOCATION(polynomial_degree=5, method="radau"),
                phase=i,
            )
        )

        # The phase starts at the measured angle and the measured velocity, see INITIAL_VELOCITY_TOLERANCE
        measured_qdot = np.gradient(q_target[i], final_time[i] / n_shooting_list[i])

        q_x_bounds = models[i].bounds_from_ranges("q")
        q_x_bounds.min[0][0] = q_x_bounds.max[0][0] = q_target[i][0]
        qdot_x_bounds = models[i].bounds_from_ranges("qdot")
        qdot_x_bounds.min[0][0] = measured_qdot[0] - initial_velocity_tolerance
        qdot_x_bounds.max[0][0] = measured_qdot[0] + initial_velocity_tolerance
        x_bounds.add(key="q", bounds=q_x_bounds, phase=i)
        x_bounds.add(key="qdot", bounds=qdot_x_bounds, phase=i)

        x_init.add(key="q", initial_guess=q_target[i][np.newaxis, :], interpolation=InterpolationType.EACH_FRAME, phase=i)
        x_init.add(
            key="qdot",
            initial_guess=measured_qdot[np.newaxis, :],
            interpolation=InterpolationType.EACH_FRAME,
            phase=i,
        )

        # No muscle is stimulated during a relaxation phase, the joint torque is therefore bounded to zero
        u_bounds.add(key="tau", min_bound=[0] * models[i].nb_tau, max_bound=[0] * models[i].nb_tau, phase=i)
        u_init.add(key="tau", initial_guess=[0] * models[i].nb_tau, phase=i)

        objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_STATE,
            key="q",
            weight=1000,
            target=q_target[i][np.newaxis, :-1],
            node=Node.ALL_SHOOTING,
            quadratic=True,
            index=[0],
            phase=i,
        )

    # --- Each phase is an independent relaxation, the state is not carried from one to the next --- #
    phase_transitions = PhaseTransitionList()
    for i in range(n_phases - 1):
        phase_transitions.add(PhaseTransitionFcn.DISCONTINUOUS, phase_pre_idx=i)

    return OptimalControlProgram(
        bio_model=models,
        dynamics=dynamics,
        n_shooting=n_shooting_list,
        phase_time=final_time,
        x_init=x_init,
        x_bounds=x_bounds,
        u_init=u_init,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        parameters=parameters,
        parameter_bounds=parameters_bounds,
        parameter_init=parameters_init,
        phase_transitions=phase_transitions,
        control_type=ControlType.CONSTANT,
        use_sx=use_sx,
        n_threads=20,
    )


def predict(subject, global_q, global_final_time, global_time, parameters, formulation="riener", max_iter=300):
    """
    Run the identified passive torque forward on relaxations it was not identified on, and score it.
    """
    participants = pd.read_excel(COMPARISON_ROOT / "data" / "exp" / "data_participants.xlsx")
    elbow_joint_limit = participants.loc[participants["participant"] == int(subject), "elbow_joint_limit"].values[0]

    ocp = prepare_ocp(
        model_path=str(COMPARISON_ROOT / "model" / f"p{subject}_scaling_scaled.bioMod"),
        final_time=global_final_time,
        q_target=global_q,
        max_elbow_position=np.deg2rad(180 - elbow_joint_limit),
        formulation=formulation,
        fixed_parameters=parameters,
    )
    sol = ocp.solve(Solver.IPOPT(_max_iter=max_iter))

    # Read at the shooting nodes, where the tracked angle is defined
    time, q_identified = results.solution_at_nodes(sol, "q", [len(phase) - 1 for phase in global_q])
    q_tracked = np.concatenate(global_q)
    return {
        "time": time,
        "identified": q_identified,
        "tracked": q_tracked,
        "rmse": results.rmse(q_tracked, q_identified),
        "converged": sol.status == 0,
    }


def identify(
    subject,
    global_q,
    global_final_time,
    global_time,
    method,
    formulation="riener",
    initial_guess=None,
    plot=True,
    save=True,
    debug=True,
    show_debug=None,
    max_iter=1000,
    extra=None,
):
    """
    Identify the passive joint torque of one subject on the relaxation phases given to it.

    Parameters
    ----------
    subject: str
        The subject id, ex: "01"
    global_q, global_final_time, global_time
        The relaxation phases, as returned by collect_relaxation_phases
    method: str
        The name the result is stored under, ex: "passive_torque_id_single_riener"
    formulation: str
        Which passive torque formulation to identify: "double_exponential" (the sigmoid gated form) or "riener"
        (Riener and Edrich). See FORMULATIONS.
    show_debug: bool
        If the debug figures should also be opened on screen. Defaults to `plot`, see force_ocp.identify.
    """
    if formulation not in FORMULATIONS:
        raise ValueError(f"Unknown formulation '{formulation}', pick one of {sorted(FORMULATIONS)}.")

    debug_plots.output_for(method, subject, debug=debug, show=plot if show_debug is None else show_debug)
    passive_torque_class, _ = FORMULATIONS[formulation]
    participants = pd.read_excel(COMPARISON_ROOT / "data" / "exp" / "data_participants.xlsx")

    elbow_joint_limit = participants.loc[participants["participant"] == int(subject), "elbow_joint_limit"].values[0]
    flexion_limit = max(float(np.deg2rad(180 - elbow_joint_limit)), float(np.max(np.concatenate(global_q))))
    ocp = prepare_ocp(
        model_path=str(COMPARISON_ROOT / "model" / f"p{subject}_scaling_scaled.bioMod"),
        final_time=global_final_time,
        q_target=global_q,
        max_elbow_position=np.deg2rad(180 - elbow_joint_limit),
        formulation=formulation,
        initial_guess=initial_guess,
    )

    # --- Debug, pre ocp: the target the cost function actually holds --- #
    if debug:
        debug_plots.plot_tracking_target(
            raw_time=global_time,
            raw_data=global_q,
            target_time=[time[:-1] for time in global_time],
            target=debug_plots.cost_function_target(ocp, key="q"),
            title=f"P{subject} elbow angle target held by the cost function",
            unit="deg",
            scale=180 / np.pi,
            name="angle_target",
        )

    if plot:
        ocp.add_plot_penalty(CostType.ALL)
    sol = ocp.solve(Solver.IPOPT(_max_iter=max_iter))

    # Read at the shooting nodes, where the tracked angle is defined
    time, q_identified = results.solution_at_nodes(sol, "q", [len(phase) - 1 for phase in global_q])
    q_tracked = np.concatenate(global_q)
    parameters = {key: float(value.squeeze()) for key, value in sol.decision_parameters().items()}
    at_bound = results.parameters_at_bound(ocp, parameters)

    resolved = (
        passive_torque_class(flexion_limit=flexion_limit).resolved_parameters(parameters)
        if hasattr(passive_torque_class, "resolved_parameters")
        else {**passive_torque_class.DEFAULTS, **parameters}
    )

    print(
        f"P{subject} RMSE: {np.rad2deg(results.rmse(q_tracked, q_identified)):.3f} deg "
        f"({len(global_q)} relaxation phase(s), {formulation})"
    )
    for key, value in parameters.items():
        print(f"  {key}: {value}{'  (at bound)' if key in at_bound else ''}")

    # --- Debug, solution: the identified motion against the tracked one, and the torque it implies --- #
    if plot or debug:
        debug_plots.plot_solution(
            time=time,
            tracked=q_tracked,
            identified=q_identified,
            title=f"P{subject} passive torque identification",
            unit="deg",
            ylabel="Joint angle (deg)",
            scale=180 / np.pi,
            phase_lengths=[len(phase) for phase in global_q],
            parameters=parameters,
            name="passive_identification",
        )
        # Bounded to the angles the relaxation phases
        debug_plots.plot_passive_torque(
            passive_torque_class(
                theta_bounds=(float(np.min(q_tracked)), float(np.max(q_tracked))),
                flexion_limit=flexion_limit,
                **resolved,
            ),
            subject=subject,
            parameters=parameters,
        )

    if save:
        results.save(
            subject=subject,
            method=method,
            parameters=resolved,
            time=time,
            tracked=q_tracked,
            identified=q_identified,
            unit="rad",
            sol=sol,
            at_bound=at_bound,
            bounds=results.parameter_bounds(ocp, resolved),
            phase_lengths=[len(phase) for phase in global_q],
            extra={
                **(extra if extra else {}),
                "formulation": formulation,
                "n_phases": len(global_q),
                "elbow_joint_limit": float(elbow_joint_limit),
                "flexion_limit": flexion_limit,
                "identified_stop": {k: v for k, v in parameters.items() if k.startswith("stop_")},
                "identified_range": (float(np.min(q_tracked)), float(np.max(q_tracked))),
            },
        )

    return resolved
