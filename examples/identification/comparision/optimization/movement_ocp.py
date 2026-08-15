"""
Identification of the Ding et al. (2007) pulse width frequency FES model from the measured elbow flexion.
"""

import pickle
import re
import sys
from pathlib import Path

import numpy as np

from bioptim import (
    BoundsList,
    ControlType,
    CostType,
    DynamicsOptions,
    DynamicsOptionsList,
    InitialGuessList,
    VariableScalingList,
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

from cocofest import DingModelPulseWidthFrequency, OcpFesId

COMPARISON_ROOT = Path(__file__).resolve().parent.parent
for _root in (COMPARISON_ROOT, Path(__file__).resolve().parent):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
from processing.helper.c3d_to_q import C3dToQ
from passive_ocp import settled_end
from helper import debug_plots, results
from helper.passive_torque_model import PassiveTorque, PassiveTorqueFesMskModel
from helper.passive_torque_riener import RienerPassiveTorque

MUSCLE_NAME = "BIClong"
SUM_STIM_TRUNCATION = 10

ACTIVATE_FORCE_LENGTH_RELATIONSHIP = False
ACTIVATE_FORCE_VELOCITY_RELATIONSHIP = False
ACTIVATE_PASSIVE_FORCE_RELATIONSHIP = False
NON_WEIGHTED_FREQUENCIES = [20, 33, 50]
SEEDS_PULSE_WIDTH_PATH = COMPARISON_ROOT / "processing" / "helper" / "seeds_pulse_width.pkl"

NOMINAL = {
    "tau1_rest": 0.060601,
    "tau2": 0.001,
    "km_rest": 0.137,
    "a_scale": 4920.0,
    "pd0": 1.31405e-4,
    "pdt": 1.94138e-4,
}

FACTOR_BOUNDS = (0.2, 5.0)
FACTOR_OVERRIDES = {"a_scale": (0.03, 2.0)}

IDENTIFIED_PARAMETERS = {
    name: (value, FACTOR_OVERRIDES.get(name, FACTOR_BOUNDS)[0] * value, FACTOR_OVERRIDES.get(name, FACTOR_BOUNDS)[1] * value, value)
    for name, value in NOMINAL.items()
}

FIXED_AT_LITERATURE = {name: NOMINAL[name] for name in ("pd0", "pdt", "tau2")}

PASSIVE_CLASSES = {"double_exponential": PassiveTorque, "riener": RienerPassiveTorque}
# The flexion side exponential exp(a3 + a4 * theta), in the same parameters passive_ocp identifies
PASSIVE_FLEXION_STOP = {
    "riener": ["a3", "a4"],
    "double_exponential": ["k3", "k4", "theta_max"],
}
PASSIVE_CLIP_MARGIN = 0.15
TRUNCATED_TRAIN_FRACTION = 0.5
PD0_MARGIN = 0.95

# Initial guess of (Cn, F) after the first node. (0, 0) is rest, which leaves four muscle parameters
# with an exactly zero Jacobian column at iteration 0, see prepare_ocp.
MUSCLE_STATE_GUESS = (0.3, 25.0)
# Ceiling of the F state (N). The cocofest default of 248 N sits below the 458 N measured on this cohort.
FMAX = 1000.0
PULSE_WIDTH_SCALING = 1e-4
# Threads of the ocp. Above 1 the reduction order depends on the machine, and on a problem with basins this
# close together the last bits of the gradient decide which one the solve ends in.
N_THREADS = 20


# ---- Reading the experimental flexions ---- #
def find_motion_file(subject, frequency):
    """The motion recording of a subject at a given frequency, without the added weight."""
    folder = COMPARISON_ROOT / "data" / "exp" / f"P{subject}"
    matching = [
        path for path in sorted(folder.glob(f"p{subject}_motion*{frequency}Hz*.c3d")) if "weight" not in path.stem
    ]
    if not matching:
        raise FileNotFoundError(f"No unweighted {frequency}Hz motion recording for P{subject} in {folder}.")
    return matching[0]


def pulse_widths_of(file):
    """
    The pulse width applied to each train of a recording, in seconds.
    """
    with open(SEEDS_PULSE_WIDTH_PATH, "rb") as handle:
        seeds = pickle.load(handle)
    seed = int(re.search(r"_(\d+)\.c3d$", file.name).group(1))
    return [float(value) / 1e6 for value in seeds[seed]]


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


def movement_phases(subject, frequency, debug=False):
    """
    The movements of one motion recording, rise and fall, with the pulse width that drove each of them.

    Returns
    -------
    list[dict]
        One entry per movement, with "q", "time", "stim_time", "pulse_width" and "frequency"
    """
    file = find_motion_file(subject, frequency)
    print(f"Loading file: {file}")

    converter = C3dToQ(str(file))
    converter.frequency_stimulation = frequency
    stim_time = converter.get_sliced_stim_time()["stim_time"]
    q_rad, time, phase_stim, kept = movement_slicing(converter.get_q_rad(), converter.get_time(), stim_time)
    pulse_widths = pulse_widths_of(file)

    panel = {
        "full_time": converter.get_time(),
        "full_data": converter.get_q_rad(),
        "phase_times": time,
        "phase_data": q_rad,
        "stim_times": stim_time,
        "label": f"{frequency} Hz",
    }

    return panel, [
        {
            "q": q_rad[i],
            "time": time[i],
            "stim_time": phase_stim[i],
            "pulse_width": pulse_widths[kept[i] % len(pulse_widths)],
            "frequency": frequency,
        }
        for i in range(len(q_rad))
    ]


def collect_movement_phases(subject, frequencies=None, debug=False):
    """Gather the movements of several recordings and lay them out on one continuous time base."""
    frequencies = NON_WEIGHTED_FREQUENCIES if frequencies is None else frequencies

    phases, panels = [], []
    for frequency in frequencies:
        try:
            panel, recording_phases = movement_phases(subject, frequency, debug=debug)
        except (FileNotFoundError, ValueError, OSError, KeyError) as error:
            print(f"  P{subject} {frequency}Hz: unusable ({error}), skipped.")
            continue
        phases += recording_phases
        panels.append(panel)

    # --- Debug, pre ocp: every recording of the subject on one figure, one row each --- #
    if debug and panels:
        debug_plots.plot_phase_extraction(
            panels=panels,
            title=f"P{subject} movement extraction (rise and fall)",
            unit="deg",
            scale=180 / np.pi,
            name="movement_extraction",
        )

    offset = 0.0
    for phase in phases:
        phase["final_time"] = float(phase["time"][-1] - phase["time"][0])
        phase["global_time"] = phase["time"] - phase["time"][0] + offset
        phase["time_offset"] = offset
        offset += phase["final_time"]
    return phases


# ---- Building the ocp ---- #
def get_numerical_data_time_series(model, n_shooting, final_time, stim_time, previous_model=None, time_offset=0.0):
    """
    The truncated previous stimulation times at each node of one phase, on the ocp global time base.
    """
    truncation = model.sum_stim_truncation

    if previous_model is None:
        model.previous_stim = model._get_additional_previous_stim_time()
    else:
        model.previous_stim["time"] = list(previous_model.all_stim[-truncation:])

    model.all_stim = model.previous_stim["time"][-truncation:] + [time + time_offset for time in stim_time]
    all_stim = np.array(model.all_stim)
    dt = final_time / n_shooting

    tolerance = 1e-6 * dt
    node_idx = [np.where(all_stim <= time_offset + i * dt + tolerance)[0][-1] for i in range(n_shooting + 1)]
    stim_time_per_node = np.array([all_stim[: idx + 1][-truncation:] for idx in node_idx])
    return {"stim_time": np.transpose(stim_time_per_node[:, np.newaxis, :], (2, 1, 0))}


def aligned_shooting_grid(final_time, stim_time, target_n_shooting):
    """
    A node grid whose interval is a whole fraction of the stimulation period.

    Parameters
    ----------
    final_time: float
        The measured duration of the phase (s)
    stim_time: list | np.ndarray
        The stimulation times of the phase, from its start (s)
    target_n_shooting: int
        The number of nodes asked for, matched as closely as the period allows

    Returns
    -------
    tuple
        The number of shooting intervals, and the phase duration rounded to a whole number of them
    """
    stim_time = np.asarray(stim_time, dtype=float)
    if stim_time.size < 2 or final_time <= 0:
        return target_n_shooting, final_time

    period = float(np.median(np.diff(stim_time)))
    subdivisions = max(1, int(round(period * target_n_shooting / final_time)))
    dt = period / subdivisions
    n_shooting = max(1, int(round(final_time / dt)))
    return n_shooting, n_shooting * dt


def build_model(model_path, stim_time, passive_torque, parameters=None):
    """The FES driven musculoskeletal model of the elbow, carrying the identified passive articular torque."""
    muscle = DingModelPulseWidthFrequency(muscle_name=MUSCLE_NAME, sum_stim_truncation=SUM_STIM_TRUNCATION)
    muscle.fmax = FMAX
    return PassiveTorqueFesMskModel(
        name=None,
        biorbd_path=model_path,
        muscles_model=[muscle],
        stim_time=list(stim_time),
        activate_force_length_relationship=ACTIVATE_FORCE_LENGTH_RELATIONSHIP,
        activate_force_velocity_relationship=ACTIVATE_FORCE_VELOCITY_RELATIONSHIP,
        activate_passive_force_relationship=ACTIVATE_PASSIVE_FORCE_RELATIONSHIP,
        activate_residual_torque=False,
        parameters=parameters,
        external_force_set=None,
        passive_torque=passive_torque,
    )


def _muscle_setter(name):
    """
    A setter bioptim can call on any phase's model.
    """

    def setter(model, value, **kwargs):
        muscle_models = getattr(model, "muscles_dynamics_model", None)
        if not muscle_models:
            return
        getattr(muscle_models[0], f"set_{name}")(muscle_models[0], value)

    return setter


def _passive_setter(name):
    """
    A setter for the passive torque parameters identified here.
    """

    def setter(model, value, **kwargs):
        passive = getattr(model, "passive_torque", None)
        if passive is None:
            return
        getattr(passive, f"set_{name}")(model, value)

    return setter


def set_default_values(
    passive_torque, formulation, max_elbow_position, min_pulse_width=None, include_passive_stop=True
):
    """
    The identification settings of the muscle parameters, plus the flexion stop of the passive torque.

    Parameters
    ----------
    passive_torque: PassiveTorque | RienerPassiveTorque
        The torque identified on the relaxations, whose flexion stop is re-identified here
    formulation: str
        Which passive formulation it is, to know which of its parameters make up the flexion stop
    max_elbow_position: float
        Used by the double exponential settings for the bounds of theta_max (rad)
    min_pulse_width: float
        The smallest pulse width the subject actually received (s), which caps pd0. See PD0_MARGIN.
    include_passive_stop: bool
        Whether the flexion stop is re-identified here, or kept as the relaxations found it
    """
    settings = {
        name: {
            "initial_guess": initial_guess,
            "min_bound": min_bound,
            "max_bound": max_bound,
            "function": _muscle_setter(name),
            "scaling": scaling,
        }
        for name, (initial_guess, min_bound, max_bound, scaling) in IDENTIFIED_PARAMETERS.items()
    }

    if min_pulse_width:
        pd0 = dict(settings["pd0"])
        pd0["max_bound"] = PD0_MARGIN * min_pulse_width
        pd0["initial_guess"] = min(pd0["initial_guess"], 0.5 * pd0["max_bound"])
        settings["pd0"] = pd0

    if not include_passive_stop:
        return settings

    passive_settings = passive_torque.default_parameter_settings(max_elbow_position=max_elbow_position)
    for name in PASSIVE_FLEXION_STOP[formulation]:
        setting = dict(passive_settings[name])
        # Start from what the relaxations found rather than from the generic guess
        found = getattr(passive_torque, name, None)
        if found is None:
            found = getattr(passive_torque, f"_{name}", None)
        if found is not None:
            setting["initial_guess"] = float(np.clip(found, setting["min_bound"], setting["max_bound"]))
        setting["function"] = _passive_setter(name)
        settings[name] = setting

    return settings


def prepare_ocp(
    model_path,
    phases,
    passive_torque,
    formulation="riener",
    n_shooting_per_phase=90,
    fixed_parameters=None,
    clip_margin=PASSIVE_CLIP_MARGIN,
    use_sx=False,
    identify_passive_stop=True,
    initial_states=None,
    initial_guesses=None,
    passive_theta_bounds=None,
):
    """
    Build the multiphase ocp identifying the FES model over every measured flexion at once.

    Parameters
    ----------
    model_path: str
        The path to the subject's scaled bioMod
    phases: list[dict]
        The movement phases, as built by collect_movement_phases
    passive_torque: PassiveTorque | RienerPassiveTorque
        The articular torque identified beforehand
    n_shooting_per_phase: int
        The number of shooting points of each phase, the measured angle is resampled on them
    clip_margin: float
        How far beyond the reached angles the passive torque stays live, see PASSIVE_CLIP_MARGIN
    identify_passive_stop: bool
        Whether the flexion stop of the passive torque is identified alongside the Ding model
    initial_states: list
        One (4, n_shooting + 1) array per phase, holding Cn, F, q and qdot to start from
    initial_guesses: dict
        Where each named parameter starts from, clipped to its bounds
    passive_theta_bounds: tuple
        The angle range the passive torque stays live over, given explicitly so two ocps can share it
    """
    # The passive torque is used over the angles this movement reaches, not over the ones the relaxations reached.
    reached = np.concatenate([phase["q"] for phase in phases])
    margin = clip_margin * float(reached.max() - reached.min())
    passive_torque.theta_bounds = (
        passive_theta_bounds
        if passive_theta_bounds is not None
        else (float(reached.min()) - margin, float(reached.max()) + margin)
    )

    anchor = getattr(passive_torque, "flexion_limit", None)
    settings = set_default_values(
        passive_torque,
        formulation,
        max_elbow_position=max(float(anchor or 0.0), float(reached.max())),
        min_pulse_width=min(phase["pulse_width"] for phase in phases),
        include_passive_stop=identify_passive_stop,
    )

    for name, value in (initial_guesses or {}).items():
        if name in settings:
            bounds = (settings[name]["min_bound"], settings[name]["max_bound"])
            settings[name] = {**settings[name], "initial_guess": float(np.clip(value, *bounds))}

    for name, value in (fixed_parameters or {}).items():
        if name in settings:
            # None pins the parameter where set_default_values put it
            value = settings[name]["initial_guess"] if value is None else value
            settings[name] = {**settings[name], "initial_guess": value, "min_bound": value, "max_bound": value}

    # --- The identified parameters are common to every phase, declared once for the whole ocp --- #
    parameters, parameters_bounds, parameters_init = OcpFesId.set_parameters(
        parameter_to_identify=list(settings),
        parameter_setting=settings,
        use_sx=use_sx,
    )

    models = [build_model(model_path, phase["stim_time"], passive_torque) for phase in phases]
    muscle_models = [model.muscles_dynamics_model[0] for model in models]

    dynamics = DynamicsOptionsList()
    x_bounds, x_init = BoundsList(), InitialGuessList()
    u_bounds, u_init = BoundsList(), InitialGuessList()
    u_scaling = VariableScalingList()
    objective_functions = ObjectiveList()
    targets = []

    # One grid per phase, aligned on its own stimulation period
    grid = [aligned_shooting_grid(phase["final_time"], phase["stim_time"], n_shooting_per_phase) for phase in phases]
    final_times = [duration for _, duration in grid]
    offsets = np.concatenate(([0.0], np.cumsum(final_times)[:-1]))

    for i, (model, phase) in enumerate(zip(models, phases)):
        n_shooting, final_time = grid[i]
        target = np.interp(
            np.linspace(0, final_time, n_shooting + 1),
            phase["time"] - phase["time"][0],
            phase["q"],
        )
        targets.append(target)

        dynamics.add(
            DynamicsOptions(
                expand_dynamics=True,
                phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
                ode_solver=OdeSolver.COLLOCATION(polynomial_degree=5, method="radau"),
                numerical_data_timeseries=get_numerical_data_time_series(
                    model=muscle_models[i],
                    n_shooting=n_shooting,
                    final_time=final_time,
                    stim_time=phase["stim_time"],
                    previous_model=muscle_models[i - 1] if i > 0 else None,
                    time_offset=float(offsets[i]),
                ),
                phase=i,
            )
        )

        # --- The muscle starts at rest, the elbow at the measured angle with no velocity --- #
        rest = muscle_models[i].standard_rest_values()
        ceiling = np.array([[10.0], [muscle_models[i].fmax]])
        for j, name in enumerate(muscle_models[i].name_dofs):
            x_bounds.add(
                name,
                min_bound=np.array([np.concatenate((rest, rest, rest), axis=1)[j]]),
                max_bound=np.array([np.concatenate((rest, ceiling, ceiling), axis=1)[j]]),
                phase=i,
                interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
            )
 
            if initial_states is None:
                guess = np.full((1, n_shooting + 1), MUSCLE_STATE_GUESS[j])
                guess[0, 0] = float(np.squeeze(rest[j]))
            else:
                guess = np.asarray(initial_states[i], dtype=float)[j][np.newaxis, :]
            x_init.add(name, guess, interpolation=InterpolationType.EACH_FRAME, phase=i)

        q_bounds = model.bounds_from_ranges("q")
        q_bounds.min[0][0] = q_bounds.max[0][0] = target[0]
        qdot_bounds = model.bounds_from_ranges("qdot")
        qdot_bounds.min[0][0] = qdot_bounds.max[0][0] = 0
        x_bounds.add(key="q", bounds=q_bounds, phase=i)
        x_bounds.add(key="qdot", bounds=qdot_bounds, phase=i)
        if initial_states is None:
            q_guess = target[np.newaxis, :]
            qdot_guess = np.gradient(target, final_time / n_shooting)[np.newaxis, :]
        else:
            simulated = np.asarray(initial_states[i], dtype=float)
            q_guess, qdot_guess = simulated[2][np.newaxis, :], simulated[3][np.newaxis, :]
        x_init.add(key="q", initial_guess=q_guess, interpolation=InterpolationType.EACH_FRAME, phase=i)
        x_init.add(key="qdot", initial_guess=qdot_guess, interpolation=InterpolationType.EACH_FRAME, phase=i)

        # --- The pulse width is the one that was applied, so only the parameters are identified --- #
        key = "last_pulse_width_" + MUSCLE_NAME
        control = np.array([[phase["pulse_width"]] * n_shooting])
        u_init.add(key=key, initial_guess=control, interpolation=InterpolationType.EACH_FRAME, phase=i)
        u_bounds.add(key, min_bound=control, max_bound=control, interpolation=InterpolationType.EACH_FRAME, phase=i)
        # No u_scaling: inert for the solve, but it makes nlp.dynamics_func expect the control in graph units

        objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_STATE,
            key="q",
            weight=1,
            target=target[np.newaxis, :-1],
            node=Node.ALL_SHOOTING,
            quadratic=True,
            index=[0],
            phase=i,
        )

        models[i] = update_model(model, parameters)
        update_model_param(muscle_models[i], parameters)

    # --- Each flexion is an independent trial, the state is not carried from one to the next --- #
    phase_transitions = PhaseTransitionList()
    for i in range(len(phases) - 1):
        phase_transitions.add(PhaseTransitionFcn.DISCONTINUOUS, phase_pre_idx=i)

    ocp = OptimalControlProgram(
        bio_model=models,
        dynamics=dynamics,
        n_shooting=[intervals for intervals, _ in grid],
        phase_time=final_times,
        x_init=x_init,
        x_bounds=x_bounds,
        u_init=u_init,
        u_bounds=u_bounds,
        u_scaling=u_scaling,
        objective_functions=objective_functions,
        parameters=parameters,
        parameter_bounds=parameters_bounds,
        parameter_init=parameters_init,
        phase_transitions=phase_transitions,
        control_type=ControlType.CONSTANT,
        use_sx=use_sx,
        n_threads=N_THREADS,
    )
    return ocp, targets, final_times


def update_model(model, parameters):
    """Rebuild the model with the ocp parameters attached, keeping its passive articular torque."""
    return PassiveTorqueFesMskModel(
        name=model.name,
        biorbd_path=model.biorbd_path,
        muscles_model=model.muscles_dynamics_model,
        stim_time=model.muscles_dynamics_model[0].stim_time,
        previous_stim=model.muscles_dynamics_model[0].previous_stim,
        activate_force_length_relationship=model.activate_force_length_relationship,
        activate_force_velocity_relationship=model.activate_force_velocity_relationship,
        activate_passive_force_relationship=model.activate_passive_force_relationship,
        activate_residual_torque=model.activate_residual_torque,
        parameters=parameters,
        external_force_set=None,
        passive_torque=model.passive_torque,
    )


def update_model_param(muscle_model, parameters):
    """
    Apply the muscle parameters of the ocp on one phase's muscle model.
    """
    for key in parameters:
        if key not in IDENTIFIED_PARAMETERS:
            continue
        getattr(muscle_model, f"set_{key}")(muscle_model, parameters[key].cx * parameters[key].scaling.scaling)


def load_passive_torque(subject, method):
    """The articular torque identified beforehand, its formulation, and where it came from."""
    path = results.path_of(subject, method)
    formulation = "riener" if "riener" in method else "double_exponential"
    if not path.exists():
        return None, formulation, path
    return PASSIVE_CLASSES[formulation].from_identification(results.load(subject, method)), formulation, path


def predict(subject, phases, parameters, passive_method, max_iter=300):
    """
    Run the identified model forward on movements it was NOT identified on, and score it against them.
    """
    passive_torque, formulation, _ = load_passive_torque(subject, passive_method)
    model_path = str(COMPARISON_ROOT / "model" / f"p{subject}_scaling_scaled.bioMod")
    ocp, targets, final_times = prepare_ocp(
        model_path, phases, passive_torque, formulation=formulation, fixed_parameters=parameters
    )
    sol = ocp.solve(Solver.IPOPT(_max_iter=max_iter))

    time, q_identified = results.solution_at_nodes(sol, "q", [len(target) - 1 for target in targets])
    q_tracked = np.concatenate(targets)
    return {
        "time": time,
        "identified": q_identified,
        "tracked": q_tracked,
        "rmse": results.rmse(q_tracked, q_identified),
        "converged": sol.status == 0,
    }


# ---- Running it ---- #
def identify(
    subject,
    phases,
    method,
    passive_method,
    plot=True,
    save=True,
    debug=True,
    show_debug=None,
    max_iter=1000,
    extra=None,
    fixed_parameters=None,
):
    """
    Identify the FES model of one subject on the flexion phases given to it.

    Parameters
    ----------
    subject: str
        The subject id, ex: "01"
    phases: list[dict]
        The movements to identify on, as built by collect_movement_phases
    method: str
        The name the result is stored under, ex: "movement_id_all"
    passive_method: str
        The stored passive torque identification to take the articular torque from
    show_debug: bool
        If the debug figures should also be opened on screen. Defaults to `plot`, see force_ocp.identify.
    fixed_parameters: dict
        Overrides FIXED_AT_LITERATURE, which is applied otherwise.
    """
    debug_plots.output_for(method, subject, debug=debug, show=plot if show_debug is None else show_debug)

    passive_torque, formulation, passive_path = load_passive_torque(subject, passive_method)
    if passive_torque is None:
        print(f"P{subject}: no identified passive torque ({passive_path.name}), skipping.")
        return

    fixed = {**FIXED_AT_LITERATURE, **(fixed_parameters or {})}
    model_path = str(COMPARISON_ROOT / "model" / f"p{subject}_scaling_scaled.bioMod")
    ocp, targets, final_times = prepare_ocp(
        model_path, phases, passive_torque, formulation=formulation, fixed_parameters=fixed
    )

    # --- Debug, pre ocp: the target the cost function actually holds, read back from the built ocp --- #
    if debug:
        debug_plots.plot_tracking_target(
            raw_time=[phase["global_time"] for phase in phases],
            raw_data=[phase["q"] for phase in phases],
            target_time=[
                offset + np.linspace(0, duration, len(target))[:-1]
                for offset, duration, target in zip(
                    np.concatenate(([0.0], np.cumsum(final_times)[:-1])), final_times, targets
                )
            ],
            target=debug_plots.cost_function_target(ocp, key="q"),
            title=f"P{subject} elbow angle target held by the cost function",
            unit="deg",
            scale=180 / np.pi,
            name="angle_target",
        )

    if plot:
        ocp.add_plot_penalty(CostType.ALL)
    sol = ocp.solve(Solver.IPOPT(_max_iter=max_iter))

    time, q_identified = results.solution_at_nodes(sol, "q", [len(target) - 1 for target in targets])
    q_tracked = np.concatenate(targets)
    parameters = {key: float(value.squeeze()) for key, value in sol.decision_parameters().items()}

    at_bound = [name for name in results.parameters_at_bound(ocp, parameters) if name not in fixed]

    print(f"P{subject} RMSE: {np.rad2deg(results.rmse(q_tracked, q_identified)):.3f} deg ({len(phases)} movement(s))")
    for key, value in parameters.items():
        print(f"  {key}: {value}{'  (at bound)' if key in at_bound else ''}")

    if plot or debug:
        offset, conditions = 0.0, []
        for phase, duration in zip(phases, final_times):
            conditions.append(
                {
                    "t_start": offset,
                    "t_end": offset + duration,
                    "pulse_width_us": phase["pulse_width"] * 1e6,
                    "frequency_hz": phase["frequency"],
                }
            )
            offset += duration

        debug_plots.plot_solution(
            time=time,
            tracked=q_tracked,
            identified=q_identified,
            title=f"P{subject} identification from the measured flexion",
            unit="deg",
            ylabel="Joint angle (deg)",
            scale=180 / np.pi,
            phase_lengths=[len(target) for target in targets],
            conditions=conditions,
            parameters=parameters,
            name="movement_identification",
        )

    if save:
        results.save(
            subject=subject,
            method=method,
            parameters=parameters,
            time=time,
            tracked=q_tracked,
            identified=q_identified,
            unit="rad",
            sol=sol,
            at_bound=at_bound,
            bounds=results.parameter_bounds(ocp, parameters),
            phase_lengths=[len(target) for target in targets],
            extra={
                **(extra if extra else {}),
                "fixed_parameters": fixed,
                "n_movements": len(phases),
                "frequency": [phase["frequency"] for phase in phases],
                "pulse_width": [phase["pulse_width"] for phase in phases],
                "passive_method": passive_method,
                "passive_formulation": formulation,
                "passive_flexion_stop": {
                    name: parameters[name] for name in PASSIVE_FLEXION_STOP[formulation] if name in parameters
                },
                # Provenance: what the identification actually started from, so a mismatched passive input
                # cannot go unnoticed again
                "passive_parameters": {
                    key: float(value)
                    for key, value in passive_torque.identifiable_parameters.items()
                    if isinstance(value, (int, float))
                },
                "passive_flexion_limit": getattr(passive_torque, "flexion_limit", None),
            },
        )

    return parameters
