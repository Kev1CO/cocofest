"""
The same four steps, run on the measured elbow flexion instead of on the isometric force.

Only the observable changes. Each step builds its own ocp here, over the flexions of that step, declaring
only the parameters that step identifies and holding everything else on the model itself, exactly the way
isometric.py does it. optimization/movement_ocp.py still supplies the musculoskeletal model, the shooting
grid and the passive torque; what is identified is decided here, so the two routes identify the same
quantity the same way.

Three things had to give, all three recorded in the stored result:

1.  Step 1 reads tau1 off the flexion relaxation with relaxation.py, the same file and the same window
    the isometric route uses on the force
2.  The peak and the integral are taken above the starting angle of each movement, which is the excursion
    the stimulation produced.
3.  The passive articular torque is held at what the relaxations found. movement_ocp normally identifies
    its flexion stop alongside the muscle, and letting it move between steps would let it absorb what A is
    supposed to carry. identify_passive_stop=True restores the joint behaviour.

So step 2 identifies the article's three, A, Km and tau2, with tau1 measured on the relaxation beforehand,
exactly the set the isometric route identifies. The quantity the two routes can be compared on is the
plateau A * tau1, stored as "plateau_at_reference".

Five ocps per subject, so this is run one subject at a time. passive_torque_id_all_riener must have been
run first, the articular torque is read from it.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from bioptim import (
    BoundsList,
    ControlType,
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
for _root in (Path(__file__).resolve().parent, COMPARISON_ROOT, COMPARISON_ROOT / "optimization"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

import analysis  # noqa: E402
import model  # noqa: E402
import plots  # noqa: E402
import protocol  # noqa: E402
import relaxation  # noqa: E402
import movement_ocp  # noqa: E402
from helper import debug_plots, results  # noqa: E402

METHOD = "ding_movement_id"
PASSIVE_METHOD = "passive_torque_id_all_riener"

# Fixed in the used FES model
R0 = model.R0  # Value from Kesar et al. (2008)
TAUC = model.TAUC  # Value in (s) from Kesar et al. (2008)


# The settings of every parameter this route can identify, the ones isometric.py gives them, so a parameter
# is given the same freedom whichever observable it is identified from. (initial guess, min, max, scaling)
IDENTIFIED_PARAMETERS = {
    "A": (900, 1, 20000, 1000.0),
    "km_rest": (0.18, 1e-3, 5, 0.1),
    "tau2": (0.0185, 1e-4, 0.5, 0.05),
    # eq. 3, for the joint step 3 + 4. pd0 is capped per subject by the smallest pulse width applied.
    "a_scale": (2000, 1, 60000, 1000.0),
    "pd0": (1.31405e-4, model.PD0_FLOOR, 2.4e-4, 1e-4),
    "pdt": (1.94138e-4, 1e-5, 2e-3, 1e-4),
}

# What each step identifies. A name absent from these is not an ocp parameter at all: it is carried on the
# model, at what an earlier step settled it to, see `solve`. tau1 is settled before step 2 runs, as in the
# isometric route, see the step 1 block of `identify`.
STEP2_FREE = ("A", "km_rest")
STEP3_FREE = ("A",)

# Step 3 and 4 in one solve: a', pd0 and pdt straight off the identification trains, so the
# intensity information is used coherently instead of being squeezed through six independent A.
STEP34_FREE = ("a_scale", "pd0", "pdt")

# Shooting nodes per flexion, the number movement_ocp builds its own grid on
N_SHOOTING_PER_PHASE = 90

# What tells step 1's figure apart from the isometric one: same procedure and same file, different
# signal and a different home. See relaxation.mean_tau1.
TAU1_FIGURE = {
    'folder': plots.MOVEMENT_STEP1_ROOT,
    'ylabel': 'Elbow excursion (deg)',
    'unit': 'deg',
    'route': 'movement step 1',
}

@contextmanager
def article_model(model_class, constants=None):
    """
    Point movement_ocp at one of the article's models for the duration of a solve, then put it back.

    movement_ocp.build_model instantiates the muscle model itself, so this is the only way to give it a
    child class without editing it. Scoped and restored, so nothing else in the process ever sees it. The
    constants ride along on the constructor: they are what an earlier step settled and this one holds.
    tauc and R0 come with the model class itself, see model.py.
    """
    original_model = movement_ocp.DingModelPulseWidthFrequency
    movement_ocp.DingModelPulseWidthFrequency = partial(model_class, **(constants or {}))
    try:
        yield
    finally:
        movement_ocp.DingModelPulseWidthFrequency = original_model


# ---- Reading the recorded movements ---- #
def load_movements(subject, frequencies=None, debug=False, trim=True) -> list:
    """Every usable flexion of one subject, as movement_ocp extracts them, cut at the return."""
    phases = movement_ocp.collect_movement_phases(subject, frequencies=frequencies, debug=debug)
    for phase in phases:
        phase["subject"] = subject
        phase["pulse_width_us"] = protocol.as_microseconds(phase["pulse_width"])
    return trim_to_return(phases) if trim else phases


def trim_to_return(phases) -> list:
    """
    Cut each flexion where the elbow reaches its lowest angle again, not where the next train starts.

    movement_ocp slices a movement from its own first pulse to the next train's, so the window carries the
    tail of the return, where the limb has settled and the muscle is producing nothing. Every node of that
    tail is weight in the tracking cost with no information in it, and it is exactly the stretch a
    too-slow relaxation can hide in. The trim keeps the rise and the fall, which is what step 2 reads.

    The phases are laid end to end on one time base, so every offset after a trimmed phase moves with it.
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

    offset = 0.0
    for phase in phases:
        phase["final_time"] = float(phase["time"][-1] - phase["time"][0])
        phase["global_time"] = phase["time"] - phase["time"][0] + offset
        phase["time_offset"] = offset
        offset += phase["final_time"]
    return phases


def conditions_of(phases) -> tuple:
    """The two stimulation conditions of every movement, as protocol.py takes them."""
    return [phase["frequency"] for phase in phases], [phase["pulse_width"] for phase in phases]


def theta_bounds_of(phases, margin=movement_ocp.PASSIVE_CLIP_MARGIN) -> tuple:
    """
    The angle range the passive torque stays live over, taken once over every movement of the subject.

    Each step runs on a different subset. Left to itself, prepare_ocp would clip the passive torque to the
    angles of whichever subset it was given, so the same joint would be modelled slightly differently at
    each step and A would absorb the difference.
    """
    reached = np.concatenate([phase["q"] for phase in phases])
    span = margin * float(reached.max() - reached.min())
    return float(reached.min()) - span, float(reached.max()) + span


# ---- Building one step's ocp ---- #
def set_default_values(free, passive_torque, formulation, max_elbow_position, include_passive_stop=False):
    """
    The identification settings of the parameters this step leaves free, plus the passive flexion stop.

    What the step does not identify is absent from here rather than pinned with equal bounds: it rides on
    the model, at the value an earlier step settled it to. That is the isometric route's writing, and it
    keeps a settled parameter out of the KKT system instead of in it on an always-active bound.

    Parameters
    ----------
    free: tuple
        The names of the parameters this step identifies
    passive_torque: PassiveTorque | RienerPassiveTorque
        The torque identified on the relaxations, whose flexion stop may be re-identified here
    formulation: str
        Which passive formulation it is, to know which of its parameters make up the flexion stop
    max_elbow_position: float
        Used by the double exponential settings for the bounds of theta_max (rad)
    include_passive_stop: bool
        Whether the flexion stop is re-identified here, or kept as the relaxations found it
    """
    settings = {
        name: {
            "initial_guess": IDENTIFIED_PARAMETERS[name][0],
            "min_bound": IDENTIFIED_PARAMETERS[name][1],
            "max_bound": IDENTIFIED_PARAMETERS[name][2],
            # movement_ocp's setter: which model the value lands on is its wiring, not this step's business
            "function": movement_ocp._muscle_setter(name),
            "scaling": IDENTIFIED_PARAMETERS[name][3],
        }
        for name in free
    }

    if not include_passive_stop:
        return settings

    passive_settings = passive_torque.default_parameter_settings(max_elbow_position=max_elbow_position)
    for name in movement_ocp.PASSIVE_FLEXION_STOP[formulation]:
        setting = dict(passive_settings[name])
        # Start from what the relaxations found rather than from the generic guess
        found = getattr(passive_torque, name, None)
        if found is None:
            found = getattr(passive_torque, f"_{name}", None)
        if found is not None:
            setting["initial_guess"] = float(np.clip(found, setting["min_bound"], setting["max_bound"]))
        setting["function"] = movement_ocp._passive_setter(name)
        settings[name] = setting

    return settings


def update_model_param(muscle_model, parameters, names):
    """Apply the muscle parameters this step identifies on one phase's muscle model."""
    for name in names:
        getattr(muscle_model, f"set_{name}")(muscle_model, parameters[name].cx * parameters[name].scaling.scaling)


def prepare_ocp(model_path, phases, passive_torque, free, formulation="riener",
                n_shooting_per_phase=N_SHOOTING_PER_PHASE, clip_margin=movement_ocp.PASSIVE_CLIP_MARGIN,
                use_sx=False, identify_passive_stop=False, initial_states=None, initial_guesses=None,
                passive_theta_bounds=None):
    """
    Build the multiphase ocp of one step, one phase per measured flexion.

    The counterpart of isometric.prepare_ocp on the flexion: same staging, same `free` argument, the elbow
    angle tracked in place of the force. movement_ocp.prepare_ocp declares every parameter it knows of and
    pins whatever it is told to; this one declares only what the step identifies, which is what lets a step
    hold a parameter on the model rather than on a bound.

    Parameters
    ----------
    model_path: str
        The path to the subject's scaled bioMod
    phases: list[dict]
        The movements this step is run on, as movement_ocp.collect_movement_phases builds them
    passive_torque: PassiveTorque | RienerPassiveTorque
        The articular torque identified beforehand
    free: tuple
        The names of the parameters this step identifies
    formulation: str
        Which passive formulation `passive_torque` is, see movement_ocp.PASSIVE_CLASSES
    initial_states: list
        One (4, n_shooting + 1) array per phase, holding Cn, F, q and qdot to start from
    initial_guesses: dict
        Where each named parameter starts from, clipped to its bounds
    passive_theta_bounds: tuple
        The angle range the passive torque stays live over, given explicitly so every step shares it
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
        free,
        passive_torque,
        formulation,
        max_elbow_position=max(float(anchor or 0.0), float(reached.max())),
        include_passive_stop=identify_passive_stop,
    )

    for name, value in (initial_guesses or {}).items():
        if name in settings:
            bounds = (settings[name]["min_bound"], settings[name]["max_bound"])
            settings[name] = {**settings[name], "initial_guess": float(np.clip(value, *bounds))}

    # --- The identified parameters are common to every phase, declared once for the whole ocp --- #
    parameters, parameters_bounds, parameters_init = OcpFesId.set_parameters(
        parameter_to_identify=list(settings),
        parameter_setting=settings,
        use_sx=use_sx,
    )

    # models = [movement_ocp.build_model(model_path, phase["stim_time"], passive_torque) for phase in phases]
    models = [movement_ocp.build_model(model_path, phase["stim_time"], None) for phase in phases]
    muscle_models = [instance.muscles_dynamics_model[0] for instance in models]

    dynamics = DynamicsOptionsList()
    x_bounds, x_init = BoundsList(), InitialGuessList()
    u_bounds, u_init = BoundsList(), InitialGuessList()
    objective_functions = ObjectiveList()
    targets = []

    # One grid per phase, aligned on its own stimulation period
    grid = [movement_ocp.aligned_shooting_grid(phase["final_time"], phase["stim_time"], n_shooting_per_phase)
            for phase in phases]
    final_times = [duration for _, duration in grid]
    offsets = np.concatenate(([0.0], np.cumsum(final_times)[:-1]))

    for i, (instance, phase) in enumerate(zip(models, phases)):
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
                numerical_data_timeseries=movement_ocp.get_numerical_data_time_series(
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
                guess = np.full((1, n_shooting + 1), movement_ocp.MUSCLE_STATE_GUESS[j])
                guess[0, 0] = float(np.squeeze(rest[j]))
            else:
                guess = np.asarray(initial_states[i], dtype=float)[j][np.newaxis, :]
            x_init.add(name, guess, interpolation=InterpolationType.EACH_FRAME, phase=i)

        q_bounds = instance.bounds_from_ranges("q")
        q_bounds.min[0][0] = q_bounds.max[0][0] = target[0]
        qdot_bounds = instance.bounds_from_ranges("qdot")
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
        key = "last_pulse_width_" + movement_ocp.MUSCLE_NAME
        control = np.array([[phase["pulse_width"]] * n_shooting])
        u_init.add(key=key, initial_guess=control, interpolation=InterpolationType.EACH_FRAME, phase=i)
        u_bounds.add(key, min_bound=control, max_bound=control, interpolation=InterpolationType.EACH_FRAME, phase=i)

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

        models[i] = movement_ocp.update_model(instance, parameters)
        update_model_param(muscle_models[i], parameters, free)

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
        objective_functions=objective_functions,
        parameters=parameters,
        parameter_bounds=parameters_bounds,
        parameter_init=parameters_init,
        phase_transitions=phase_transitions,
        control_type=ControlType.CONSTANT,
        use_sx=use_sx,
        n_threads=movement_ocp.N_THREADS,
    )
    return ocp, targets, final_times


def solve(subject, phases, free, theta_bounds, model_class, constants=None, passive_method=PASSIVE_METHOD,
          initial_states=None, initial_guesses=None, identify_passive_stop=False, max_iter=5000) -> dict:
    """
    Build and solve one step's ocp.

    Parameters
    ----------
    phases: list[dict]
        The movements this step is run on
    free: tuple
        The names of the parameters this step identifies. Everything else is carried on the model, at the
        value `constants` gives it, and never enters the ocp.
    theta_bounds: tuple
        The passive torque's live range, shared by every step
    model_class: type
        DingModelSingleIntensity for steps 1 and 2, DingModelArticle once eq. 3 is known
    constants: dict
        What the earlier steps settled, and what the article fixes rather than identifies, see pilot.py
    """
    passive_torque, formulation, path = movement_ocp.load_passive_torque(subject, passive_method)
    model_path = str(COMPARISON_ROOT / "model" / f"p{subject}_scaling_scaled.bioMod")
    with article_model(model_class, constants):
        ocp, targets, final_times = prepare_ocp(
            model_path,
            phases,
            passive_torque,
            free,
            formulation=formulation,
            initial_states=initial_states,
            initial_guesses=initial_guesses,
            identify_passive_stop=identify_passive_stop,
            passive_theta_bounds=theta_bounds,
        )
        sol = ocp.solve(Solver.IPOPT(_max_iter=max_iter))

    lengths = [len(target) for target in targets]
    _, identified = results.solution_at_nodes(sol, "q", [length - 1 for length in lengths])
    starts = np.concatenate(([0], np.cumsum(lengths)))
    parameters = {key: float(value.squeeze()) for key, value in sol.decision_parameters().items()}
    return {
        "parameters": parameters,
        "targets": list(targets),
        "identified": [identified[starts[i]:starts[i + 1]] for i in range(len(lengths))],
        "times": [np.linspace(0, final_times[i], lengths[i]) for i in range(len(lengths))],
        "offsets": np.concatenate(([0.0], np.cumsum(final_times)[:-1])),
        "final_times": final_times,
        "at_bound": results.parameters_at_bound(ocp, parameters),
        "cost": float(np.sum(sol.cost)),
        "converged": bool(sol.status == 0),
        "iterations": int(getattr(sol, "iterations", -1) or -1),
        "formulation": formulation,
        "bounds": {name: (float(ocp.parameter_bounds[name].min.min()),
                          float(ocp.parameter_bounds[name].max.max()))
                   for name in ocp.parameter_bounds.keys()},
    }


# ---- Step 5, scoring the prediction ---- #
def score(step, phases, set_name) -> tuple:
    """
    Score every movement of one step the way the article scores its testing trains.

    The counterpart of isometric.score on the flexion. The peak and the integral are taken above the angle
    each movement started from, which is the excursion the stimulation produced rather than the absolute
    elbow angle, so the two routes score the same quantity, see the module docstring.

    Parameters
    ----------
    step: dict
        What `solve` returned, holding "times", "targets" and "identified" per movement
    phases: list[dict]
        The movements it was run on, in the same order
    set_name: str
        "testing" or "identification", the column the article's tables pool on

    Returns
    -------
    tuple
        One row per movement, and the starting angle each was scored above
    """
    rows, baselines = [], []
    for i, phase in enumerate(phases):
        measured = np.asarray(step["targets"][i], dtype=float)
        predicted = np.asarray(step["identified"][i], dtype=float)
        baseline = float(measured[0])
        baselines.append(baseline)
        rows.append(
            {
                "frequency": phase["frequency"],
                "pulse_width_us": protocol.as_microseconds(phase["pulse_width"]),
                "condition": protocol.condition_name(phase["frequency"], phase["pulse_width"]),
                "set": set_name,
                **analysis.score_train(step["times"][i], measured, predicted, baseline=baseline),
            }
        )

    return pd.DataFrame(rows), baselines


# ---- Running the four steps on one subject ---- #
def identify(subject, frequencies=None, passive_method=PASSIVE_METHOD, save=True, debug=True,
             show_debug=False, identify_passive_stop=False, max_iter=5000,
             tau2=model.TAU2_LITERATURE, eq3="fit") -> dict:
    """
    The whole article algorithm on one subject's elbow flexions.

    Parameters
    ----------
    subject: str
        The subject id, ex: "01"
    frequencies: list[int]
        The recordings to read, defaults to the three unweighted ones
    passive_method: str
        The stored passive torque identification the articular torque is taken from
    identify_passive_stop: bool
        Whether the passive flexion stop moves with the muscle parameters
    max_iter: int
        The solver's iteration cap, per step
    tau2: float
        Hold tau2 at this value (s) instead of identifying it, `model.TAU2_LITERATURE` by default. None
        identifies it at step 2 alongside A and Km, which is what the article does. Held by default because
        free it fails in both routes and in opposite directions on this cohort, see model.TAU2_LITERATURE.
    eq3: str
        How a', pd0 and pdt are obtained. "fit" is the article's step 4: one A per intensity at step 3, then
        eq. 3 regressed through the six points. "ocp" merges steps 3 and 4 into one solve over every
        identification train, with eq. 3 inside the dynamics, so no six points exist to scatter. On this
        cohort the two agree closely and "fit" predicts marginally better, so "fit" stays the default and
        "ocp" is the cross check.
    """
    # --- Print and plots --- #
    debug_plots.output_for(METHOD, subject, debug=debug, show=show_debug)
    plots.output_for(METHOD, subject, debug=debug, show=show_debug)
    print(f"=== P{subject} ===")


    ### --- Data preparation --- ###
    # --- Motion curve --- #
    phases = load_movements(subject, frequencies=frequencies, debug=debug)

    # --- Sort data and condition for identification protocol --- #
    conditions = conditions_of(phases)
    reference, other_intensities = protocol.steps(*conditions)
    identification_index, testing_index = protocol.split(*conditions)
    protocol.report_split(*conditions, label=f"P{subject}: ")
    theta_bounds = theta_bounds_of(phases)

    # --- Building the reference trains for steps 1 and 2 identification --- #
    reference_phases = [phases[i] for i in reference]
    reference_times = [phase["time"] for phase in reference_phases]
    reference_excursions = [np.rad2deg(np.asarray(phase["q"], dtype=float) - phase["q"][0])
                            for phase in reference_phases]
    labels = [protocol.condition_name(phase["frequency"], phase["pulse_width"])
              for phase in reference_phases]


    ### --- Identification procedure --- ###
    watch = analysis.Stopwatch()

    # --- Step 1: tau1 off the relaxations, with no solver --- #
    tau1, estimates = relaxation.mean_tau1(
        reference_times,
        reference_excursions,
        debug=debug,
        subject=subject,
        labels=labels,
        **TAU1_FIGURE,
    )
    # tau2 either rides on the model at the value given, or joins the free set of step 2
    step2_free = STEP2_FREE + (("tau2",) if tau2 is None else ())
    constants = {"tau1_rest": tau1} if tau2 is None else {"tau1_rest": tau1, "tau2": tau2}
    watch.lap("step1")
    print(f"Step 1: tau1 = {1e3 * tau1:.1f} ms")

    # --- Step 2: A, Km and tau2 at the reference pulse train --- #
    first = solve(subject,
                  phases=[phases[i] for i in reference],
                  free=step2_free,
                  theta_bounds=theta_bounds, model_class=model.DingModelSingleIntensity,
                  constants=constants, passive_method=passive_method,
                  identify_passive_stop=identify_passive_stop, max_iter=max_iter)
    # Everything step 2 settled, handed to the next step as it stands rather than named parameter by parameter
    watch.lap("step2")
    carried = {**constants, **first["parameters"]}
    print(f"Step 2: A = {first['parameters']['A']:.1f} N/s, Km = {carried['km_rest']:.4f}, "
          f"tau2 = {1e3 * carried['tau2']:.1f} ms, converged = {first['converged']}")

    if debug:
        plots.plot_step2_fit(
            times=first["times"],
            tracked=first["targets"],
            identified=first["identified"],
            parameters=first["parameters"],
            subject=subject,
            labels=labels,
            free=step2_free,
            at_bound=[name for name in first["at_bound"] if name in step2_free],
            converged=first["converged"],
            cost=first["cost"],
            folder=plots.MOVEMENT_STEP2_ROOT,
            ylabel="Elbow angle (deg)",
            unit="deg",
            scale=180 / np.pi,
            route="movement step 2",
        )

    # --- Step 3: A alone, one ocp per remaining identification intensity --- #
    a_points = {protocol.REFERENCE_PULSE_WIDTH_US: first["parameters"]["A"]}
    at_bound = [name for name in first["at_bound"] if name in step2_free]
    steps, groups = {}, []
    if eq3 == "ocp":
        # Steps 3 and 4 in one solve: eq. 3 sits inside the dynamics, so a', pd0 and pdt answer to every
        # identification flexion at once rather than to six independently estimated A. A is not handed over,
        # DingModelArticle derives it from the three at the pulse width of each phase.
        held = {name: value for name, value in carried.items() if name != "A"}
        joint = solve(subject, [phases[i] for i in identification_index], free=STEP34_FREE,
                      theta_bounds=theta_bounds, model_class=model.DingModelArticle, constants=held,
                      passive_method=passive_method, identify_passive_stop=identify_passive_stop,
                      max_iter=max_iter)
        relationship = {name: joint["parameters"][name] for name in STEP34_FREE}
        relationship["r2"], relationship["monotonic"] = np.nan, True
        relationship["bounds"] = {name: IDENTIFIED_PARAMETERS[name][1:3] for name in STEP34_FREE}
        at_bound += [name for name in joint["at_bound"] if name in STEP34_FREE]
        steps["joint"] = joint
        print(f"  steps 3+4 (ocp): a' = {relationship['a_scale']:.1f} N/s, "
              f"pd0 = {1e6 * relationship['pd0']:.1f} us, pdt = {1e6 * relationship['pdt']:.1f} us, "
              f"converged = {joint['converged']}")
    # Continuation along the intensity axis, see the isometric route
    seed = first["parameters"]["A"]
    for pulse_width_us, indices in (other_intensities if eq3 != "ocp" else []):
        step = solve(subject, [phases[i] for i in indices], free=STEP3_FREE,
                     theta_bounds=theta_bounds, model_class=model.DingModelSingleIntensity,
                     constants=carried, passive_method=passive_method,
                     initial_guesses={"A": seed},
                     identify_passive_stop=identify_passive_stop, max_iter=max_iter)
        seed = step["parameters"]["A"]
        a_points[pulse_width_us] = step["parameters"]["A"]
        at_bound += [name for name in step["at_bound"] if name in STEP3_FREE]
        steps[pulse_width_us] = step
        groups.append((pulse_width_us, step,
                       [protocol.condition_name(phases[i]["frequency"], phases[i]["pulse_width"])
                        for i in indices]))
        print(f"  step 3: {pulse_width_us:.0f} us -> A = {step['parameters']['A']:.1f} N/s "
              f"({step['iterations']} iterations, converged = {step['converged']})")

    if debug:
        plots.plot_step3_fit(groups, subject, folder=plots.MOVEMENT_STEP3_ROOT,
                             ylabel="Elbow angle (deg)", unit="deg", scale=180 / np.pi,
                             route="movement step 3")

    watch.lap("step3")

    # --- Step 4: the force-intensity relationship through those points --- #
    ordered = sorted(a_points.items())
    pulse_widths = np.array([pulse_width for pulse_width, _ in ordered]) * 1e-6
    a_values = np.array([a for _, a in ordered])
    if eq3 != "ocp":
        relationship = model.fit_a_pulse_width(pulse_widths, a_values)
    print(f"  step 4: a' = {relationship['a_scale']:.1f} N/s, pd0 = {1e6 * relationship['pd0']:.1f} us, "
          f"pdt = {1e6 * relationship['pdt']:.1f} us (R2 = {relationship['r2']:.4f})")

    if debug:
        plots.plot_step4_relationship(pulse_widths, a_values, relationship, subject,
                                      reference_us=protocol.REFERENCE_PULSE_WIDTH_US,
                                      folder=plots.MOVEMENT_STEP4_ROOT, route="movement step 4")

    parameters = {
        "a_scale": relationship["a_scale"],
        "pd0": relationship["pd0"],
        "pdt": relationship["pdt"],
        "tau1_rest": carried["tau1_rest"],
        "tau2": carried["tau2"],
        "km_rest": carried["km_rest"],
    }

    watch.lap("step4")

    # --- Step 5: prediction --- #
    testing = [phases[i] for i in testing_index]
    identification = [phases[i] for i in identification_index]
    print(f"  step 5: solving the {len(testing)} testing movement(s) with every parameter fixed")
    prediction = solve(subject, testing, free=(), theta_bounds=theta_bounds,
                       model_class=model.DingModelArticle, constants=parameters,
                       passive_method=passive_method, identify_passive_stop=False, max_iter=max_iter)
    refit = solve(subject, identification, free=(), theta_bounds=theta_bounds,
                  model_class=model.DingModelArticle, constants=parameters,
                  passive_method=passive_method, identify_passive_stop=False, max_iter=max_iter)

    testing_table, testing_baselines = score(prediction, testing, "testing")
    identification_table, fit_baselines = score(refit, identification, "identification")
    table = pd.concat([identification_table, testing_table], ignore_index=True)
    watch.lap("step5")
    print(analysis.report_prediction(testing_table))
    print(f"          {watch.total():.1f} s total: "
          + ", ".join(f"{key} {value:.1f} s" for key, value in watch.laps.items()))

    if debug:
        plots.plot_step5_prediction(prediction["times"], prediction["targets"], prediction["identified"],
                                    testing_table, subject, folder=plots.MOVEMENT_STEP5_ROOT,
                                    ylabel="Elbow angle (deg)", unit="deg", scale=180 / np.pi,
                                    route="movement step 5")
        draw(subject, relationship, pulse_widths, a_values, table, parameters,
             (testing, prediction, testing_baselines), (identification, refit, fit_baselines))

    if save:
        limits = {**relationship["bounds"], **{name: first["bounds"].get(name, (np.nan, np.nan))
                                               for name in ("tau1_rest", "tau2", "km_rest")}}
        results.save(
            subject=subject,
            method=METHOD,
            parameters=parameters,
            time=np.concatenate([prediction["offsets"][i] + prediction["times"][i]
                                 for i in range(len(testing))]),
            tracked=np.concatenate([np.asarray(target) for target in prediction["targets"]]),
            identified=np.concatenate(prediction["identified"]),
            unit="rad",
            converged=bool(first["converged"] and all(step["converged"] for step in steps.values())),
            at_bound=sorted(set(at_bound)),
            bounds=(np.array([limits[name][0] for name in parameters]),
                    np.array([limits[name][1] for name in parameters])),
            phase_lengths=[len(piece) for piece in prediction["identified"]],
            extra={
                "route": "movement", "eq3": eq3,
                "tau2_identified": "tau2" in step2_free,
                "reference_pulse_width_us": protocol.REFERENCE_PULSE_WIDTH_US,
                "tauc": TAUC,
                "r0": R0,
                # tau1 is read off the flexion relaxation, the same way and with the same file the isometric
                # route reads it off the force, see the step 1 block of `identify`
                "tau1_from_relaxation": True,
                "tau1_estimates": estimates,
                "identify_passive_stop": identify_passive_stop,
                "identification_conditions": list(identification_table["condition"]),
                "testing_conditions": list(testing_table["condition"]),
                "frequency": [phase["frequency"] for phase in testing],
                "pulse_width": [phase["pulse_width"] for phase in testing],
                "a_points_us": {f"{key:.0f}": value for key, value in a_points.items()},
                "a_pulse_width_r2": relationship["r2"],
                "a_pulse_width_monotonic": relationship["monotonic"],
                # The product both routes can actually measure, see the isometric route for why
                "plateau_at_reference": a_points[protocol.REFERENCE_PULSE_WIDTH_US] * carried["tau1_rest"],
                "step_converged": {"reference": first["converged"],
                                   **{f"{key:.0f}": step["converged"] for key, step in steps.items()},
                                   "prediction": prediction["converged"]},
                "passive_method": passive_method,
                "passive_formulation": first["formulation"],
                "trains": table.to_dict(orient="list"),
                "step_seconds": watch.laps,
            },
        )

    return {"subject": subject, "parameters": parameters, "trains": table, "a_points": a_points}


def draw(subject, relationship, pulse_widths, a_values, table, parameters, testing, identification):
    """The per subject figures of one identification."""
    plots.plot_force_intensity(
        pulse_widths, a_values, relationship, title=f"P{subject} force-intensity relationship (eq. 3)"
    )
    for label, (group, step, baselines), name in (
        ("predicted testing movements", testing, "testing_predictions"),
        ("identification movements", identification, "identification_fit"),
    ):
        plots.plot_predictions(
            times=step["times"],
            measured=step["targets"],
            predicted=step["identified"],
            conditions=group,
            unit="deg",
            ylabel="Elbow excursion (deg)",
            title=f"P{subject} {label}",
            scale=180 / np.pi,
            baselines=baselines,
            name=name,
        )

    in_degrees = table.copy()
    for column in ("peak_measured", "peak_predicted", "fti_measured", "fti_predicted"):
        in_degrees[column] = in_degrees[column] * 180 / np.pi
    plots.plot_agreement(in_degrees, unit="deg", title=f"P{subject} measured against predicted")

    group, step, _ = testing
    debug_plots.plot_solution(
        time=np.concatenate([step["offsets"][i] + step["times"][i] for i in range(len(group))]),
        tracked=np.concatenate([np.asarray(target) for target in step["targets"]]),
        identified=np.concatenate(step["identified"]),
        title=f"P{subject} Ding 2007 prediction of the testing movements",
        unit="deg",
        ylabel="Elbow angle (deg)",
        scale=180 / np.pi,
        phase_lengths=[len(piece) for piece in step["identified"]],
        conditions=[
            {
                "t_start": float(step["offsets"][i]),
                "t_end": float(step["offsets"][i] + step["final_times"][i]),
                "pulse_width_us": group[i]["pulse_width_us"],
                "frequency_hz": group[i]["frequency"],
            }
            for i in range(len(group))
        ],
        parameters=parameters,
        name="ding_movement_prediction",
    )


def main(subjects=range(1, 2), frequencies=None, passive_method=PASSIVE_METHOD, save=True, debug=True,
         show_debug=False, identify_passive_stop=False, max_iter=5000,
         tau2=model.TAU2_LITERATURE, eq3="fit"):
    """Run the five steps on a list of subjects, in sequence."""
    for subject in subjects:
        subject = f"{int(subject):02d}"
        try:
            identify(subject, frequencies=frequencies, passive_method=passive_method, save=save,
                     debug=debug, show_debug=show_debug, identify_passive_stop=identify_passive_stop,
                     max_iter=max_iter, tau2=tau2, eq3=eq3)
        except Exception as error:  # one bad subject must not take the rest of the batch down
            print(f"P{subject}: {type(error).__name__}: {error}")

    print()
    print(results.summary_table(METHOD))


if __name__ == "__main__":
    main()
