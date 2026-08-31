"""
The optimal control program every method uses for the identification process.

Every route builds the same program over the same node grid, with the same states, the same fixed pulse
width and the same parameter declaration. Two switches, and nothing else, tell them apart:

    kinematics  "none"        no skeleton at all, the Ding model alone
                "prescribed"  the musculoskeletal model, the elbow held on the measured trajectory (or on
                              90 deg for the isometric trials) by a residual joint torque, which is the
                              reaction the handle applies

                "free"        the musculoskeletal model answering to the muscle alone
    cost        "force"       the tracked signal is the muscle force state
                "angle"       the tracked signal is the elbow angle state
                "torque"      the tracked signal is the joint torque the muscle produces, -r(q) * F

The musculoskeletal model carries the force-length and force-velocity relationships.
"""

from __future__ import annotations

import os

import numpy as np
from casadi import Function

from bioptim import (
    BoundsList,
    ConstraintFcn,
    ConstraintList,
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
)

from cocofest import FesMskModel, OcpFesId

import store

import observables
import force_ocp
from helper import results
from helper.passive_torque_model import PassiveTorque, PassiveTorqueFesMskModel
import ding_model as article_model
import movement_ocp

MUSCLE_NAME = movement_ocp.MUSCLE_NAME
FMAX = 1000.0
POLYNOMIAL_DEGREE = 5
N_THREADS = os.cpu_count() or 1

MUSCLE_STATE_GUESS = (0.3, 25.0)
RESIDUAL_TORQUE_BOUND = 200.0


# ---- The models ---- #
def build_muscle(model_class, stim_time, constants=None, muscle_name=None):
    """One phase's Ding model"""
    muscle = model_class(
        muscle_name=muscle_name,
        stim_time=list(stim_time),
        sum_stim_truncation=article_model.SUM_STIM_TRUNCATION,
        **(constants or {}),
    )
    muscle.fmax = FMAX
    return muscle


PASSIVE_METHOD = "passive_torque_id_all_riener"
PASSIVE_CLIP_MARGIN = movement_ocp.PASSIVE_CLIP_MARGIN


def passive_torque_for(subject, kind):
    """
    The articular torque a route asks for, or None when it asks for none.
    """
    if not kind:
        return None
    if kind == "default":
        return PassiveTorque()

    original = results.RESULTS_ROOT
    results.RESULTS_ROOT = store.DATA_ROOT
    try:
        torque, _, path = movement_ocp.load_passive_torque(subject, PASSIVE_METHOD)
    finally:
        results.RESULTS_ROOT = original

    if torque is None:
        raise FileNotFoundError(f"No identified passive torque for P{subject}, looked in {path}")
    return torque


def build_msk(path, muscle, prescribed, passive=None, parameters=None):
    """
    The FES driven musculoskeletal model of the elbow.
    """
    kwargs = dict(
        name=None,
        biorbd_path=path,
        muscles_model=[muscle],
        stim_time=list(muscle.stim_time),
        activate_force_length_relationship=True,
        activate_force_velocity_relationship=True,
        activate_passive_force_relationship=False,
        activate_residual_torque=prescribed,
        parameters=parameters,
        external_force_set=None,
    )
    return PassiveTorqueFesMskModel(passive_torque=passive, **kwargs) if passive else FesMskModel(**kwargs)


def update_model(instance, parameters):
    """Rebuild a musculoskeletal model with the ocp parameters attached."""
    kwargs = dict(
        name=instance.name,
        biorbd_path=instance.biorbd_path,
        muscles_model=instance.muscles_dynamics_model,
        stim_time=instance.muscles_dynamics_model[0].stim_time,
        previous_stim=instance.muscles_dynamics_model[0].previous_stim,
        activate_force_length_relationship=instance.activate_force_length_relationship,
        activate_force_velocity_relationship=instance.activate_force_velocity_relationship,
        activate_passive_force_relationship=instance.activate_passive_force_relationship,
        activate_residual_torque=instance.activate_residual_torque,
        parameters=parameters,
        external_force_set=None,
    )
    passive = getattr(instance, "passive_torque", None)
    return PassiveTorqueFesMskModel(passive_torque=passive, **kwargs) if passive else FesMskModel(**kwargs)


def moment_arm_function(instance) -> Function:
    """The biceps moment arm as a casadi Function of (q, qdot), read off the model of the phase."""
    names = [instance.model.muscle(i).name().to_string() for i in range(len(instance.model.muscles()))]
    q, qdot = instance.q, instance.qdot
    updated = instance.model.UpdateKinematicsCustom(q, qdot)
    instance.model.updateMuscles(updated, q, qdot)
    jacobian = instance.model.musclesLengthJacobian(updated, q, False).to_mx()
    return Function("moment_arm", [q, qdot], [jacobian[names.index(MUSCLE_NAME), :]])


def joint_torque(controller, moment_arm):
    """The joint torque the muscle produces at one node, the quantity the torque routes track."""
    return -moment_arm(controller.states["q"].cx, controller.states["qdot"].cx) * controller.states[
        "F_" + MUSCLE_NAME
    ].cx


# ---- The program ---- #
def set_parameter_settings(models, free, msk, initial_guesses=None, table=None):
    """
    The identification settings of the parameters this step leaves free.

    What the step does not identify is absent rather than pinned with equal bounds: it rides on the model
    at the value an earlier step settled it to, and never enters the KKT system.
    """
    settings = {
        name: {
            "initial_guess": table[name][0],
            "min_bound": table[name][1],
            "max_bound": table[name][2],
            "function": movement_ocp._muscle_setter(name) if msk else [getattr(m, f"set_{name}") for m in models],
            "scaling": table[name][3],
        }
        for name in free
    }

    for name, value in (initial_guesses or {}).items():
        if name in settings:
            bounds = (settings[name]["min_bound"], settings[name]["max_bound"])
            settings[name] = {**settings[name], "initial_guess": float(np.clip(value, *bounds))}

    return settings


def apply_parameters(muscle, parameters, free):
    """Apply the parameters this step identifies on one phase's muscle model."""
    for name in free:
        getattr(muscle, f"set_{name}")(muscle, parameters[name].cx * parameters[name].scaling.scaling)


def prepare_ocp(subject, phases, free, model_class, table, constants=None, initial_guesses=None,
                kinematics="prescribed", cost="force", passive=None, use_sx=False):
    """
    Build the multiphase program of one step, one phase per train or per flexion.
    """
    msk = kinematics != "none"
    joint = passive_torque_for(subject, passive) if msk else None
    if joint is not None:
        reached = np.concatenate([phase["q"] if phase["q"] is not None else phase["target"]
                                  for phase in phases])
        margin = PASSIVE_CLIP_MARGIN * float(reached.max() - reached.min())
        joint.theta_bounds = (float(reached.min()) - margin, float(reached.max()) + margin)
    muscles = [build_muscle(model_class, phase["stim_time"], constants, MUSCLE_NAME if msk else None)
               for phase in phases]
    models = [build_msk(observables.model_path(subject), muscle, kinematics == "prescribed", joint)
              for muscle in muscles] \
        if msk else muscles

    parameters, parameters_bounds, parameters_init = OcpFesId.set_parameters(
        parameter_to_identify=list(free),
        parameter_setting=set_parameter_settings(muscles, free, msk, initial_guesses, table),
        use_sx=use_sx,
    )

    dynamics = DynamicsOptionsList()
    x_bounds, x_init = BoundsList(), InitialGuessList()
    u_bounds, u_init = BoundsList(), InitialGuessList()
    objective_functions = ObjectiveList()
    constraints = ConstraintList()
    arm = moment_arm_function(models[0]) if cost == "torque" else None

    for i, phase in enumerate(phases):
        n_shooting, final_time = phase["n_shooting"], phase["final_time"]
        dynamics.add(
            DynamicsOptions(
                expand_dynamics=True,
                phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
                ode_solver=OdeSolver.COLLOCATION(polynomial_degree=POLYNOMIAL_DEGREE, method="radau"),
                numerical_data_timeseries=force_ocp.get_numerical_data_time_series(
                    model=muscles[i],
                    n_shooting=n_shooting,
                    final_time=final_time,
                    stim_time=phase["stim_time"],
                    previous_model=muscles[i - 1] if i > 0 else None,
                    time_offset=phase["time_offset"],
                ),
                phase=i,
            )
        )

        _set_muscle_states(x_bounds, x_init, muscles[i], phase, msk, i, cost)
        if msk:
            _set_skeleton_states(x_bounds, x_init, models[i], phase, kinematics, i)
            _set_controls(u_bounds, u_init, phase, kinematics, i)
            if kinematics == "prescribed":
                constraints.add(ConstraintFcn.TRACK_STATE, key="q", node=Node.ALL,
                                target=phase["q"][np.newaxis, :], min_bound=0, max_bound=0, phase=i)
        else:
            force_ocp.set_u_bounds(u_bounds, u_init, phase["pulse_width"], n_shooting, phase=i)

        _add_cost(objective_functions, phase, cost, msk, arm, i)

        if msk:
            models[i] = update_model(models[i], parameters)
            apply_parameters(muscles[i], parameters, free)
        else:
            force_ocp.update_model_param(models[i], parameters, i)

    # --- Each train and each flexion is an independent trial, the state is not carried over --- #
    phase_transitions = PhaseTransitionList()
    for i in range(len(phases) - 1):
        phase_transitions.add(PhaseTransitionFcn.DISCONTINUOUS, phase_pre_idx=i)

    return OptimalControlProgram(
        bio_model=models,
        dynamics=dynamics,
        n_shooting=[phase["n_shooting"] for phase in phases],
        phase_time=[phase["final_time"] for phase in phases],
        x_init=x_init,
        x_bounds=x_bounds,
        u_init=u_init,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        constraints=constraints,
        parameters=parameters,
        parameter_bounds=parameters_bounds,
        parameter_init=parameters_init,
        phase_transitions=phase_transitions,
        control_type=ControlType.CONSTANT,
        use_sx=use_sx,
        n_threads=max(1, min(N_THREADS, len(phases))),
    )


def muscle_force_guess(phase, cost) -> np.ndarray:
    """
    Where the F state starts, read off the target rather than off a flat number.
    """
    if cost == "force":
        return np.asarray(phase["target"], dtype=float)
    if cost == "torque":
        return np.asarray(phase["target"], dtype=float) / np.abs(phase["moment_arm"])
    return np.full(phase["n_shooting"] + 1, MUSCLE_STATE_GUESS[1])


def _set_muscle_states(x_bounds, x_init, muscle, phase, msk, index, cost="force"):
    """The muscle starts every trial at rest, and is free between rest and its ceiling afterwards."""
    if not msk:
        force_ocp.set_x_bounds(x_bounds, x_init, muscle, phase["target"][np.newaxis, :], phase=index)
        return

    rest = muscle.standard_rest_values()
    ceiling = np.array([[10.0], [muscle.fmax]])
    for j, name in enumerate(muscle.name_dofs):
        x_bounds.add(
            name,
            min_bound=np.array([np.concatenate((rest, rest, rest), axis=1)[j]]),
            max_bound=np.array([np.concatenate((rest, ceiling, ceiling), axis=1)[j]]),
            phase=index,
            interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
        )
        guess = (muscle_force_guess(phase, cost) if name.startswith("F_")
                 else np.full(phase["n_shooting"] + 1, MUSCLE_STATE_GUESS[j]))
        guess = np.clip(guess, 0.0, float(np.squeeze(ceiling[j])))[np.newaxis, :].copy()
        guess[0, 0] = float(np.squeeze(rest[j]))
        x_init.add(name, guess, interpolation=InterpolationType.EACH_FRAME, phase=index)


def _set_skeleton_states(x_bounds, x_init, instance, phase, kinematics, index):
    """The elbow, either held on the trajectory that was measured or free to answer to the muscle."""
    trajectory = phase["q"] if kinematics == "prescribed" else phase["target"]

    q_bounds = instance.bounds_from_ranges("q")
    qdot_bounds = instance.bounds_from_ranges("qdot")
    qdot_bounds.min[0][0] = qdot_bounds.max[0][0] = phase["qdot0"]
    if kinematics == "free":
        q_bounds.min[0][0] = q_bounds.max[0][0] = trajectory[0]
    x_bounds.add(key="q", bounds=q_bounds, phase=index)
    x_bounds.add(key="qdot", bounds=qdot_bounds, phase=index)

    dt = phase["final_time"] / phase["n_shooting"]
    x_init.add(key="q", initial_guess=trajectory[np.newaxis, :], interpolation=InterpolationType.EACH_FRAME,
               phase=index)
    x_init.add(key="qdot", initial_guess=np.gradient(trajectory, dt)[np.newaxis, :],
               interpolation=InterpolationType.EACH_FRAME, phase=index)


def _set_controls(u_bounds, u_init, phase, kinematics, index):
    """The pulse width is the one that was applied, so only the parameters are identified."""
    key = "last_pulse_width_" + MUSCLE_NAME
    control = np.array([[phase["pulse_width"]] * phase["n_shooting"]])
    u_init.add(key=key, initial_guess=control, interpolation=InterpolationType.EACH_FRAME, phase=index)
    u_bounds.add(key, min_bound=control, max_bound=control, interpolation=InterpolationType.EACH_FRAME,
                 phase=index)

    if kinematics == "prescribed":
        u_bounds.add("tau", min_bound=[-RESIDUAL_TORQUE_BOUND], max_bound=[RESIDUAL_TORQUE_BOUND], phase=index)
        u_init.add("tau", initial_guess=[0.0], phase=index)


def _add_cost(objective_functions, phase, cost, msk, moment_arm, index):
    """The one term of the cost: the difference between what was measured and what the model produces."""
    target = phase["target"][np.newaxis, :-1]
    if cost == "torque":
        objective_functions.add(
            joint_torque,
            custom_type=ObjectiveFcn.Lagrange,
            moment_arm=moment_arm,
            weight=phase["weight"],
            target=target,
            node=Node.ALL_SHOOTING,
            quadratic=True,
            phase=index,
        )
        return

    key = "q" if cost == "angle" else ("F_" + MUSCLE_NAME if msk else "F")
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_STATE,
        key=key,
        weight=phase["weight"],
        target=target,
        node=Node.ALL_SHOOTING,
        quadratic=True,
        index=[0],
        phase=index,
    )
