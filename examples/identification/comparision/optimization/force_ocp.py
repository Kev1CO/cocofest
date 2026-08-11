"""
Identification of the Ding et al. (2007) pulse width frequency FES model from isometric experimental data.

This module holds everything the identification process needs.
Each stimulation train is a phase of a multiphase ocp, only one parameter set is identified over every train
given to it at once.
"""

import sys
from pathlib import Path

import numpy as np

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

from cocofest import DingModelPulseWidthFrequency, IvpFes, OcpFesId

COMPARISON_ROOT = Path(__file__).resolve().parent.parent
for _root in (COMPARISON_ROOT, Path(__file__).resolve().parent):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
from processing.helper.data_to_ocp import DataToOCP
from helper import debug_plots, results

SUM_STIM_TRUNCATION = 10

# Identified parameters of the Ding 2007 model, as (initial guess, min bound, max bound, scaling).
IDENTIFIED_PARAMETERS = {
    "tau1_rest": (0.11, 1e-4, 1, 0.05),
    "tau2": (0.14, 1e-4, 1, 0.05),
    "km_rest": (0.11, 1e-3, 1, 0.1),
    "a_scale": (900, 1, 10000, 1000.0),
    "pd0": (1.1e-4, 1e-5, 6e-4, 1e-4),
    "pdt": (2.0e-4, 1e-5, 6e-4, 1e-4),
}


def get_numerical_data_time_series(model, n_shooting, final_time, stim_time, previous_model=None, time_offset=0.0):
    """
    Build the numerical time series (the truncated previous stimulation times at each node) of one phase.

    Contrary to FesModel.get_numerical_data_time_series, the node times are expressed in the ocp global time
    (in bioptim, the node time of a phase is offset by the duration of all the previous phases) and the
    stimulation history of the previous phase is carried over, so the truncated sum never holds nan values.

    Parameters
    ----------
    model: DingModelPulseWidthFrequency
        The model of the current phase
    n_shooting: int
        The number of shooting points of the current phase
    final_time: float
        The duration of the current phase
    stim_time: list
        The stimulation times of the current phase, expressed from the beginning of the phase
    previous_model: DingModelPulseWidthFrequency
        The model of the previous phase, None for the first phase
    time_offset: float
        Starting time of the current phase (used for the multiphase problem)

    Returns
    -------
    dict
        The numerical time series of truncated previous stimulation times per node
    """
    truncation = model.sum_stim_truncation

    # --- Set the previous stim time for the numerical data time series (mandatory to avoid nan values) --- #
    if previous_model is None:
        model.previous_stim = model._get_additional_previous_stim_time()
    else:
        model.previous_stim["time"] = list(previous_model.all_stim[-truncation:])

    model.all_stim = model.previous_stim["time"][-truncation:] + [time + time_offset for time in stim_time]
    stim_time = np.array(model.all_stim)
    dt = final_time / n_shooting

    # For each node, keep the last 'truncation' stimulation times that already occurred.
    tolerance = 1e-6 * dt
    node_idx = [np.where(stim_time <= time_offset + i * dt + tolerance)[0][-1] for i in range(n_shooting + 1)]
    stim_time_per_node = np.array([stim_time[: idx + 1][-truncation:] for idx in node_idx])

    # Reshape to the (truncation, 1, n_shooting + 1) format bioptim expects
    return {"stim_time": np.transpose(stim_time_per_node[:, np.newaxis, :], (2, 1, 0))}


def set_x_bounds(x_bounds, x_init, model, force_tracking, phase):
    """Bound every state to its rest value at the first node, and guess F with the tracked force."""
    rest = model.standard_rest_values()
    ceiling = np.array([[10.0 if name == "Cn" else 500.0] for name in model.name_dofs])

    # First node at rest, the two others free between the rest value and the ceiling
    lower = np.concatenate((rest, rest, rest), axis=1)
    upper = np.concatenate((rest, ceiling, ceiling), axis=1)

    for i, name in enumerate(model.name_dofs):
        x_bounds.add(
            name,
            min_bound=np.array([lower[i]]),
            max_bound=np.array([upper[i]]),
            phase=phase,
            interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
        )

    x_init.add("F", force_tracking, phase=phase, interpolation=InterpolationType.EACH_FRAME)
    x_init.add("Cn", [0], phase=phase, interpolation=InterpolationType.CONSTANT)
    return x_bounds, x_init


def set_u_bounds(u_bounds, u_init, control_value, n_shooting, phase):
    """Fix the pulse width to the one applied during the recording, so only the model parameters are identified."""
    control_bounds = np.array([[control_value] * n_shooting])
    u_init.add(
        key="last_pulse_width",
        initial_guess=control_bounds,
        interpolation=InterpolationType.EACH_FRAME,
        phase=phase,
    )
    u_bounds.add(
        "last_pulse_width",
        min_bound=control_bounds,
        max_bound=control_bounds,
        interpolation=InterpolationType.EACH_FRAME,
        phase=phase,
    )
    return u_bounds, u_init


def set_default_values(models):
    """
    The identification settings of every parameter. One setter per phase is given, as the parameters are shared by
    all the phases but each phase owns its own model instance.
    """
    return {
        name: {
            "initial_guess": initial_guess,
            "min_bound": min_bound,
            "max_bound": max_bound,
            "function": [getattr(model, f"set_{name}") for model in models],
            "scaling": scaling,
        }
        for name, (initial_guess, min_bound, max_bound, scaling) in IDENTIFIED_PARAMETERS.items()
    }


def update_model_param(model, parameters, phase):
    """Apply the ocp parameters on the model of the given phase."""
    for key in parameters:
        parameters[key].function[phase](
            model, parameters[key].cx * parameters[key].scaling.scaling, **parameters[key].kwargs
        )


def prepare_ocp(ocp_data, use_sx=False):
    """Build the multiphase ocp identifying the model parameters over every stimulation train at once."""
    n_shooting, final_time = ocp_data["n_shooting"], ocp_data["final_time"]
    force_tracking, pulse_width, stim_time = ocp_data["force"], ocp_data["pulse_width"], ocp_data["stim_time"]
    models = [
        DingModelPulseWidthFrequency(stim_time=stim, sum_stim_truncation=SUM_STIM_TRUNCATION) for stim in stim_time
    ]

    dynamics = DynamicsOptionsList()
    x_bounds, x_init = BoundsList(), InitialGuessList()
    u_bounds, u_init = BoundsList(), InitialGuessList()
    objective_functions = ObjectiveList()

    # --- The identified parameters are common to every phase, they are declared once for the whole ocp --- #
    parameters, parameters_bounds, parameters_init = OcpFesId.set_parameters(
        parameter_to_identify=list(IDENTIFIED_PARAMETERS),
        parameter_setting=set_default_values(models),
        use_sx=use_sx,
    )

    for i in range(len(models)):
        dynamics.add(
            DynamicsOptions(
                expand_dynamics=True,
                phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
                ode_solver=OdeSolver.COLLOCATION(polynomial_degree=5, method="radau"),
                numerical_data_timeseries=get_numerical_data_time_series(
                    model=models[i],
                    n_shooting=n_shooting[i],
                    final_time=final_time[i],
                    stim_time=stim_time[i],
                    previous_model=models[i - 1] if i > 0 else None,
                    time_offset=sum(final_time[:i]),
                ),
                phase=i,
            )
        )
        x_bounds, x_init = set_x_bounds(x_bounds, x_init, models[i], force_tracking[i], phase=i)
        u_bounds, u_init = set_u_bounds(u_bounds, u_init, pulse_width[i], n_shooting[i], phase=i)

        objective_functions.add(
            ObjectiveFcn.Lagrange.TRACK_STATE,
            key="F",
            weight=ocp_data["weight_cost"][i] / 1000,
            target=force_tracking[i],
            node=Node.ALL,
            quadratic=True,
            phase=i,
        )
        update_model_param(models[i], parameters, i)

    # --- Each phase is an independent stimulation train, the muscle state is not carried from one to the next --- #
    phase_transitions = PhaseTransitionList()
    for i in range(len(models) - 1):
        phase_transitions.add(PhaseTransitionFcn.DISCONTINUOUS, phase_pre_idx=i)

    return OptimalControlProgram(
        bio_model=models,
        dynamics=dynamics,
        n_shooting=n_shooting,
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


def node_times(ocp_data):
    """The time vector of every phase, on the ocp global (cumulative over the phases) time base."""
    times, offset = [], 0.0
    for i in range(len(ocp_data["force"])):
        times.append(offset + np.linspace(0, ocp_data["final_time"][i], ocp_data["n_shooting"][i] + 1))
        offset += ocp_data["final_time"][i]
    return times


def recompute_force(ocp_data, parameters, n_integration_steps=10):
    """
    Recompute the force the identified model produces, by integrating it forward over every train with the pulse
    width that was actually applied.

    Parameters
    ----------
    ocp_data: dict
        The ocp data built by DataToOCP
    parameters: dict
        The identified model parameters, name -> value
    n_integration_steps: int
        The number of RK4 steps per shooting interval of the re-integration

    Returns
    -------
    tuple
        The time vector, the recomputed force, and the number of samples of every train
    """
    times, forces, offset = [], [], 0.0
    for i, stim_time in enumerate(ocp_data["stim_time"]):
        model = DingModelPulseWidthFrequency(stim_time=stim_time, sum_stim_truncation=SUM_STIM_TRUNCATION)
        for name, value in parameters.items():
            getattr(model, f"set_{name}")(model, value)

        ivp = IvpFes(
            fes_parameters={"model": model, "pulse_width": ocp_data["pulse_width"][i]},
            ivp_parameters={
                "final_time": ocp_data["final_time"][i],
                "ode_solver": OdeSolver.RK4(n_integration_steps=n_integration_steps),
            },
        )
        result, time = ivp.integrate()
        times.append(offset + np.asarray(time).squeeze())
        forces.append(np.asarray(result["F"][0]).squeeze())
        offset += ocp_data["final_time"][i]

    return np.concatenate(times), np.concatenate(forces), [len(force) for force in forces]


def tracked_force(ocp_data):
    """The experimental force at the shooting nodes, with its time vector, continuous over the trains."""
    return np.concatenate(node_times(ocp_data)), np.concatenate(
        [np.asarray(force[0]) for force in ocp_data["force"]]
    )


def train_conditions(ocp_data):
    """
    The stimulation condition of every train, for the condition strip of the solution plot: the pulse width that
    was applied (from the seed of the recording, resolved in processing/helper/seeds_pulse_width.pkl) and the
    stimulation frequency.
    """
    conditions, offset = [], 0.0
    for i in range(len(ocp_data["final_time"])):
        conditions.append(
            {
                "t_start": offset,
                "t_end": offset + ocp_data["final_time"][i],
                "pulse_width_us": ocp_data["pulse_width"][i] * 1e6,
                "frequency_hz": ocp_data["frequency"][i],
            }
        )
        offset += ocp_data["final_time"][i]
    return conditions


def load_force_data(subject, data_folder="force"):
    """
    The processed force of one subject, as the ocp data dictionary built by DataToOCP.
    Every train of every recording of the subject is a phase of that dictionary.
    """
    data_root = COMPARISON_ROOT / "data" / data_folder / f"P{subject}"
    paths = sorted(str(path) for path in data_root.glob("*force*.pkl"))
    if not paths:
        return None

    data = DataToOCP()
    data.open_files(file_path_list=paths)
    return data.get_data_for_ocp(plot=False)


def select_trains(ocp_data, keep):
    """
    Keep only some trains of the ocp data.

    Parameters
    ----------
    ocp_data: dict
        The full ocp data, one entry per train in each of its lists
    keep: list[int]
        The indices of the trains to keep

    Returns
    -------
    dict
        The same dictionary restricted to those trains
    """
    per_train = (
        "final_time",
        "frequency",
        "n_shooting",
        "force",
        "pulse_width",
        "stim_time",
        "force_decayed",
        "covers_stimulation",
    )
    selected = {key: [ocp_data[key][i] for i in keep] for key in per_train if key in ocp_data}
    # The tracking weights balance the phases against each other, they have to be recomputed on the selection
    longest = max(selected["n_shooting"])
    selected["weight_cost"] = [longest / n for n in selected["n_shooting"]]
    return selected


def train_conditions_of(ocp_data):
    """The stimulation condition of every train, as the hold out helper and the condition strip expect them."""
    return train_conditions(ocp_data)


def predict(ocp_data, parameters):
    """
    Run the identified model forward on phases it was not identified on, and score it against them.
    """
    time, force_identified, phase_lengths = recompute_force(ocp_data, parameters)
    node_time, force_at_node = tracked_force(ocp_data)
    return {
        "time": time,
        "identified": force_identified,
        "tracked": np.interp(time, node_time, force_at_node),
        "phase_lengths": phase_lengths,
        "rmse": results.rmse(force_at_node, np.interp(node_time, time, force_identified)),
    }


def identify(subject, ocp_data, method, plot=True, save=True, debug=True, show_debug=None, max_iter=1000, extra=None):
    """
    Identify the model on the given trains, then report, plot and save the result.

    Parameters
    ----------
    subject: str
        The subject id, ex: "01"
    ocp_data: dict
        The trains to identify on, as built by load_force_data and possibly restricted by select_trains
    method: str
        The name the result is stored under, ex: "force_id_single" or "force_id_all"
    plot: bool
        If the solver's own penalty plot should be shown while it converges
    debug: bool
        If the debug figures should be written under results/debug/<method>/P<subject>
    show_debug: bool
        If the debug figures should also be opened on screen, which blocks until they are closed. Defaults to
        `plot`. Saving does not depend on it, so a batch can be run with it off and still leave every figure.
    """
    debug_plots.output_for(method, subject, debug=debug, show=plot if show_debug is None else show_debug)

    ocp = prepare_ocp(ocp_data)

    # --- Debug, pre ocp: every train on one axes, raw against the target the cost function holds --- #
    if debug:
        debug_plots.plot_tracking_target(
            raw_time=node_times(ocp_data),
            raw_data=[np.asarray(force[0]) for force in ocp_data["force"]],
            target_time=node_times(ocp_data),
            target=debug_plots.cost_function_target(ocp, key="F"),
            title=f"P{subject} force target held by the cost function",
            unit="N",
            name="force_target",
        )

    if plot:
        ocp.add_plot_penalty()
    sol = ocp.solve(solver=Solver.IPOPT(_max_iter=max_iter))

    parameters = {key: float(value.squeeze()) for key, value in sol.decision_parameters().items()}
    at_bound = results.parameters_at_bound(ocp, parameters)

    time, force_identified, phase_lengths = recompute_force(ocp_data, parameters)
    node_time, force_at_node = tracked_force(ocp_data)

    force_tracked = np.interp(time, node_time, force_at_node)
    rmse = results.rmse(force_at_node, np.interp(node_time, time, force_identified))

    print(f"P{subject} RMSE: {rmse:.3f} N  ({len(ocp_data['force'])} train(s))")
    for key, value in parameters.items():
        print(f"  {key}: {value}{'  (at bound)' if key in at_bound else ''}")

    # --- Debug, solution: identified against tracked, the residual, and the train conditions --- #
    if plot or debug:
        debug_plots.plot_solution(
            time=time,
            tracked=force_tracked,
            identified=force_identified,
            title=f"P{subject} isometric identification",
            unit="N",
            ylabel="Force (N)",
            phase_lengths=phase_lengths,
            conditions=train_conditions(ocp_data),
            parameters=parameters,
            name="force_identification",
        )

    if save:
        results.save(
            subject=subject,
            method=method,
            parameters=parameters,
            time=time,
            tracked=force_tracked,
            identified=force_identified,
            unit="N",
            sol=sol,
            at_bound=at_bound,
            bounds=results.parameter_bounds(ocp, parameters),
            phase_lengths=phase_lengths,
            extra={
                "frequency": ocp_data["frequency"],
                "pulse_width": ocp_data["pulse_width"],
                "n_trains": len(ocp_data["force"]),
                **(extra if extra else {}),
            },
        )

    return parameters
