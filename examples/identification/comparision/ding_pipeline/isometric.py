"""
Identification of the Ding et al. (2007) model from isometric experimental data, in the four steps the
article identifies in.

Each step is an ocp built : one phase per stimulation train, the pulse width fixed to the one that was applied,
the model parameters declared through OcpFesId.set_parameters, and the measured force tracked. Only the free parameter
set changes from step to step, and the model class changes with it, see model.py.

    step 1  tau1 alone, off the relaxation of the reference intensity trains, with no solver:
            "Parameter tau1 was calculated by fitting the forces immediately following the tetanic
             stimulation while the muscle relaxed for forces between 50% to 25% of the peak force."
    step 2  the same trains, tracking A, Km and tau2 with tau1 held where step 1 put it:
            "G(A,Km,tau2) = sum_p [Fpred(tp;A,Km,tau2) - Fmeas(tp)]^2"
    step 3  one ocp per remaining identification intensity, with A the only free parameter
    step 4  the (pulse width, A) points are fitted with eq. 3, giving a', pd0 and pdt
    step 5  the whole parameter set is integrated forward over the testing trains, and scored

Which trains are which is protocol.py, the scoring is analysis.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from bioptim import (
    BoundsList,
    ControlType,
    DynamicsOptions,
    DynamicsOptionsList,
    InitialGuessList,
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

from cocofest import IvpFes, OcpFesId

COMPARISON_ROOT = Path(__file__).resolve().parent.parent
for _root in (Path(__file__).resolve().parent, COMPARISON_ROOT, COMPARISON_ROOT / "optimization"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

import analysis  # noqa: E402
import model  # noqa: E402
import plots  # noqa: E402
import protocol  # noqa: E402
import relaxation  # noqa: E402
import force_ocp  # noqa: E402
from helper import debug_plots, results  # noqa: E402

METHOD = "ding_force_id"

# The processing chain already brings each train back to a zero resting force
BASELINE = 0.0

# Fixed in the used FES model
R0 = model.R0  # Value from Kesar et al. (2008)
TAUC = model.TAUC  # Value in (s) from Kesar et al. (2008)

# "A" is the force-scaling factor of one intensity, which DingModelSingleIntensity carries on its own
# attribute. It is not "a_scale", the eq. 3 asymptote a' step 4 fits through the A of every intensity.
IDENTIFIED_PARAMETERS = {
    "A": (900, 1, 20000, 1000.0),
    "km_rest": (0.18, 1e-3, 5, 0.1),
    "tau2": (0.0185, 1e-4, 0.5, 0.05),
    # eq. 3, for the joint step 3 + 4. pd0 is capped per subject by the smallest pulse width applied.
    "a_scale": (2000, 1, 60000, 1000.0),
    "pd0": (1.31405e-4, model.PD0_FLOOR, 2.4e-4, 1e-4),
    "pdt": (1.94138e-4, 1e-5, 2e-3, 1e-4),
}

STEP2_FREE = ("A", "km_rest")
STEP3_FREE = ("A",)

# Step 3 and 4 in one solve: a', pd0 and pdt straight off the identification trains, so the
# intensity information is used coherently instead of being squeezed through six independent A.
STEP34_FREE = ("a_scale", "pd0", "pdt")

N_INTEGRATION_STEPS = 10


# ---- Building one step's ocp ---- #
def set_default_values(models, free, initial_guesses=None):
    """
    The identification settings of the parameters this step leaves free.

    One setter per phase is given, as the parameters are shared by all the phases but each phase owns its
    own model instance.

    Parameters
    ----------
    models: list[DingModelArticle]
        The model of every phase
    free: tuple
        The names of the parameters this step identifies
    initial_guesses: dict
        Where a named parameter starts from, clipped to its bounds, in place of the table's guess. Step 3
        uses it to start each intensity from the A of the one before it rather than from the table.
    """
    settings = {
        name: {
            "initial_guess": IDENTIFIED_PARAMETERS[name][0],
            "min_bound": IDENTIFIED_PARAMETERS[name][1],
            "max_bound": IDENTIFIED_PARAMETERS[name][2],
            "function": [getattr(instance, f"set_{name}") for instance in models],
            "scaling": IDENTIFIED_PARAMETERS[name][3],
        }
        for name in free
    }

    for name, value in (initial_guesses or {}).items():
        if name in settings:
            bounds = (settings[name]["min_bound"], settings[name]["max_bound"])
            settings[name] = {**settings[name], "initial_guess": float(np.clip(value, *bounds))}

    return settings


def prepare_ocp(ocp_data, free, model_class=model.DingModelSingleIntensity, constants=None,
                initial_guesses=None, use_sx=False):
    """
    Build the multiphase ocp of one step, one phase per stimulation train.

    Parameters
    ----------
    ocp_data: dict
        The trains of this step, as force_ocp.load_force_data builds them
    free: tuple
        The names of the parameters this step identifies
    model_class: type
        DingModelSingleIntensity for steps 1 and 2, DingModelArticle once eq. 3 is known
    constants: dict
        What the earlier steps settled. tauc and R0 come with the model class itself, see model.py.
    initial_guesses: dict
        Where each named parameter starts from, clipped to its bounds
    """
    n_shooting, final_time = ocp_data["n_shooting"], ocp_data["final_time"]
    force_tracking, pulse_width, stim_time = ocp_data["force"], ocp_data["pulse_width"], ocp_data["stim_time"]

    constants = constants or {}
    models = [
        model_class(stim_time=stim, sum_stim_truncation=model.SUM_STIM_TRUNCATION, **constants)
        for stim in stim_time
    ]

    dynamics = DynamicsOptionsList()
    x_bounds, x_init = BoundsList(), InitialGuessList()
    u_bounds, u_init = BoundsList(), InitialGuessList()
    objective_functions = ObjectiveList()

    # --- The identified parameters are common to every phase, declared once for the whole ocp --- #
    parameters, parameters_bounds, parameters_init = OcpFesId.set_parameters(
        parameter_to_identify=list(free),
        parameter_setting=set_default_values(models, free, initial_guesses),
        use_sx=use_sx,
    )

    for i in range(len(models)):
        dynamics.add(
            DynamicsOptions(
                expand_dynamics=True,
                phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
                ode_solver=OdeSolver.COLLOCATION(polynomial_degree=5, method="radau"),
                numerical_data_timeseries=force_ocp.get_numerical_data_time_series(
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
        x_bounds, x_init = force_ocp.set_x_bounds(x_bounds, x_init, models[i], force_tracking[i], phase=i)
        u_bounds, u_init = force_ocp.set_u_bounds(u_bounds, u_init, pulse_width[i], n_shooting[i], phase=i)

        objective_functions.add(
            ObjectiveFcn.Lagrange.TRACK_STATE,
            key="F",
            weight=ocp_data["weight_cost"][i] / 1000,
            target=force_tracking[i],
            node=Node.ALL,
            quadratic=True,
            phase=i,
        )
        force_ocp.update_model_param(models[i], parameters, i)

    # --- Each phase is an independent stimulation train, the muscle state is not carried over --- #
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


def solve(ocp_data, free, max_iter=5000, model_class=model.DingModelSingleIntensity,
          constants=None, initial_guesses=None) -> dict:
    """Build and solve one step, and return what it identified, with the fit it identified it on."""
    ocp = prepare_ocp(ocp_data, free, model_class=model_class, constants=constants,
                      initial_guesses=initial_guesses)
    sol = ocp.solve(solver=Solver.IPOPT(_max_iter=max_iter))

    # The tracked force and the solved F at the same nodes, which is what the cost compared, see
    # plots.plot_step2_fit
    n_shooting = ocp_data["n_shooting"]
    _, identified = results.solution_at_nodes(sol, "F", n_shooting)
    starts = np.concatenate(([0], np.cumsum([intervals + 1 for intervals in n_shooting])))

    parameters = {key: float(value.squeeze()) for key, value in sol.decision_parameters().items()}
    return {
        "parameters": parameters,
        "times": force_ocp.node_times(ocp_data),
        "targets": [np.asarray(force[0], dtype=float) for force in ocp_data["force"]],
        "identified": [identified[starts[i]:starts[i + 1]] for i in range(len(n_shooting))],
        "at_bound": results.parameters_at_bound(ocp, parameters),
        "cost": float(np.sum(sol.cost)),
        "converged": bool(sol.status == 0),
        "iterations": int(getattr(sol, "iterations", -1) or -1),
    }


# ---- Step 5, the prediction ---- #
def simulate(ocp_data, parameters, constants=None) -> tuple:
    """
    Integrate the identified model forward over every train, with the pulse width that was applied.

    The counterpart of force_ocp.recompute_force, on the article's model rather than on cocofest's default.

    Returns
    -------
    tuple
        The time vector, the force, and the number of samples of every train
    """
    times, forces, offset = [], [], 0.0
    for i, stim_time in enumerate(ocp_data["stim_time"]):
        instance = model.DingModelArticle(
            stim_time=stim_time, sum_stim_truncation=model.SUM_STIM_TRUNCATION, **(constants or {})
        )
        for name, value in parameters.items():
            getattr(instance, f"set_{name}")(instance, value)

        ivp = IvpFes(
            fes_parameters={"model": instance, "pulse_width": ocp_data["pulse_width"][i]},
            ivp_parameters={
                "final_time": ocp_data["final_time"][i],
                "ode_solver": OdeSolver.RK4(n_integration_steps=N_INTEGRATION_STEPS),
            },
        )
        result, time = ivp.integrate()
        times.append(offset + np.asarray(time).squeeze())
        forces.append(np.asarray(result["F"][0]).squeeze())
        offset += ocp_data["final_time"][i]

    return times, forces, [len(force) for force in forces]


def score(ocp_data, parameters, set_name, constants=None) -> dict:
    """
    Predict every train of `ocp_data` and score it the way the article scores its testing trains.

    Returns
    -------
    dict
        "table" with one row per train, and "times", "measured" and "predicted" per train, all read on the
        integration grid
    """
    times, forces, _ = simulate(ocp_data, parameters, constants=constants)
    node_times = force_ocp.node_times(ocp_data)

    rows, measured = [], []
    for i, (time, force) in enumerate(zip(times, forces)):
        # The measurement lives on the shooting nodes, the prediction on the finer integration grid: both
        # are read on the latter, so a peak or an integral compares two signals and not two resamplings
        tracked = np.interp(time, node_times[i], np.asarray(ocp_data["force"][i][0]))
        measured.append(tracked)
        rows.append(
            {
                "frequency": ocp_data["frequency"][i],
                "pulse_width_us": protocol.as_microseconds(ocp_data["pulse_width"][i]),
                "condition": protocol.condition_name(ocp_data["frequency"][i], ocp_data["pulse_width"][i]),
                "set": set_name,
                **analysis.score_train(time, tracked, force, baseline=BASELINE),
            }
        )

    return {
        "data": ocp_data,
        "table": pd.DataFrame(rows),
        "times": times,
        "measured": measured,
        "predicted": forces,
    }


# ---- Running the four steps on one subject ---- #
def identify(subject, data_folder="force", save=True, debug=True, show_debug=False, max_iter=5000,
             tau2=model.TAU2_LITERATURE, eq3="fit") -> dict:
    """
    The whole article algorithm on one subject.

    Parameters
    ----------
    subject: str
        The subject id, ex: "01"
    save: bool
        Whether the result is written to results/ding_force_id/
    debug: bool
        Whether the per subject figures are written under results/debug/ding_force_id/P<subject>
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

    Returns
    -------
    dict
        The identified parameters and the per train table, or None when the subject does not hold the
        conditions the algorithm needs
    """
    # --- Print and plots --- #
    debug_plots.output_for(METHOD, subject, debug=debug, show=show_debug)
    plots.output_for(METHOD, subject, debug=debug, show=show_debug)
    print(f"=== P{subject} ===")


    ### --- Data preparation --- ###
    # --- Force curve --- #
    ocp_data = force_ocp.load_force_data(subject, data_folder=data_folder)

    # --- Sort data and condition for identification protocol --- #
    trains = usable(ocp_data, subject)
    conditions = (trains["frequency"], trains["pulse_width"])
    reference, other_intensities = protocol.steps(*conditions)
    identification_index, testing_index = protocol.split(*conditions)
    protocol.report_split(*conditions, label=f"P{subject}: ")

    # --- Building the reference trains for steps 1 and 2 identification --- #
    reference_data = force_ocp.select_trains(trains, reference)
    reference_times = force_ocp.node_times(reference_data)
    reference_forces = [np.asarray(force[0]) for force in reference_data["force"]]


    ### --- Identification procedure --- ###
    watch = analysis.Stopwatch()

    # --- Step 1: tau1 off the relaxations, with no solver --- #
    tau1, estimates = relaxation.mean_tau1(
        reference_times,
        reference_forces,
        debug=debug,
        subject=subject,
        labels=[protocol.condition_name(frequency, pulse_width)
                for frequency, pulse_width in zip(reference_data["frequency"], reference_data["pulse_width"])],
    )
    # tau2 either rides on the model at the value given, or joins the free set of step 2
    step2_free = STEP2_FREE + (("tau2",) if tau2 is None else ())
    constants = {"tau1_rest": tau1} if tau2 is None else {"tau1_rest": tau1, "tau2": tau2}
    watch.lap("step1")
    print(f"Step 1: tau1 = {1e3 * tau1:.1f} ms")

    # --- Step 2: A, Km and tau2 at the reference pulse train --- #
    first = solve(reference_data, step2_free, max_iter=max_iter, constants=constants)
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
            labels=[protocol.condition_name(frequency, pulse_width)
                    for frequency, pulse_width in zip(reference_data["frequency"],
                                                      reference_data["pulse_width"])],
            free=step2_free,
            at_bound=[name for name in first["at_bound"] if name in step2_free],
            converged=first["converged"],
            cost=first["cost"],
            folder=plots.STEP2_ROOT,
            ylabel="Force (N)",
            unit="N",
            route="isometric step 2",
        )

    if eq3 == "ocp":
        return _joint_eq3(subject, trains, identification_index, testing_index, first, carried, constants,
                          watch, estimates, step2_free, save=save, debug=debug, max_iter=max_iter)

    # --- Step 3: A alone, one ocp per remaining identification intensity --- #
    a_points = {protocol.REFERENCE_PULSE_WIDTH_US: first["parameters"]["A"]}
    at_bound = [name for name in first["at_bound"] if name in step2_free]
    steps, groups = {}, []
    # Continuation along the intensity axis: A moves smoothly with the pulse width, so the intensity solved
    # last is a far better start than the table's generic guess, which sits several times off for this muscle
    seed = first["parameters"]["A"]
    for pulse_width_us, indices in other_intensities:
        step = solve(force_ocp.select_trains(trains, indices), STEP3_FREE, max_iter=max_iter,
                     constants=carried, initial_guesses={"A": seed})
        seed = step["parameters"]["A"]
        a_points[pulse_width_us] = step["parameters"]["A"]
        at_bound += [name for name in step["at_bound"] if name in STEP3_FREE]
        steps[pulse_width_us] = step
        groups.append((pulse_width_us, step,
                       [protocol.condition_name(trains["frequency"][i], trains["pulse_width"][i])
                        for i in indices]))
        print(f"  step 3: {pulse_width_us:.0f} us -> A = {step['parameters']['A']:.1f} N/s "
              f"(converged = {step['converged']})")

    if debug:
        plots.plot_step3_fit(groups, subject, folder=plots.STEP3_ROOT, ylabel="Force (N)", unit="N",
                             route="isometric step 3")

    watch.lap("step3")

    # --- Step 4: the force-intensity relationship through those points --- #
    ordered = sorted(a_points.items())
    pulse_widths = np.array([pulse_width for pulse_width, _ in ordered]) * 1e-6
    a_values = np.array([a for _, a in ordered])
    relationship = model.fit_a_pulse_width(pulse_widths, a_values)
    print(f"  step 4: a' = {relationship['a_scale']:.1f} N/s, pd0 = {1e6 * relationship['pd0']:.1f} us, "
          f"pdt = {1e6 * relationship['pdt']:.1f} us (R2 = {relationship['r2']:.4f})")

    if debug:
        plots.plot_step4_relationship(pulse_widths, a_values, relationship, subject,
                                      reference_us=protocol.REFERENCE_PULSE_WIDTH_US,
                                      folder=plots.STEP4_ROOT, route="isometric step 4")

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
    testing = score(force_ocp.select_trains(trains, testing_index), parameters, "testing", constants)
    fitted = score(force_ocp.select_trains(trains, identification_index), parameters, "identification", constants)
    table = pd.concat([fitted["table"], testing["table"]], ignore_index=True)
    watch.lap("step5")
    print(analysis.report_prediction(testing["table"]))
    print(f"          {watch.total():.1f} s total: "
          + ", ".join(f"{key} {value:.1f} s" for key, value in watch.laps.items()))

    if debug:
        plots.plot_step5_prediction(testing["times"], testing["measured"], testing["predicted"],
                                    testing["table"], subject, folder=plots.STEP5_ROOT,
                                    ylabel="Force (N)", unit="N", route="isometric step 5")
        draw(subject, relationship, pulse_widths, a_values, table, parameters, testing, fitted)

    if save:
        limits = {**relationship["bounds"], "tau1_rest": relaxation.TAU1_BOUNDS,
                  "km_rest": IDENTIFIED_PARAMETERS["km_rest"][1:3], "tau2": IDENTIFIED_PARAMETERS["tau2"][1:3]}
        results.save(
            subject=subject,
            method=METHOD,
            parameters=parameters,
            time=np.concatenate(testing["times"]),
            tracked=np.concatenate(testing["measured"]),
            identified=np.concatenate(testing["predicted"]),
            unit="N",
            converged=bool(first["converged"] and all(step["converged"] for step in steps.values())),
            at_bound=sorted(set(at_bound)),
            bounds=(np.array([limits[name][0] for name in parameters]),
                    np.array([limits[name][1] for name in parameters])),
            phase_lengths=[len(piece) for piece in testing["predicted"]],
            extra={
                "route": "isometric", "eq3": "fit",
                "tau2_identified": "tau2" in step2_free,
                "reference_pulse_width_us": protocol.REFERENCE_PULSE_WIDTH_US,
                "tauc": TAUC,
                "r0": R0,
                "identification_conditions": list(fitted["table"]["condition"]),
                "testing_conditions": list(testing["table"]["condition"]),
                "frequency": list(testing["data"]["frequency"]),
                "pulse_width": list(testing["data"]["pulse_width"]),
                "a_points_us": {f"{key:.0f}": value for key, value in a_points.items()},
                "a_pulse_width_r2": relationship["r2"],
                "a_pulse_width_monotonic": relationship["monotonic"],
                "plateau_at_reference": a_points[protocol.REFERENCE_PULSE_WIDTH_US] * carried["tau1_rest"],
                "tau1_estimates": estimates,
                "step_costs": {"reference": first["cost"],
                               **{f"{key:.0f}": step["cost"] for key, step in steps.items()}},
                "trains": table.to_dict(orient="list"),
                "step_seconds": watch.laps,
            },
        )

    return {"subject": subject, "parameters": parameters, "trains": table, "a_points": a_points}


def _joint_eq3(subject, trains, identification_index, testing_index, first, carried, constants,
               watch, estimates, step2_free, save=True, debug=True, max_iter=1000) -> dict:
    """
    Steps 3 and 4 in one solve: a', pd0 and pdt straight off every identification train.

    The staged route identifies one A per intensity and regresses eq. 3 through those six numbers, so the
    per point noise decides the curve. Here eq. 3 is inside the dynamics, `DingModelArticle` scales A by the
    pulse width of each phase, and the three parameters answer to all 13 trains at once. Everything step 2
    settled is held, exactly as step 3 holds it.
    """
    identification = force_ocp.select_trains(trains, identification_index)
    # A is what eq. 3 produces here, so it is not handed over: DingModelArticle scales it from a scale, pd0
    # and pdt at the pulse width of each phase, and does not carry the scalar the single intensity model has
    held = {name: value for name, value in carried.items() if name != "A"}
    step = solve(identification, STEP34_FREE, max_iter=max_iter, model_class=model.DingModelArticle,
                 constants=held)
    watch.lap("step3")
    watch.lap("step4")

    relationship = {name: step["parameters"][name] for name in STEP34_FREE}
    print(f"  steps 3+4 (joint): a' = {relationship['a_scale']:.1f} N/s, "
          f"pd0 = {1e6 * relationship['pd0']:.1f} us, pdt = {1e6 * relationship['pdt']:.1f} us, "
          f"converged = {step['converged']}")

    parameters = {
        "a_scale": relationship["a_scale"],
        "pd0": relationship["pd0"],
        "pdt": relationship["pdt"],
        "tau1_rest": carried["tau1_rest"],
        "tau2": carried["tau2"],
        "km_rest": carried["km_rest"],
    }

    testing = score(force_ocp.select_trains(trains, testing_index), parameters, "testing", constants)
    fitted = score(identification, parameters, "identification", constants)
    watch.lap("step5")
    table = pd.concat([fitted["table"], testing["table"]], ignore_index=True)
    print(analysis.report_prediction(testing["table"]))
    print(f"          {watch.total():.1f} s total: "
          + ", ".join(f"{key} {value:.1f} s" for key, value in watch.laps.items()))

    if debug:
        plots.plot_step5_prediction(testing["times"], testing["measured"], testing["predicted"],
                                    testing["table"], subject, folder=plots.STEP5_ROOT,
                                    ylabel="Force (N)", unit="N", route="isometric step 5")

    if save:
        limits = {name: IDENTIFIED_PARAMETERS[name][1:3] for name in ("a_scale", "pd0", "pdt", "km_rest",
                                                                      "tau2")}
        limits["tau1_rest"] = relaxation.TAU1_BOUNDS
        results.save(
            subject=subject, method=METHOD, parameters=parameters,
            time=np.concatenate(testing["times"]),
            tracked=np.concatenate(testing["measured"]),
            identified=np.concatenate(testing["predicted"]),
            unit="N",
            converged=bool(first["converged"] and step["converged"]),
            at_bound=sorted(set([n for n in first["at_bound"] if n in step2_free]
                                + [n for n in step["at_bound"] if n in STEP34_FREE])),
            bounds=(np.array([limits[name][0] for name in parameters]),
                    np.array([limits[name][1] for name in parameters])),
            phase_lengths=[len(piece) for piece in testing["predicted"]],
            extra={
                "route": "isometric", "eq3": "ocp",
                "tau2_identified": "tau2" in step2_free,
                "reference_pulse_width_us": protocol.REFERENCE_PULSE_WIDTH_US,
                "tauc": TAUC, "r0": R0,
                "identification_conditions": list(fitted["table"]["condition"]),
                "testing_conditions": list(testing["table"]["condition"]),
                "frequency": list(testing["data"]["frequency"]),
                "pulse_width": list(testing["data"]["pulse_width"]),
                "plateau_at_reference": model.a_calculation(
                    1e-6 * protocol.REFERENCE_PULSE_WIDTH_US, relationship["a_scale"],
                    relationship["pd0"], relationship["pdt"]) * carried["tau1_rest"],
                "tau1_estimates": estimates,
                "step_costs": {"reference": first["cost"], "joint": step["cost"]},
                "trains": table.to_dict(orient="list"),
                "step_seconds": watch.laps,
            },
        )

    return {"subject": subject, "parameters": parameters, "trains": table, "a_points": {}}


def usable(ocp_data, subject=None) -> dict:
    """
    Drop the trains that carry no usable force response, the two rejections all_data/force_id.py makes.

    A train whose force never decayed is drift rather than a response, and it is exactly the relaxation
    step 1 reads tau1 from; a train cut before its last pulse asks the model to produce force in a window
    where none was recorded.
    """
    keep = []
    for i in range(len(ocp_data["force"])):
        if not ocp_data.get("force_decayed", [True] * len(ocp_data["force"]))[i]:
            reason = "force never decayed"
        elif not ocp_data.get("covers_stimulation", [True] * len(ocp_data["force"]))[i]:
            reason = "recording stops before the last pulse"
        else:
            keep.append(i)
            continue
        print(f"  {'P' + subject + ' ' if subject else ''}"
              f"{protocol.condition_name(ocp_data['frequency'][i], ocp_data['pulse_width'][i])} dropped: {reason}")

    return force_ocp.select_trains(ocp_data, keep)


def _offsets(ocp_data) -> list:
    """The start time of every train, once they are laid end to end."""
    return list(np.concatenate(([0.0], np.cumsum(ocp_data["final_time"])[:-1])))


def _conditions(ocp_data) -> list:
    """The stimulation condition of every train, for the condition strip of the solution plot."""
    return [
        {
            "t_start": offset,
            "t_end": offset + ocp_data["final_time"][i],
            "pulse_width_us": protocol.as_microseconds(ocp_data["pulse_width"][i]),
            "frequency_hz": ocp_data["frequency"][i],
        }
        for i, offset in enumerate(_offsets(ocp_data))
    ]


def draw(subject, relationship, pulse_widths, a_values, table, parameters, testing, fitted):
    """The per subject figures of one identification."""
    plots.plot_force_intensity(
        pulse_widths, a_values, relationship, title=f"P{subject} force-intensity relationship (eq. 3)"
    )
    for label, scored, name in (
        ("predicted testing trains", testing, "testing_predictions"),
        ("identification trains", fitted, "identification_fit"),
    ):
        plots.plot_predictions(
            # Each panel starts at zero: the stored times are laid end to end over the whole set
            times=[time - time[0] for time in scored["times"]],
            measured=scored["measured"],
            predicted=scored["predicted"],
            conditions=[{"frequency": frequency, "pulse_width": pulse_width}
                        for frequency, pulse_width in zip(scored["data"]["frequency"],
                                                          scored["data"]["pulse_width"])],
            unit="N",
            ylabel="Force (N)",
            title=f"P{subject} {label}",
            name=name,
        )
    plots.plot_agreement(table, unit="N", title=f"P{subject} measured against predicted")

    debug_plots.plot_solution(
        time=np.concatenate(testing["times"]),
        tracked=np.concatenate(testing["measured"]),
        identified=np.concatenate(testing["predicted"]),
        title=f"P{subject} Ding 2007 prediction of the testing trains",
        unit="N",
        ylabel="Force (N)",
        phase_lengths=[len(piece) for piece in testing["predicted"]],
        conditions=_conditions(testing["data"]),
        parameters=parameters,
        name="ding_force_prediction",
    )


def main(subjects=range(1, 21), data_folder="force", save=True, debug=True, show_debug=False,
         max_iter=5000, tau2=model.TAU2_LITERATURE, eq3="fit"):
    """Run the five steps on a list of subjects, then print the cohort table."""
    for i in subjects:
        subject = f"{int(i):02d}"
        try:
            identify(subject, data_folder=data_folder, save=save, debug=debug,
                     show_debug=show_debug, max_iter=max_iter, tau2=tau2, eq3=eq3)
        except Exception as error:  # one bad subject must not take the rest of the cohort down
            print(f"P{subject}: {type(error).__name__}: {error}")

    print()
    print(results.summary_table(METHOD))


if __name__ == "__main__":
    main()
