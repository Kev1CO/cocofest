"""
The identification procedure, run the same way for every route.

    step 1  tau1 alone, off the relaxation of the reference intensity phases, with no solver. A route
            whose observable holds no relaxation says so with TAU1: "identified" puts tau1 in the free set
            of step 2, a number holds it there.
    step 2  the same phases, tracking A and Km with tau1 held where step 1 put it
    step 3  one program per remaining identification intensity, with A the only free parameter
    step 4  the (pulse width, A) points are fitted with eq. 3, giving a', pd0 and pdt
    step 5  the whole parameter set is run over the testing phases with nothing left free, and scored
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from bioptim import Solver

import store

import ding_model
import metrics
import observables
import ocp as programs
import plots
import protocol
import relaxation
import summary
from helper import debug_plots, results


R0 = ding_model.R0  # Value from Kesar et al. (2008)
TAUC = ding_model.TAUC  # Value in (s) from Kesar et al. (2008)

IDENTIFIED_PARAMETERS = {
    "A": (900, 1, None, 1000.0),
    "tau1_rest": (0.06, 5e-3, 0.5, 0.05),
    "km_rest": (0.18, 1e-3, 1.0, 0.1),
    "tau2": (0.0185, 1e-3, 0.2, 0.05),
    "a_scale": (2000, 1, None, 1000.0),
    "pd0": (1.31405e-4, ding_model.PD0_FLOOR, 2.4e-4, 1e-4),
    "pdt": (1.94138e-4, 1e-5, 1e-3, 1e-4),
}

STEP2_FREE = ("A", "km_rest")
STEP3_FREE = ("A",)
STEP34_FREE = ("a_scale", "pd0", "pdt")

DESCENT_TOLERANCE = 0.02


# ---- Preparing the phases of one step ---- #
def select(phases, keep) -> list:
    """
    The phases of one step, with the tracking weights rebalanced over that selection.
    """
    selected = [dict(phases[i]) for i in keep]
    longest = max(phase["n_shooting"] for phase in selected)
    offset = 0.0
    for phase in selected:
        phase["weight"] = longest / phase["n_shooting"] / phase["cost_scale"] ** 2
        phase["time_offset"] = offset
        offset += phase["final_time"]
    return selected


def scale_of(phases) -> float:
    """What the observable of one subject is worth, taken over every phase, never zero."""
    peak = max(float(np.max(np.abs(phase["target"] - phase["target"][0]))) for phase in phases)
    return peak if peak > 0 else 1.0


def initial_a(route, phases) -> float:
    """
    Where A starts before step 2 identifies it, read off the measurement rather than off the table.
    """
    if route.COST == "angle":
        return None

    peaks = []
    for phase in phases:
        target = np.asarray(phase["target"], dtype=float)
        at = int(np.argmax(target))
        peaks.append(target[at] / abs(phase["moment_arm"][at]) if route.COST == "torque" else target[at])
    return float(np.mean(peaks))


def in_unit(route, values) -> np.ndarray:
    """The observable in the unit the route reports it in, ex: the angle state in rad read out in degrees."""
    return route.SCALE * np.asarray(values, dtype=float)


def excursion_of(route, phase) -> np.ndarray:
    """
    What the stimulation added to the resting level of the observable, the signal step 1 relaxes on.
    """
    target = np.asarray(phase["target"], dtype=float)
    return in_unit(route, target - target[0] if route.COST == "angle" else target)


def relaxation_window(route, phase) -> np.ndarray:
    """
    The part of one phase step 1 reads tau1 off: the contraction, and the first descent after it.
    """
    signal = excursion_of(route, phase)
    peak = int(np.argmax(signal))
    entered = np.where(signal[peak:] <= relaxation.TAU1_WINDOW[0] * signal[peak])[0]
    if entered.size == 0:
        return np.ones(len(signal), dtype=bool)

    start = peak + int(entered[0])
    tolerance = DESCENT_TOLERANCE * float(signal[peak])
    end, floor = len(signal), signal[start]
    for i in range(start + 1, len(signal)):
        floor = min(floor, signal[i])
        if signal[i] > floor + tolerance:
            end = i
            break

    window = np.zeros(len(signal), dtype=bool)
    window[:end] = True
    return window


# ---- The bounds that depend on what an earlier step settled ---- #
def carried_tau1(held_tau1, table) -> float:
    """The tau1 the initial guess of A divides by: the one step 1 measured, or the one step 2 starts from."""
    return float(held_tau1) if held_tau1 is not None else float(table["tau1_rest"][0])


def parameter_table(tau1) -> dict:
    """
    The parameter settings, with the two scaling factors bounded by the force the muscle can hold.
    """
    ceiling = programs.FMAX / max(float(tau1), 1e-3)
    table = dict(IDENTIFIED_PARAMETERS)
    for name in ("A", "a_scale"):
        guess, minimum, _, scaling = table[name]
        table[name] = (min(guess, 0.5 * ceiling), minimum, ceiling, scaling)
    return table


def eq3_limits(table) -> dict:
    """The (min, max) of a', pd0 and pdt, as step 4's fit takes them, so both routes to eq. 3 share them."""
    return {name: table[name][1:3] for name in STEP34_FREE}


def roles_of(phases, reference, identification, testing) -> dict:
    """What every recorded stimulation condition was used for, keyed by (frequency, pulse width in us)."""
    roles = {}
    for i, phase in enumerate(phases):
        key = (protocol.nearest(float(phase["frequency"]), protocol.FREQUENCIES),
               protocol.nearest(protocol.as_microseconds(phase["pulse_width"]), protocol.PULSE_WIDTHS_US))
        roles[key] = ("reference" if i in reference else
                      "testing" if i in testing else
                      "identification" if i in identification else "dropped")
    return roles


def at_bound_of(values, limits, tolerance=1e-3) -> list:
    """
    The names sitting on one of their bounds, the check results.parameters_at_bound makes on an ocp.
    """
    on = []
    for name, (minimum, maximum) in limits.items():
        margin = tolerance * (maximum - minimum) if maximum > minimum else tolerance
        if values[name] - minimum <= margin or maximum - values[name] <= margin:
            on.append(name)
    return on


# ---- Solving one step ---- #
MAX_ITER = 1000


def ipopt(max_iter):
    """
    The solver every step of every route is given, so none of them can drift from the others.

    ma57 where it exists and mumps elsewhere: the HSL solvers are not part of the Windows environment this
    study is written on, so asking for ma57 there fails at the first solve.
    """
    solver = Solver.IPOPT(_max_iter=min(int(max_iter), MAX_ITER), _constr_viol_tol=1e-8,
                          _compl_inf_tol=1e-8)
    solver.set_linear_solver("ma57" if sys.platform.startswith("linux") else "mumps")
    solver.__dict__["_acceptable_iter"] = 0
    solver.__dict__["_bound_relax_factor"] = 0.0
    return solver


def solver_status(program) -> str:
    """IPOPT's own exit string, which sol.status cannot give: casadi calls both of its successes success."""
    solver = getattr(getattr(program, "ocp_solver", None), "shaked_ocp_solver", None)
    return str(solver.stats().get("return_status", "unknown")) if solver is not None else "unknown"


def solve(route, subject, phases, free, model_class, table, constants=None, initial_guesses=None,
          max_iter=MAX_ITER) -> dict:
    """Build and solve one step, and return what it identified, with the fit it identified it on."""
    program = programs.prepare_ocp(
        subject,
        phases,
        free,
        model_class,
        table,
        constants=constants,
        initial_guesses=initial_guesses,
        kinematics=route.KINEMATICS,
        cost=route.COST,
        passive=route.PASSIVE_TORQUE,
    )
    sol = program.solve(solver=ipopt(max_iter))

    n_shooting = [phase["n_shooting"] for phase in phases]
    identified = _produced(route, sol, phases, n_shooting)
    parameters = {key: float(value.squeeze()) for key, value in sol.decision_parameters().items()}
    return {
        "parameters": parameters,
        "times": [phase["time"] for phase in phases],
        "offsets": [phase["time_offset"] for phase in phases],
        "targets": [np.asarray(phase["target"], dtype=float) for phase in phases],
        "identified": identified,
        "at_bound": results.parameters_at_bound(program, parameters),
        "cost": float(np.sum(sol.cost)),
        "converged": bool(sol.status == 0),
        "status": solver_status(program),
        "iterations": int(getattr(sol, "iterations", -1) or -1),
        "seconds": float(getattr(sol, "real_time_to_optimize", np.nan) or np.nan),
        "bounds": {name: (float(program.parameter_bounds[name].min.min()),
                          float(program.parameter_bounds[name].max.max()))
                   for name in program.parameter_bounds.keys()},
    }


def diagnostics_of(first, steps, prediction, step2_free) -> dict:
    """What the solver did at every step, the block both the saved result and the summary sheet read."""
    return {
        "step2": {"free": list(step2_free), "converged": first["converged"], "status": first["status"],
                  "iterations": first["iterations"], "cost": first["cost"], "seconds": first["seconds"],
                  "at_bound": list(first["at_bound"]), "bounds": first["bounds"]},
        **{f"step3_{key}": {"free": list(STEP3_FREE if key != "joint" else STEP34_FREE),
                            "converged": step["converged"], "status": step["status"],
                            "iterations": step["iterations"],
                            "cost": step["cost"], "seconds": step["seconds"],
                            "at_bound": list(step["at_bound"]), "bounds": step["bounds"]}
           for key, step in steps.items()},
        "step5": {"converged": prediction["converged"], "status": prediction["status"],
                  "cost": prediction["cost"],
                  "iterations": prediction["iterations"], "seconds": prediction["seconds"]},
    }


def _produced(route, sol, phases, n_shooting) -> list:
    """The quantity the cost compared, as the identified model produced it, one array per phase."""
    key = {"angle": "q", "torque": "F", "force": "F"}[route.COST]
    if route.KINEMATICS != "none" and key == "F":
        key = "F_" + programs.MUSCLE_NAME

    _, solved = results.solution_at_nodes(sol, key, n_shooting)
    starts = np.concatenate(([0], np.cumsum([intervals + 1 for intervals in n_shooting])))
    pieces = [solved[starts[i]:starts[i + 1]] for i in range(len(n_shooting))]
    if route.COST != "torque":
        return pieces
    # The muscle joint torque, the very quantity the custom objective compared, read back on the nodes
    return [-phase["moment_arm"] * piece for phase, piece in zip(phases, pieces)]


def score(route, step, phases, set_name) -> pd.DataFrame:
    """
    Score every phase of one step the way the article scores its testing trains, in the route's own unit.
    """
    rows = []
    for i, phase in enumerate(phases):
        measured = in_unit(route, step["targets"][i])
        rows.append(
            {
                "frequency": phase["frequency"],
                "pulse_width_us": protocol.as_microseconds(phase["pulse_width"]),
                "condition": protocol.condition_name(phase["frequency"], phase["pulse_width"]),
                "set": set_name,
                **metrics.score_train(step["times"][i], measured,
                                       in_unit(route, step["identified"][i]),
                                       baseline=float(measured[0]) if route.COST == "angle" else 0.0),
            }
        )
    return pd.DataFrame(rows)


# ---- The five steps on one subject ---- #
def step2_context(route, subject, design=None, tau2=None, initial_guesses=None, debug=False,
                  folders=None, **load_kwargs) -> dict:
    """
    Everything the procedure settles before step 2 solves anything, in one place.

    Returns
    -------
    dict
        The phases and how they split, the tau1 step 1 read, the free set and the constants of step 2, the
        parameter table, the initial guesses, and the stopwatch already lapped on step 1
    """
    folders = folders or store.step_folders(route.METHOD)

    phases = route.load(subject, debug=debug, **load_kwargs)
    scale = scale_of(phases)
    for phase in phases:
        phase["cost_scale"] = scale

    conditions = observables.conditions_of(phases)
    reference, other_intensities = protocol.steps(*conditions, design=design)
    identification_index, testing_index = protocol.split(*conditions, design=design)
    protocol.report_split(*conditions, label=f"P{subject}: ", design=design)
    labels = observables.labels_of(phases)

    watch = metrics.Stopwatch()

    # --- Step 1: tau1 off the relaxations, with no solver --- #
    reference_phases = select(phases, reference)
    tau1, estimates = np.nan, []
    if route.TAU1 == "relaxation":
        windows = [relaxation_window(route, phase) for phase in reference_phases]
        tau1, estimates = relaxation.mean_tau1(
            [phase["time"][window] for phase, window in zip(reference_phases, windows)],
            [excursion_of(route, phase)[window] for phase, window in zip(reference_phases, windows)],
            debug=debug,
            subject=subject,
            labels=[labels[i] for i in reference],
            folder=folders["step1"],
            ylabel=route.YLABEL,
            unit=route.UNIT,
            route=f"{route.METHOD} step 1",
        )
        if not np.isfinite(tau1):
            raise ValueError(f"P{subject} {route.METHOD}: no reference phase relaxed far enough to read tau1 "
                             f"off, the estimates were {estimates}")
        print(f"Step 1: tau1 = {1e3 * tau1:.1f} ms")
    else:
        print(f"Step 1: skipped, tau1 {'joins step 2' if route.TAU1 == 'identified' else f'held at {route.TAU1}'}")

    # tau1 and tau2 either ride on the model at the value given, or join the free set of step 2
    held_tau1 = tau1 if route.TAU1 == "relaxation" else (None if route.TAU1 == "identified" else float(route.TAU1))
    step2_free = (STEP2_FREE + (("tau1_rest",) if held_tau1 is None else ())
                  + (("tau2",) if tau2 is None else ()))
    constants = {name: value for name, value in (("tau1_rest", held_tau1), ("tau2", tau2)) if value is not None}
    watch.lap("step1")

    table = parameter_table(held_tau1 if held_tau1 is not None else IDENTIFIED_PARAMETERS["tau1_rest"][0])
    plateau = initial_a(route, reference_phases)
    guesses = {} if plateau is None else {"A": plateau / max(carried_tau1(held_tau1, table), 1e-3)}
    guesses.update(initial_guesses or {})

    return {"phases": phases, "scale": scale, "conditions": conditions, "labels": labels,
            "reference": reference, "other_intensities": other_intensities,
            "identification_index": identification_index, "testing_index": testing_index,
            "reference_phases": reference_phases, "tau1": tau1, "estimates": estimates,
            "held_tau1": held_tau1, "step2_free": step2_free, "constants": constants,
            "table": table, "guesses": guesses, "watch": watch, "folders": folders}


def identify(route, subject, save=True, debug=True, show_debug=False, max_iter=MAX_ITER,
             tau2=None, eq3="fit", design=None, initial_guesses=None, **load_kwargs) -> dict:
    """
    The whole algorithm on one subject, for one route.
    """
    debug_plots.output_for(route.METHOD, subject, debug=debug, show=show_debug)
    plots.output_for(route.METHOD, subject, debug=debug, show=show_debug)
    folders = store.step_folders(route.METHOD)
    print(f"=== P{subject} {route.METHOD} ===")

    # --- Data preparation, step 1, and everything step 2 needs --- #
    context = step2_context(route, subject, design=design, tau2=tau2, initial_guesses=initial_guesses,
                            debug=debug, folders=folders, **load_kwargs)
    phases, scale, labels = context["phases"], context["scale"], context["labels"]
    reference, other_intensities = context["reference"], context["other_intensities"]
    identification_index, testing_index = context["identification_index"], context["testing_index"]
    reference_phases, tau1, estimates = context["reference_phases"], context["tau1"], context["estimates"]
    held_tau1, step2_free, constants = context["held_tau1"], context["step2_free"], context["constants"]
    table, guesses, watch = context["table"], context["guesses"], context["watch"]

    # --- Step 2: A and Km at the reference pulse width --- #
    first = solve(route, subject, reference_phases, step2_free, ding_model.DingModelSingleIntensity, table,
                  constants=constants, initial_guesses=guesses, max_iter=max_iter)
    watch.lap("step2")
    # Everything step 2 settled, handed on as it stands, and the plateau ceiling re-read off its tau1
    carried = {**constants, **first["parameters"]}
    settled = parameter_table(carried["tau1_rest"])
    print(f"Step 2: A = {first['parameters']['A']:.1f} N/s, Km = {carried['km_rest']:.4f}, "
          f"tau1 = {1e3 * carried['tau1_rest']:.1f} ms, tau2 = {1e3 * carried['tau2']:.1f} ms, "
          f"converged = {first['converged']} ({first['status']})")

    if debug:
        plots.plot_step2_fit(
            times=first["times"], tracked=first["targets"], identified=first["identified"],
            parameters=first["parameters"], subject=subject, labels=[labels[i] for i in reference],
            free=step2_free, at_bound=[name for name in first["at_bound"] if name in step2_free],
            converged=first["converged"], cost=first["cost"], folder=folders["step2"],
            ylabel=route.YLABEL, unit=route.UNIT, scale=route.SCALE, route=f"{route.METHOD} step 2",
        )

    # --- Step 3: A alone, one program per remaining identification intensity --- #
    a_points = {protocol.REFERENCE_PULSE_WIDTH_US: first["parameters"]["A"]}
    at_bound = [name for name in first["at_bound"] if name in step2_free]
    steps, groups = {}, []

    if eq3 == "ocp":
        # eq. 3 inside the dynamics: a', pd0 and pdt answer to every identification phase at once, and A is
        # not handed over since DingModelArticle derives it
        without_a = {name: value for name, value in carried.items() if name != "A"}
        joint = solve(route, subject, select(phases, identification_index), STEP34_FREE,
                      ding_model.DingModelArticle, settled, constants=without_a, max_iter=max_iter)
        relationship = {name: joint["parameters"][name] for name in STEP34_FREE}
        relationship["r2"], relationship["monotonic"] = np.nan, True
        relationship["bounds"] = eq3_limits(settled)
        at_bound += [name for name in joint["at_bound"] if name in STEP34_FREE]
        steps["joint"] = joint
        print(f"  steps 3+4 (ocp): a' = {relationship['a_scale']:.1f} N/s, "
              f"pd0 = {1e6 * relationship['pd0']:.1f} us, pdt = {1e6 * relationship['pdt']:.1f} us, "
              f"converged = {joint['converged']} ({joint['status']})")

    # Continuation along the intensity axis: the intensity solved last is a far better start than the table
    seed = first["parameters"]["A"]
    for pulse_width_us, indices in (other_intensities if eq3 != "ocp" else []):
        step = solve(route, subject, select(phases, indices), STEP3_FREE, ding_model.DingModelSingleIntensity,
                     settled, constants=carried, initial_guesses={"A": seed}, max_iter=max_iter)
        seed = step["parameters"]["A"]
        a_points[pulse_width_us] = step["parameters"]["A"]
        at_bound += [name for name in step["at_bound"] if name in STEP3_FREE]
        steps[pulse_width_us] = step
        groups.append((pulse_width_us, step, [labels[i] for i in indices]))
        print(f"  step 3: {pulse_width_us:.0f} us -> A = {step['parameters']['A']:.1f} N/s "
              f"({step['iterations']} iterations, converged = {step['converged']}, {step['status']})")

    if debug and groups:
        plots.plot_step3_fit(groups, subject, folder=folders["step3"], ylabel=route.YLABEL,
                             unit=route.UNIT, scale=route.SCALE, route=f"{route.METHOD} step 3")
    watch.lap("step3")

    # --- Step 4: the force-intensity relationship through those points --- #
    ordered = sorted(a_points.items())
    pulse_widths = np.array([pulse_width for pulse_width, _ in ordered]) * 1e-6
    a_values = np.array([a for _, a in ordered])
    if eq3 != "ocp":
        relationship = ding_model.fit_a_pulse_width(pulse_widths, a_values, limits=eq3_limits(settled))
        # A least squares fit, so results.parameters_at_bound cannot see it. a' floored at the largest A
        # observed is eq. 3 approaching its asymptote from below, an identity rather than a bound, so it is
        # reported on its own flag instead of burying the real bounds.
        on_bound = at_bound_of(relationship, relationship["bounds"])
        saturated = "a_scale" in on_bound and bool(
            np.isclose(relationship["a_scale"], relationship["bounds"]["a_scale"][0], rtol=1e-3))
        at_bound += [name for name in on_bound if not (name == "a_scale" and saturated)]
    print(f"  step 4: a' = {relationship['a_scale']:.1f} N/s, pd0 = {1e6 * relationship['pd0']:.1f} us, "
          f"pdt = {1e6 * relationship['pdt']:.1f} us (R2 = {relationship['r2']:.4f})")

    if debug and eq3 != "ocp":
        plots.plot_step4_relationship(pulse_widths, a_values, relationship, subject,
                                      reference_us=protocol.REFERENCE_PULSE_WIDTH_US,
                                      folder=folders["step4"], route=f"{route.METHOD} step 4")

    parameters = {
        "a_scale": relationship["a_scale"],
        "pd0": relationship["pd0"],
        "pdt": relationship["pdt"],
        "tau1_rest": carried["tau1_rest"],
        "tau2": carried["tau2"],
        "km_rest": carried["km_rest"],
    }
    watch.lap("step4")

    # --- Step 5: prediction, the same program with nothing left free --- #
    testing_phases = select(phases, testing_index)
    identification_phases = select(phases, identification_index)
    print(f"  step 5: running the {len(testing_phases)} testing phase(s) with every parameter fixed")
    prediction = solve(route, subject, testing_phases, (), ding_model.DingModelArticle, settled,
                       constants=parameters, max_iter=max_iter)
    refit = solve(route, subject, identification_phases, (), ding_model.DingModelArticle, settled,
                  constants=parameters, max_iter=max_iter)

    testing_table = score(route, prediction, testing_phases, "testing")
    identification_table = score(route, refit, identification_phases, "identification")
    table = pd.concat([identification_table, testing_table], ignore_index=True)
    watch.lap("step5")
    print(metrics.report_prediction(testing_table))
    print(f"          {watch.total():.1f} s total: "
          + ", ".join(f"{key} {value:.1f} s" for key, value in watch.laps.items()))

    step_diagnostics = diagnostics_of(first, steps, prediction, step2_free)
    every_step_converged = bool(first["converged"] and all(step["converged"] for step in steps.values()))

    # What the figures, the saved result and the return value all read, derived once so they cannot differ
    measured = np.concatenate(prediction["targets"])
    measured = in_unit(route, np.concatenate(prediction["targets"]))
    produced = in_unit(route, np.concatenate(prediction["identified"]))
    rmse = float(results.rmse(measured, produced))
    on_bound = sorted(set(at_bound))
    plateau = a_points[protocol.REFERENCE_PULSE_WIDTH_US] * carried["tau1_rest"]
    phase_lengths = [len(piece) for piece in prediction["identified"]]

    if debug:
        plots.plot_step5_prediction(prediction["times"], prediction["targets"], prediction["identified"],
                                    testing_table, subject, folder=folders["step5"], ylabel=route.YLABEL,
                                    unit=route.UNIT, scale=route.SCALE, route=f"{route.METHOD} step 5")
        draw(route, subject, relationship, pulse_widths, a_values, table, parameters,
             (testing_phases, prediction), (identification_phases, refit))
        summary.plot_subject(route, subject, folder=debug_plots.DEBUG_ROOT / route.METHOD / f"P{subject}",
                             story={
            "observable": route.OBSERVABLE,
            "kinematics": route.KINEMATICS,
            "conditions": roles_of(phases, reference, identification_index, testing_index),
            "tau2_identified": "tau2" in step2_free,
            "passive_torque": route.PASSIVE_TORQUE,
            "tau1_policy": route.TAU1,
            "tau1": tau1,
            "relaxations": [(phase["time"][window], excursion_of(route, phase)[window], estimate)
                            for phase, window, estimate in
                            zip(reference_phases,
                                [relaxation_window(route, p) for p in reference_phases],
                                estimates or [np.nan] * len(reference_phases))],
            "step2": first,
            "step2_free": step2_free,
            "reference_labels": [labels[i] for i in reference],
            "testing_labels": [labels[i] for i in testing_index],
            "pulse_widths": pulse_widths,
            "a_values": a_values,
            "relationship": relationship,
            "prediction": prediction,
            "table": table,
            "parameters": parameters,
            "plateau": plateau,
            "rmse": rmse,
            "converged": every_step_converged,
            "at_bound": on_bound,
            "diagnostics": step_diagnostics,
            "seconds_per_step": watch.laps,
            "seconds": watch.total(),
        })

    if save:
        # nan for a parameter this route never optimised, so a saved result tells a measured bound apart
        # from a value that was carried over
        limits = {name: (np.nan, np.nan) for name in parameters}
        limits.update(relationship["bounds"])
        limits["tau1_rest"] = settled["tau1_rest"][1:3]
        limits.update({name: first["bounds"][name] for name in first["bounds"]})
        results.save(
            subject=subject,
            method=route.METHOD,
            parameters=parameters,
            time=np.concatenate([prediction["offsets"][i] + prediction["times"][i]
                                 for i in range(len(testing_phases))]),
            tracked=measured,
            identified=produced,
            unit=route.UNIT,
            converged=every_step_converged,
            at_bound=on_bound,
            bounds=(np.array([limits[name][0] for name in parameters]),
                    np.array([limits[name][1] for name in parameters])),
            phase_lengths=phase_lengths,
            extra={
                "route": route.METHOD, "eq3": eq3,
                "observable": route.OBSERVABLE,
                "kinematics": route.KINEMATICS,
                "cost": route.COST,
                "passive_torque": route.PASSIVE_TORQUE,
                "tau1_identified": "tau1_rest" in step2_free,
                "tau2_identified": "tau2" in step2_free,
                "reference_pulse_width_us": protocol.REFERENCE_PULSE_WIDTH_US,
                "tauc": TAUC, "r0": R0,
                "tau1_from_relaxation": route.TAU1 == "relaxation",
                "tau1_estimates": estimates,
                "identification_conditions": list(identification_table["condition"]),
                "testing_conditions": list(testing_table["condition"]),
                "frequency": [phase["frequency"] for phase in testing_phases],
                "pulse_width": [phase["pulse_width"] for phase in testing_phases],
                "a_points_us": {f"{key:.0f}": value for key, value in a_points.items()},
                "a_pulse_width_r2": relationship["r2"],
                "eq3_saturated": saturated,
                "a_pulse_width_monotonic": relationship["monotonic"],
                # The product every route can actually measure, whatever unit it identified in
                "plateau_at_reference": plateau,
                "cost_scale": scale,
                "step_costs": {"reference": first["cost"],
                               **{f"{key}": step["cost"] for key, step in steps.items()}},
                "step_diagnostics": step_diagnostics,
                "fixed_parameters": {"tauc": TAUC, "r0": R0,
                                     **({} if "tau2" in step2_free else {"tau2": float(carried["tau2"])})},
                "tau1_policy": route.TAU1,
                "trains": table.to_dict(orient="list"),
                "step_seconds": watch.laps,
            },
        )

    return {"subject": subject, "parameters": parameters, "trains": table, "a_points": a_points,
            "relationship": relationship, "at_bound": on_bound,
            "converged": every_step_converged, "diagnostics": step_diagnostics,
            "seconds": watch.total(), "seconds_per_step": dict(watch.laps), "rmse": rmse,
            "step2_cost": float(first["cost"]), "step2_parameters": dict(first["parameters"]),
            "tau1": float(tau1) if np.isfinite(tau1) else np.nan}


def draw(route, subject, relationship, pulse_widths, a_values, table, parameters, testing, identification):
    """The per subject figures of one identification."""
    plots.plot_force_intensity(
        pulse_widths, a_values, relationship, title=f"P{subject} force-intensity relationship (eq. 3)"
    )
    for label, (group, step), name in (
        ("predicted testing phases", testing, "testing_predictions"),
        ("identification phases", identification, "identification_fit"),
    ):
        plots.plot_predictions(
            times=step["times"], measured=step["targets"], predicted=step["identified"],
            conditions=[{"frequency": phase["frequency"], "pulse_width": phase["pulse_width"]}
                        for phase in group],
            unit=route.UNIT, ylabel=route.YLABEL, title=f"P{subject} {label}", scale=route.SCALE,
            baselines=[float(target[0]) for target in step["targets"]], name=name,
        )

    plots.plot_agreement(table, unit=route.UNIT, title=f"P{subject} measured against predicted")

    group, step = testing
    debug_plots.plot_solution(
        time=np.concatenate([step["offsets"][i] + step["times"][i] for i in range(len(group))]),
        tracked=np.concatenate(step["targets"]),
        identified=np.concatenate(step["identified"]),
        title=f"P{subject} {route.METHOD} prediction of the testing phases",
        unit=route.UNIT, ylabel=route.YLABEL, scale=route.SCALE,
        phase_lengths=[len(piece) for piece in step["identified"]],
        conditions=[
            {
                "t_start": float(step["offsets"][i]),
                "t_end": float(step["offsets"][i] + group[i]["final_time"]),
                "pulse_width_us": protocol.as_microseconds(group[i]["pulse_width"]),
                "frequency_hz": group[i]["frequency"],
            }
            for i in range(len(group))
        ],
        parameters=parameters,
        name=f"{route.METHOD}_prediction",
    )


def main(route, subjects=range(1, 21), save=True, debug=True, show_debug=False, max_iter=MAX_ITER,
         tau2=None, eq3="fit", **load_kwargs):
    """Run the five steps of one route on a list of subjects, then print the cohort table."""
    for i in subjects:
        subject = f"{int(i):02d}"
        try:
            identify(route, subject, save=save, debug=debug, show_debug=show_debug, max_iter=max_iter,
                     tau2=tau2, eq3=eq3, **load_kwargs)
        except Exception as error:  # one bad subject must not take the rest of the cohort down
            print(f"P{subject}: {type(error).__name__}: {error}")


    print()
    print(results.summary_table(route.METHOD))
