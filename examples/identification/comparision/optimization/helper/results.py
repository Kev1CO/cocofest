"""
Storage of the identification results, shared by the three optimization programs.
Every run is saved as one npz per subject under results/<method>/p<subject>_<method>.npz.

Contents of a result
--------------------
subject             str    the subject id, ex: "01"
method              str    "force_id_all", "movement_id_single", "passive_torque_id_all_riener", ...
parameter_names     (p,)   names of the identified parameters
parameter_values    (p,)   their identified value
parameter_min       (p,)   the bound the identification was given
parameter_max       (p,)
parameters_at_bound (b,)   names of those that landed on a bound, i.e. that the data did not constrain
time                (n,)   time vector of the solution (s), at the shooting nodes, see solution_at_nodes
tracked             (n,)   the experimental data that was tracked
identified          (n,)   the same quantity produced by the identified model
unit                str    unit of tracked/identified, "N" or "rad"
rmse                float  in `unit`
phase_lengths       (k,)   samples per phase, to split time/tracked/identified back into phases. Sums to n.
converged           bool   whether the solver reported an optimal solution
solver_status       int    the solver's own status code, 0 being an optimal solution
solver_iterations   int
solver_time         float  seconds spent inside the solver
real_time           float  seconds of wall clock for the whole solve, including what bioptim does around it
n_phases            int    number of phases of the program
n_parameters        int    number of identified parameters
extra               str    json, everything specific to one method (stimulation conditions, held out phase, ...)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bioptim import SolutionMerge

RESULTS_ROOT = Path(__file__).resolve().parent.parent.parent / "results"
SUFFIX = ".npz"


def rmse(tracked, identified) -> float:
    """Root mean square error between the tracked and the identified signal."""
    tracked, identified = np.asarray(tracked), np.asarray(identified)
    if tracked.shape != identified.shape:
        raise ValueError(
            f"The tracked signal holds {tracked.shape} samples and the identified one {identified.shape}. "
            f"They have to be read on the same grid, see solution_at_nodes."
        )
    return float(np.sqrt(np.mean((identified - tracked) ** 2)))


def solution_at_nodes(sol, key: str, n_shooting) -> tuple:
    """
    The time vector and one state of a solution, read at the shooting nodes only.

    Parameters
    ----------
    sol: Solution
        The solved bioptim solution
    key: str
        The state to read, ex: "q"
    n_shooting: list[int]
        The number of shooting intervals of every phase, in order

    Returns
    -------
    tuple
        The time vector and the state, both sampled at the shooting nodes and continuous over the phases
    """
    times = sol.decision_time(to_merge=[SolutionMerge.NODES])
    states = sol.decision_states(to_merge=[SolutionMerge.NODES])
    # A single phase ocp returns the arrays directly rather than a list holding one entry
    if not isinstance(times, (list, tuple)):
        times, states = [times], [states]

    node_times, node_states = [], []
    for phase, intervals in enumerate(n_shooting):
        time = np.asarray(times[phase]).squeeze()
        state = np.asarray(states[phase][key]).squeeze()
        stride = (len(time) - 1) // intervals
        if stride * intervals + 1 != len(time):
            raise ValueError(
                f"Phase {phase} holds {len(time)} decision points for {intervals} shooting intervals, which is "
                f"not a whole number of points per interval."
            )
        node_times.append(time[::stride])
        node_states.append(state[::stride])

    return np.concatenate(node_times), np.concatenate(node_states)


def parameters_at_bound(ocp, parameters: dict, tolerance: float = 1e-3) -> list:
    """
    The names of the identified parameters that landed on one of their bounds, which is the usual sign that the
    data does not constrain them.

    Parameters
    ----------
    tolerance: float
        How close to a bound counts as being on it, as a fraction of that parameter's own range.
    """
    at_bound = []
    for name, value in parameters.items():
        bounds = ocp.parameter_bounds[name]
        minimum, maximum = float(bounds.min.min()), float(bounds.max.max())
        margin = tolerance * (maximum - minimum) if maximum > minimum else tolerance
        if value - minimum <= margin or maximum - value <= margin:
            at_bound.append(name)
    return at_bound


def parameter_bounds(ocp, names) -> tuple:
    """
    The min and max the identification was given for each parameter, in the same order as `names`.

    Names the ocp never optimized get nan, which is how a saved result tells apart a parameter that was
    identified from one that was carried over at a fixed value.
    """
    optimized = set(ocp.parameter_bounds.keys())
    minimum, maximum = [], []
    for name in names:
        if name in optimized:
            bounds = ocp.parameter_bounds[name]
            minimum.append(float(bounds.min.min()))
            maximum.append(float(bounds.max.max()))
        else:
            minimum.append(np.nan)
            maximum.append(np.nan)
    return np.array(minimum), np.array(maximum)


def path_of(subject: str, method: str) -> Path:
    """Where the result of one subject for one method lives."""
    return RESULTS_ROOT / method / f"p{subject}_{method}{SUFFIX}"


def save(
    subject: str,
    method: str,
    parameters: dict,
    time,
    tracked,
    identified,
    unit: str,
    sol=None,
    converged: bool = None,
    at_bound: list = None,
    bounds: tuple = None,
    phase_lengths=None,
    extra: dict = None,
) -> Path:
    """
    Save one identification result. See the module docstring for the contents.

    Parameters
    ----------
    sol: Solution
        The bioptim solution. Everything about the solve, its status, its iteration count and the two timings,
        is read from it, so a run can be judged on its cost as well as on its error.
    converged: bool
        Only needed when there is no `sol` to read it from.

    Returns
    -------
    Path
        The path the result was written to
    """
    path = path_of(subject, method)
    path.parent.mkdir(parents=True, exist_ok=True)

    names = list(parameters)
    minimum, maximum = bounds if bounds else (np.full(len(names), np.nan), np.full(len(names), np.nan))
    lengths = np.asarray(phase_lengths if phase_lengths is not None else [], dtype=int)

    np.savez_compressed(
        path,
        subject=subject,
        method=method,
        parameter_names=np.array(names, dtype="U32"),
        parameter_values=np.array([float(parameters[name]) for name in names]),
        parameter_min=np.asarray(minimum, dtype=float),
        parameter_max=np.asarray(maximum, dtype=float),
        parameters_at_bound=np.array(at_bound if at_bound else [], dtype="U32"),
        time=np.asarray(time, dtype=float),
        tracked=np.asarray(tracked, dtype=float),
        identified=np.asarray(identified, dtype=float),
        unit=unit,
        rmse=rmse(tracked, identified),
        phase_lengths=lengths,
        converged=bool(sol.status == 0 if sol is not None else converged),
        solver_status=int(getattr(sol, "status", -1)) if sol is not None else -1,
        solver_iterations=int(getattr(sol, "iterations", -1) or -1) if sol is not None else -1,
        solver_time=float(getattr(sol, "solver_time_to_optimize", np.nan) or np.nan) if sol is not None else np.nan,
        real_time=float(getattr(sol, "real_time_to_optimize", np.nan) or np.nan) if sol is not None else np.nan,
        n_phases=int(len(lengths)),
        n_parameters=int(len(names)),
        extra=json.dumps(extra if extra else {}, default=_to_builtin),
    )
    return path


def _to_builtin(value):
    """Make numpy types json serialisable, so `extra` can hold whatever a program wants to record."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return str(value)


def load(subject: str, method: str) -> dict:
    """Load the result of one subject for one method, as a plain dict."""
    with np.load(path_of(subject, method), allow_pickle=False) as data:
        result = {key: data[key] for key in data.files}

    for key in ("subject", "method", "unit"):
        result[key] = str(result[key])
    for key in ("rmse", "solver_time", "real_time"):
        result[key] = float(result[key]) if key in result else float("nan")
    for key in ("solver_iterations", "solver_status", "n_phases", "n_parameters"):
        result[key] = int(result[key]) if key in result else -1
    result["converged"] = bool(result["converged"])
    result["extra"] = json.loads(str(result["extra"]))
    result["parameters"] = dict(zip(result["parameter_names"].tolist(), result["parameter_values"].tolist()))
    result["parameters_at_bound"] = result["parameters_at_bound"].tolist()
    return result


def load_method(method: str) -> dict:
    """Load every subject's result for one method, as a {subject: result} dict sorted by subject."""
    folder = RESULTS_ROOT / method
    results = {}
    for path in sorted(folder.glob(f"p*_{method}{SUFFIX}")):
        subject = path.name.split("_")[0][1:]
        results[subject] = load(subject, method)
    return results


def phases_of(result: dict) -> list:
    """Split a result back into its phases, as (time, tracked, identified) triples."""
    lengths = result["phase_lengths"]
    if len(lengths) == 0:
        return [(result["time"], result["tracked"], result["identified"])]

    phases, start = [], 0
    for length in lengths:
        stop = start + int(length)
        phases.append((result["time"][start:stop], result["tracked"][start:stop], result["identified"][start:stop]))
        start = stop
    return phases


def summary_table(method: str) -> str:
    """A one-line-per-subject summary of a method, handy to check a whole batch."""
    results = load_method(method)
    if not results:
        return f"No result found for method '{method}' in {RESULTS_ROOT / method}"

    # The union of every subject's parameter names, as one subject may carry extra ones (ex: the passive stop)
    parameter_names = []
    for result in results.values():
        parameter_names += [name for name in result["parameters"] if name not in parameter_names]
    unit = next(iter(results.values()))["unit"]
    header = f"{'subject':>8} {'rmse (' + unit + ')':>12} {'conv':>5}  " + "  ".join(
        f"{name:>10}" for name in parameter_names
    )
    lines = [header, "-" * len(header)]
    for subject, result in results.items():
        values = "  ".join(
            f"{result['parameters'][name]:>10.4g}" if name in result["parameters"] else f"{'-':>10}"
            for name in parameter_names
        )
        lines.append(
            f"{subject:>8} {result['rmse']:>12.4g} {str(result['converged']):>5}  {values}"
            + (f"   at bound: {result['parameters_at_bound']}" if result["parameters_at_bound"] else "")
        )
    return "\n".join(lines)


if __name__ == "__main__":
    for folder in sorted(p.name for p in RESULTS_ROOT.glob("*") if p.is_dir() and p.name != "debug"):
        print(f"\n=== {folder} ===")
        print(summary_table(folder))
