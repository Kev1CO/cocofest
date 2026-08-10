"""
Cross validated identification, for every subject and any of the three models.

The first number is the tracking error, how well the model fits what it was given. The second is the prediction
error, how well those parameters carry over. Their ratio is what says whether the identification generalizes.

The scheme is stratified k-fold, k = 5 by default.
"""

import sys
from pathlib import Path

import numpy as np

OPTIMIZATION_ROOT = Path(__file__).resolve().parent.parent
if str(OPTIMIZATION_ROOT) not in sys.path:
    sys.path.insert(0, str(OPTIMIZATION_ROOT))
import force_ocp
import movement_ocp
import passive_ocp
from helper import debug_plots, holdout, results


def _force_case(subject, formulation, passive_method):
    """Everything the pipeline needs to run the isometric force model on one subject."""
    ocp_data = force_ocp.load_force_data(subject)
    if ocp_data is None:
        return None

    def identify(keep, method, extra):
        return force_ocp.identify(
            subject=subject,
            ocp_data=force_ocp.select_trains(ocp_data, keep),
            method=method,
            plot=False,
            save=True,
            debug=False,
            extra=extra,
        )

    def predict(test, parameters):
        return force_ocp.predict(force_ocp.select_trains(ocp_data, test), parameters)

    return {
        "n_phases": len(ocp_data["force"]),
        "conditions": force_ocp.train_conditions(ocp_data),
        "identify": identify,
        "predict": predict,
    }


def _movement_case(subject, formulation, passive_method):
    """Everything the pipeline needs to run the FES model on the measured movements of one subject."""
    phases = movement_ocp.collect_movement_phases(subject, debug=False)
    if not phases:
        return None

    def rebase(selection):
        """Lay a subset of the phases back out on one continuous time base."""
        offset, out = 0.0, []
        for i in selection:
            phase = dict(phases[i])
            phase["time_offset"] = offset
            phase["global_time"] = phase["time"] - phase["time"][0] + offset
            offset += phase["final_time"]
            out.append(phase)
        return out

    def identify(keep, method, extra):
        return movement_ocp.identify(
            subject=subject,
            phases=rebase(keep),
            method=method,
            passive_method=passive_method,
            plot=False,
            save=True,
            debug=False,
            extra=extra,
        )

    def predict(test, parameters):
        return movement_ocp.predict(subject, rebase(test), parameters, passive_method)

    return {
        "n_phases": len(phases),
        "conditions": [{"frequency_hz": p["frequency"], "pulse_width_us": p["pulse_width"] * 1e6} for p in phases],
        "identify": identify,
        "predict": predict,
    }


def _passive_case(subject, formulation, passive_method):
    """Everything the pipeline needs to run the passive torque model on one subject."""
    global_q, global_final_time, global_time = passive_ocp.collect_relaxation_phases(
        subject, passive_ocp.ALL_CONDITIONS, debug=False
    )
    if not global_q:
        return None

    per_recording = max(1, len(global_q) // len(passive_ocp.ALL_CONDITIONS))
    conditions = [
        {
            "frequency_hz": passive_ocp.ALL_CONDITIONS[min(i // per_recording, len(passive_ocp.ALL_CONDITIONS) - 1)][0],
            "pulse_width_us": 375.0,
        }
        for i in range(len(global_q))
    ]

    def rebase(selection):
        offset, q, final_time, time = 0.0, [], [], []
        for i in selection:
            q.append(global_q[i])
            final_time.append(global_final_time[i])
            time.append(global_time[i] - global_time[i][0] + offset)
            offset += global_final_time[i]
        return q, final_time, time

    def identify(keep, method, extra):
        q, final_time, time = rebase(keep)
        return passive_ocp.identify(
            subject=subject,
            global_q=q,
            global_final_time=final_time,
            global_time=time,
            method=method,
            formulation=formulation,
            plot=False,
            save=True,
            debug=False,
            extra=extra,
        )

    def predict(test, parameters):
        return passive_ocp.predict(subject, *rebase(test), parameters, formulation)

    return {"n_phases": len(global_q), "conditions": conditions, "identify": identify, "predict": predict}


CASES = {"force": _force_case, "movement": _movement_case, "passive": _passive_case}


def main(
    kind="force",
    subjects=range(1, 21),
    folds=holdout.DEFAULT_FOLDS,
    seed=0,
    formulation="riener",
    passive_method="passive_torque_id_all_riener",
):
    """
    Parameters
    ----------
    kind: str
        "force", "movement" or "passive"
    subjects: iterable
        The subjects to run
    folds: int | str
        Number of folds, or "loo" for leave one out. See helper/holdout.py.
    seed: int
        Shuffling seed of the fold assignment, so a run can be repeated exactly
    formulation: str
        The passive torque formulation, for kind="passive"
    passive_method: str
        The stored passive identification the movement model takes its articular torque from
    """
    if kind not in CASES:
        raise ValueError(f"Unknown kind '{kind}', pick one of {sorted(CASES)}.")

    debug_plots.set_output(None)  # the pipeline is a batch, no figures
    per_subject = {}

    for i in subjects:
        subject = f"{int(i):02d}"
        case = CASES[kind](subject, formulation, passive_method)
        if case is None:
            print(f"P{subject}: no usable data for '{kind}', skipping.")
            continue

        splits = holdout.k_fold(case["conditions"], folds=folds, seed=seed)
        tracking_errors, prediction_errors = [], []

        for fold, (train, test) in enumerate(splits):
            method = f"{kind}_id_cv"
            tag = {"fold": fold, "n_folds": len(splits), "train_phases": train, "test_phases": test, "seed": seed}
            print(f"\n=== P{subject} {kind} fold {fold + 1}/{len(splits)}: train {len(train)}, test {len(test)}")

            try:
                parameters = case["identify"](train, f"{method}_fold{fold}", tag)
                prediction = case["predict"](test, parameters)
            except Exception as error:  # one bad fold must not take the batch down
                print(f"  failed ({type(error).__name__}: {error})")
                continue

            identification = results.load(subject, f"{method}_fold{fold}")
            tracking_errors.append(identification["rmse"])
            prediction_errors.append(prediction["rmse"])
            print(
                f"  tracking {identification['rmse']:.4g} {identification['unit']}, "
                f"prediction {prediction['rmse']:.4g} {identification['unit']}, "
                f"{identification['solver_iterations']} iters in {identification['real_time']:.1f} s"
            )

            results.save(
                subject=subject,
                method=f"{method}_prediction_fold{fold}",
                parameters=parameters,
                time=prediction["time"],
                tracked=prediction["tracked"],
                identified=prediction["identified"],
                unit=identification["unit"],
                converged=prediction.get("converged", True),
                extra={**tag, "identified_with": f"{method}_fold{fold}", "tracking_rmse": identification["rmse"]},
            )

        if prediction_errors:
            per_subject[subject] = {
                "tracking": holdout.summarise(tracking_errors),
                "prediction": holdout.summarise(prediction_errors),
            }

    if per_subject:
        print(f"\n{'subject':>8} {'folds':>6} {'tracking':>18} {'prediction':>18} {'ratio':>7}")
        print("-" * 62)
        for subject, stats in per_subject.items():
            tracking, prediction = stats["tracking"], stats["prediction"]
            ratio = prediction["mean"] / tracking["mean"] if tracking["mean"] > 0 else float("nan")
            print(
                f"{subject:>8} {prediction['n']:>6} "
                f"{tracking['mean']:>10.4g} +- {tracking['std']:<5.3g} "
                f"{prediction['mean']:>10.4g} +- {prediction['std']:<5.3g} {ratio:>7.2f}"
            )
        ratios = [
            s["prediction"]["mean"] / s["tracking"]["mean"] for s in per_subject.values() if s["tracking"]["mean"] > 0
        ]
        print(f"\nprediction / tracking over subjects: median {np.median(ratios):.2f}, max {np.max(ratios):.2f}")


if __name__ == "__main__":
    kind = sys.argv[1] if len(sys.argv) > 1 else "force"
    folds = sys.argv[2] if len(sys.argv) > 2 else holdout.DEFAULT_FOLDS
    main(kind=kind, folds=folds if folds == "loo" else int(folds))
