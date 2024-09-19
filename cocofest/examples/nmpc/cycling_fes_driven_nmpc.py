"""
This example will do a nmpc of 10 stimulation example with Ding's 2007 frequency model.
This ocp was build to produce a elbow motion from 5 to 120 degrees.
The pulse duration between minimal sensitivity threshold and 600us to satisfy the flexion and minimizing required elbow
torque control.
"""

import numpy as np
import biorbd
from bioptim import Solver
from cocofest import (
    DingModelPulseDurationFrequencyWithFatigue,
    NmpcFesMsk,
    FesMskModel,
    PickleAnimate,
    SolutionToPickle,
)


minimum_pulse_duration = DingModelPulseDurationFrequencyWithFatigue().pd0
DeltoideusClavicle_A_model = DingModelPulseDurationFrequencyWithFatigue(
    muscle_name="DeltoideusClavicle_A"
)
DeltoideusScapula_P_model = DingModelPulseDurationFrequencyWithFatigue(
    muscle_name="DeltoideusScapula_P"
)
TRIlong_model = DingModelPulseDurationFrequencyWithFatigue(muscle_name="TRIlong")
BIC_long_model = DingModelPulseDurationFrequencyWithFatigue(muscle_name="BIC_long")
BIC_brevis_model = DingModelPulseDurationFrequencyWithFatigue(muscle_name="BIC_brevis")

DeltoideusClavicle_A_model.alpha_a = -4.0 * 10e-1
DeltoideusScapula_P_model.alpha_a = -4.0 * 10e-1
TRIlong_model.alpha_a = -4.0 * 10e-1
BIC_long_model.alpha_a = -4.0 * 10e-1
BIC_brevis_model.alpha_a = -4.0 * 10e-1

model = FesMskModel(
    name=None,
    biorbd_path="../msk_models/simplified_UL_Seth.bioMod",
    muscles_model=[
        DeltoideusClavicle_A_model,
        DeltoideusScapula_P_model,
        TRIlong_model,
        BIC_long_model,
        BIC_brevis_model,
    ],
    activate_force_length_relationship=True,
    activate_force_velocity_relationship=True,
)

nmpc_fes_msk = NmpcFesMsk()
nmpc_fes_msk.n_cycles = 8
nmpc = NmpcFesMsk.prepare_nmpc(
    model=model,
    stim_time=list(np.round(np.linspace(0, 1, 31), 3))[:-1],
    cycle_len=100,
    cycle_duration=1,
    pulse_duration={
        "min": minimum_pulse_duration,
        "max": 0.0006,
        "bimapping": False,
    },
    msk_info={"with_residual_torque": True},
    objective={
        "cycling": {"x_center": 0.35, "y_center": 0, "radius": 0.1, "target": "marker"},
        "minimize_muscle_fatigue": True,
        "minimize_residual_torque": True,
    },
    warm_start=True,
    n_threads=8,
)

sol = nmpc.solve(
    nmpc_fes_msk.update_functions,
    solver=Solver.IPOPT(_hessian_approximation="limited-memory", max_iter=100000),
    cyclic_options={"states": {}},
    get_all_iterations=True,
)

SolutionToPickle(
    sol[0], "results/cycling_fes_driven_nmpc_full_fatigue.pkl", ""
).pickle()
[
    SolutionToPickle(
        sol[1][i], "cycling_fes_driven_nmpc_" + str(i) + "_fatigue.pkl", ""
    ).pickle()
    for i in range(len(sol[1]))
]

# biorbd_model = biorbd.Model("../msk_models/simplified_UL_Seth_full_mesh.bioMod")
# PickleAnimate("cycling_fes_driven_nmpc_full_force.pkl").animate(
#     model=biorbd_model
# )