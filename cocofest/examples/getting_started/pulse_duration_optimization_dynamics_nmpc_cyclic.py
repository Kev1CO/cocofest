"""
This example showcases a moving time horizon simulation problem of cyclic muscle force tracking.
The FES model used here is Ding's 2007 pulse duration and frequency model with fatigue.
Only the pulse duration is optimized, frequency is fixed.
The nmpc cyclic problem is composed of 3 cycles and will move forward 1 cycle at each step.
Only the middle cycle is kept in the optimization problem, the nmpc cyclic problem stops once the last 6th cycle is reached.
"""

import numpy as np
import matplotlib.pyplot as plt

from bioptim import OdeSolver, ObjectiveFcn, ObjectiveList, Solver
from cocofest import OcpFesDynamicsNmpcCyclic, DingModelPulseDurationFrequencyWithFatigue, ModelConfig, ModelBuilder, FesMskModel


# --- Build nmpc cyclic --- #
n_total_cycles = 3
n_stim = 10
minimum_pulse_duration = DingModelPulseDurationFrequencyWithFatigue().pd0

delt_ant = DingModelPulseDurationFrequencyWithFatigue(muscle_name="DeltoideusClavicle_A", a_scale=5000, alpha_a=-4.0 * 10e-1, alpha_tau1=2.1 * 10e-3)
delt_post = DingModelPulseDurationFrequencyWithFatigue(muscle_name="DeltoideusScapula_P")
triceps = DingModelPulseDurationFrequencyWithFatigue(muscle_name="TRIlong")
biceps = DingModelPulseDurationFrequencyWithFatigue(muscle_name="BIC_long")
biceps_short = DingModelPulseDurationFrequencyWithFatigue(muscle_name="BIC_brevis")

model_configuration = ModelConfig(model=FesMskModel,
                                  biorbd_model_path="../msk_models/simplified_UL_Seth.bioMod",
                                  fes_muscle_models=[
                                    delt_ant,
                                    delt_post,
                                    triceps,
                                    biceps,
                                    biceps_short,
                                  ],
                                  activate_force_length_relationship=True,
                                  activate_force_velocity_relationship=True,
                                  )

ding_builder = ModelBuilder(config=model_configuration, stim_time=np.linspace(0, 1, n_stim+1)[:-1].tolist())
models = ding_builder.build(cycle_final_time=1)

# --- Minimize residual torque --- #
objective_functions = ObjectiveList()
for i in range(n_stim*n_total_cycles):
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", weight=100, quadratic=True, phase=i)

nmpc = OcpFesDynamicsNmpcCyclic(
    models=models,
    n_stim=n_stim,
    n_shooting=5,
    final_time=1,
    pulse_duration={
        "min": minimum_pulse_duration,
        "max": 0.0006,
        "bimapping": False,
    },
    objective={"cycling": {"x_center": 0.35, "y_center": 0, "radius": 0.1, "target": "marker"}},
    with_residual_torque=True,
    n_total_cycles=n_total_cycles,
    n_simultaneous_cycles=3,
    n_cycle_to_advance=1,
    cycle_to_keep="middle",
    use_sx=True,
    ode_solver=OdeSolver.COLLOCATION(),
)

nmpc.prepare_nmpc()
nmpc.solve(solver=Solver.IPOPT(show_online_optim=False, _max_iter=10000))

# --- Show results --- #
time = [j for sub in nmpc.result["time"] for j in sub]
fatigue = [j for sub in nmpc.result["states"]["A"] for j in sub]
force = [j for sub in nmpc.result["states"]["F"] for j in sub]

ax1 = plt.subplot(221)
ax1.plot(time, fatigue, label="A", color="green")
ax1.set_title("Fatigue", weight="bold")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Force scaling factor (-)")
plt.legend()

ax2 = plt.subplot(222)
ax2.plot(time, force, label="F", color="red", linewidth=4)
for i in range(n_total_cycles):
    if i == 0:
        ax2.plot(target_time, target_force, label="Target", color="purple")
    else:
        ax2.plot(target_time + i, target_force, color="purple")
ax2.set_title("Force", weight="bold")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Force (N)")
plt.legend()

barWidth = 0.25  # set width of bar
cycles = nmpc.result["parameters"]["pulse_duration"]  # set height of bar
bar = []  # Set position of bar on X axis
for i in range(n_total_cycles):
    if i == 0:
        br = [barWidth * (x + 1) for x in range(len(cycles[i]))]
    else:
        br = [bar[-1][-1] + barWidth * (x + 1) for x in range(len(cycles[i]))]
    bar.append(br)

ax3 = plt.subplot(212)
for i in range(n_total_cycles):
    ax3.bar(bar[i], cycles[i], width=barWidth, edgecolor="grey", label=f"cycle n°{i+1}")
ax3.set_xticks([np.mean(r) for r in bar], [str(i + 1) for i in range(n_total_cycles)])
ax3.set_xlabel("Cycles")
ax3.set_ylabel("Pulse duration (s)")
plt.legend()
ax3.set_title("Pulse duration", weight="bold")
plt.show()
