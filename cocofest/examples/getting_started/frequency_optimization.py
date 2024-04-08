"""
This example will do a 10 stimulation example with Ding's 2003 frequency model.
This ocp was build to match a force value of 270N at the end of the last node.
"""
import numpy as np
from cocofest import DingModelFrequencyWithFatigue, DingModelFrequency, CustomConstraint
from bioptim import (
    OptimalControlProgram,
    DynamicsList,
    BoundsList,
    InitialGuessList,
    InterpolationType,
    ObjectiveList,
    ObjectiveFcn,
    Node,
    ConstraintList,
    ParameterList,
    PhaseDynamics,
    ControlType,
    OdeSolver,
    VariableScaling,
    ParameterObjectiveList,
Solver)

# # --- Build ocp --- #
# # This ocp was build to match a force value of 270N at the end of the last node.
# # The stimulation will be optimized between 0.01 to 0.1 seconds and are equally spaced (a fixed frequency).
#
# ocp = OcpFes().prepare_ocp(
#     model=DingModelFrequencyWithFatigue(),
#     n_stim=10,
#     n_shooting=20,
#     final_time=1,
#     end_node_tracking=270,
#     time_min=0.01,
#     time_max=0.1,
#     time_bimapping=True,
#     use_sx=True,
# )
#
# # --- Solve the program --- #
# sol = ocp.solve()
#
# # --- Show results --- #
# sol.graphs()

model = DingModelFrequencyWithFatigue()
n_stim = 10
n_shooting = 100
final_time = 1
end_node_tracking = 100
time_min = 0.01
time_max = 0.1
# time_bimapping = True
use_sx = True

models = DingModelFrequencyWithFatigue()
final_time_phase = (final_time,)

parameters = ParameterList(use_sx=use_sx)
parameters_bounds = BoundsList()
parameters_init = InitialGuessList()
parameter_objectives = ParameterObjectiveList()
constraints = ConstraintList()

parameters.add(
    name="pulse_apparition_time",
    function=DingModelFrequency.set_pulse_apparition_time,
    size=n_stim,
    scaling=VariableScaling("pulse_apparition_time", [1] * n_stim),
)

# if time_min and time_max:
#     time_min_list = [time_max * n for n in range(n_stim)]
#     time_max_list = [time_min * n for n in range(n_stim)]
# else:
# time_min_list = [0] * n_stim
# time_max_list = [1] * n_stim

time_min_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
time_max_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# time_min_list = [0, 0.01, 0.02]
# time_max_list = [0, 0.1, 0.2]

parameters_bounds.add(
    "pulse_apparition_time",
    min_bound=np.array(time_min_list),
    max_bound=np.array(time_max_list),
    interpolation=InterpolationType.CONSTANT,
)
param_init = []
for i in range(n_stim):
    param_init.append((time_min_list[i] + time_max_list[i]) / 2)

# parameters_init["pulse_apparition_time"] = np.array([0, 0.1, 0.2])
parameters_init["pulse_apparition_time"] = np.array(param_init)

# constraints.add(CustomConstraint.pulse_time_apparition_as_node, node=Node.START, phase=0, target=0, pulse_idx=1)
# constraints.add(CustomConstraint.pulse_time_apparition_as_node, node=Node.START, phase=0, target=0, pulse_idx=2)

# for i in range(1, n_stim):
#     constraints.add(CustomConstraint.pulse_time_apparition_as_node, node=Node.START, phase=0, target=0, pulse_idx=i)

# if time_bimapping and time_min and time_max:
# constraints.add(CustomConstraint.equal_to_first_pulse_interval_time_single_phase, node=Node.START, target=0, phase=0, pulse_idx=1)
for i in range(2, n_stim):
    constraints.add(CustomConstraint.equal_to_first_pulse_interval_time_single_phase, node=Node.START, target=0, phase=0, pulse_idx=i)

dynamics = DynamicsList()
dynamics.add(
    models.declare_ding_variables,
    dynamic_function=models.dynamics,
    expand_dynamics=True,
    expand_continuity=False,
    phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
)

x_bounds = BoundsList()
variable_bound_list = models.name_dof
starting_bounds, min_bounds, max_bounds = (
    models.standard_rest_values(),
    models.standard_rest_values(),
    models.standard_rest_values(),
)

for i in range(len(variable_bound_list)):
    if variable_bound_list[i] == "Cn" or variable_bound_list[i] == "F":
        max_bounds[i] = 1000
    elif variable_bound_list[i] == "Tau1" or variable_bound_list[i] == "Km":
        max_bounds[i] = 1
    elif variable_bound_list[i] == "A":
        min_bounds[i] = 0

starting_bounds_min = np.concatenate((starting_bounds, min_bounds, min_bounds), axis=1)
starting_bounds_max = np.concatenate((starting_bounds, max_bounds, max_bounds), axis=1)
middle_bound_min = np.concatenate((min_bounds, min_bounds, min_bounds), axis=1)
middle_bound_max = np.concatenate((max_bounds, max_bounds, max_bounds), axis=1)

for j in range(len(variable_bound_list)):
    x_bounds.add(
        variable_bound_list[j],
        min_bound=np.array([starting_bounds_min[j]]),
        max_bound=np.array([starting_bounds_max[j]]),
        phase=0,
        interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
    )

x_init = InitialGuessList()
for j in range(len(variable_bound_list)):
    x_init.add(variable_bound_list[j], models.standard_rest_values()[j], phase=0)

# Creates the objective for our problem
objective_functions = ObjectiveList()
objective_functions.add(
    ObjectiveFcn.Mayer.MINIMIZE_STATE,
    node=Node.END,
    key="F",
    quadratic=True,
    weight=1,
    target=end_node_tracking,
    phase=0,
)

ocp = OptimalControlProgram(
    bio_model=[models],
    dynamics=dynamics,
    n_shooting=n_shooting,
    phase_time=final_time_phase,
    objective_functions=objective_functions,
    x_init=x_init,
    x_bounds=x_bounds,
    constraints=constraints,
    parameters=parameters,
    parameter_bounds=parameters_bounds,
    parameter_init=parameters_init,
    parameter_objectives=parameter_objectives,
    control_type=ControlType.CONSTANT,
    use_sx=use_sx,
    ode_solver=OdeSolver.RK4(n_integration_steps=1),
    n_threads=1,
)

# --- Solve the program --- #
sol = ocp.solve(Solver.IPOPT(_max_iter=10000))

print(sol.parameters["pulse_apparition_time"])

# --- Show results --- #
sol.graphs(show_bounds=False)
