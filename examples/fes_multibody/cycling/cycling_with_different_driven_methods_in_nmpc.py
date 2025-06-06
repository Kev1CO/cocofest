# NMPC FES not working yet
"""
This example will do an optimal control program of a 100 steps tracking a hand cycling motion with a torque driven and
a torque resistance at the handle.
"""

import matplotlib.pyplot as plt
import numpy as np

from bioptim import (
    Axis,
    BiorbdModel,
    BoundsList,
    ConstraintList,
    ConstraintFcn,
    CostType,
    DynamicsFcn,
    DynamicsList,
    ExternalForceSetTimeSeries,
    InitialGuessList,
    InterpolationType,
    MultiCyclicCycleSolutions,
    MultiCyclicNonlinearModelPredictiveControl,
    ObjectiveFcn,
    ObjectiveList,
    OdeSolver,
    PhaseDynamics,
    SolutionMerge,
    Solution,
    Solver,
    ParameterList,
)

from cocofest import (
    CustomObjective,
    DingModelPulseWidthFrequency,
    DingModelPulseWidthFrequencyWithFatigue,
    FesMskModel,
    inverse_kinematics_cycling,
    OcpFesMsk,
    FesNmpcMsk,
)


class MyCyclicNMPC(FesNmpcMsk):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def advance_window_bounds_states(self, sol, n_cycles_simultaneous=None, **extra):
        min_pedal_bound = self.nlp[0].x_bounds["q"].min[-1, 0]
        max_pedal_bound = self.nlp[0].x_bounds["q"].max[-1, 0]
        super(MyCyclicNMPC, self).advance_window_bounds_states(sol)
        self.nlp[0].x_bounds["q"].min[-1, 0] = min_pedal_bound
        self.nlp[0].x_bounds["q"].max[-1, 0] = max_pedal_bound
        return True

    def advance_window_initial_guess_states(self, sol, n_cycles_simultaneous=None):
        # Reimplementation of the advance_window method so the rotation of the wheel restart at -pi
        super(MyCyclicNMPC, self).advance_window_initial_guess_states(sol)
        q = sol.decision_states(to_merge=SolutionMerge.NODES)["q"]
        self.nlp[0].x_init["q"].init[:, :] = q[:, :]  # Keep the previously found value for the wheel
        return True


def prepare_nmpc(
    model: BiorbdModel | FesMskModel,
    cycle_duration: int | float,
    cycle_len: int,
    n_cycles_to_advance: int,
    n_cycles_simultaneous: int,
    total_n_cycles: int,
    turn_number: int,
    pedal_config: dict,
    external_force: dict,
    dynamics_type: str = "torque_driven",
    use_sx: bool = True,
):

    window_n_shooting = cycle_len * n_cycles_simultaneous
    window_cycle_duration = cycle_duration * n_cycles_simultaneous

    # Dynamics
    numerical_time_series, external_force_set = set_external_forces(
        n_shooting=window_n_shooting, external_force_dict=external_force
    )

    if dynamics_type == "fes_driven":
        numerical_data_time_series, stim_idx_at_node_list = model.muscles_dynamics_model[
            0
        ].get_numerical_data_time_series(window_n_shooting, window_cycle_duration)
        numerical_time_series.update(numerical_data_time_series)

    dynamics = set_dynamics(
        model=model,
        numerical_time_series=numerical_time_series,
        dynamics_type_str=dynamics_type,
        ode_solver=OdeSolver.RK1(n_integration_steps=5),
    )

    # Define objective functions
    objective_functions = set_objective_functions(model, dynamics_type)

    # Initial q guess
    x_init = set_x_init(window_n_shooting, pedal_config, turn_number)

    # Path constraint
    x_bounds, x_init = set_bounds(
        model=model,
        x_init=x_init,
        n_shooting=window_n_shooting,
        turn_number=turn_number,
        interpolation_type=InterpolationType.EACH_FRAME,
        cardinal=2,
    )

    # Control path constraint
    u_bounds, u_init = set_u_bounds_and_init(model, dynamics_type_str=dynamics_type)

    # Constraints
    constraints = set_constraints(model, window_n_shooting, turn_number)

    # Update model
    model = updating_model(model=model, external_force_set=external_force_set, parameters=ParameterList(use_sx=use_sx))

    return MyCyclicNMPC(
        bio_model=[model],
        dynamics=dynamics,
        cycle_len=cycle_len,
        cycle_duration=cycle_duration,
        n_cycles_simultaneous=n_cycles_simultaneous,
        n_cycles_to_advance=n_cycles_to_advance,
        common_objective_functions=objective_functions,
        constraints=constraints,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_init,
        u_init=u_init,
        n_threads=20,
        use_sx=False,
    )


def set_external_forces(n_shooting, external_force_dict):
    external_force_set = ExternalForceSetTimeSeries(nb_frames=n_shooting)
    external_force_array = np.array(external_force_dict["torque"])
    reshape_values_array = np.tile(external_force_array[:, np.newaxis], (1, n_shooting))
    external_force_set.add_torque(segment=external_force_dict["Segment_application"], values=reshape_values_array)
    numerical_time_series = {"external_forces": external_force_set.to_numerical_time_series()}
    return numerical_time_series, external_force_set


def updating_model(model, external_force_set, parameters=None):
    if isinstance(model, FesMskModel):
        model = FesMskModel(
            name=model.name,
            biorbd_path=model.biorbd_path,
            muscles_model=model.muscles_dynamics_model,
            stim_time=model.muscles_dynamics_model[0].stim_time,
            previous_stim=model.muscles_dynamics_model[0].previous_stim,
            activate_force_length_relationship=model.activate_force_length_relationship,
            activate_force_velocity_relationship=model.activate_force_velocity_relationship,
            activate_residual_torque=model.activate_residual_torque,
            parameters=parameters,
            external_force_set=external_force_set,
        )
    else:
        model = BiorbdModel(model.path, external_force_set=external_force_set)

    return model


def set_dynamics(
    model, numerical_time_series, dynamics_type_str="torque_driven", ode_solver=OdeSolver.RK4(n_integration_steps=10)
):
    dynamics_type = (
        DynamicsFcn.TORQUE_DRIVEN
        if dynamics_type_str == "torque_driven"
        else (
            DynamicsFcn.MUSCLE_DRIVEN
            if dynamics_type_str == "muscle_driven"
            else model.declare_model_variables if dynamics_type_str == "fes_driven" else None
        )
    )
    if dynamics_type is None:
        raise ValueError("Dynamics type not recognized")

    dynamics = DynamicsList()
    dynamics.add(
        dynamics_type=dynamics_type,
        dynamic_function=(
            None if dynamics_type in (DynamicsFcn.TORQUE_DRIVEN, DynamicsFcn.MUSCLE_DRIVEN) else model.muscle_dynamic
        ),
        expand_dynamics=True,
        expand_continuity=False,
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        numerical_data_timeseries=numerical_time_series,
        with_contact=True,
        phase=0,
        ode_solver=ode_solver,
    )
    return dynamics


def set_objective_functions(model, dynamics_type):
    objective_functions = ObjectiveList()
    if isinstance(model, FesMskModel):
        objective_functions.add(
            CustomObjective.minimize_overall_muscle_force_production,
            custom_type=ObjectiveFcn.Lagrange,
            weight=1,
            quadratic=True,
        )
        # Uncomment these following lines if muscle fatigue minimization is desired:
        # objective_functions.add(
        #   CustomObjective.minimize_overall_muscle_fatigue,
        #   custom_type=ObjectiveFcn.Lagrange,
        #   weight=1,
        #   quadratic=True)
    else:
        control_key = "tau" if dynamics_type == "torque_driven" else "muscles"
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key=control_key, weight=1000, quadratic=True)
    return objective_functions


def set_x_init(n_shooting, pedal_config, turn_number):
    x_init = InitialGuessList()

    biorbd_model_path = "../../model_msk/simplified_UL_Seth_2D_cycling_for_inverse_kinematics.bioMod"
    q_guess, qdot_guess, qddotguess = inverse_kinematics_cycling(
        biorbd_model_path,
        n_shooting,
        x_center=pedal_config["x_center"],
        y_center=pedal_config["y_center"],
        radius=pedal_config["radius"],
        ik_method="lm",
        cycling_number=turn_number,
    )
    x_init.add("q", q_guess, interpolation=InterpolationType.EACH_FRAME)
    # x_init.add("qdot", qdot_guess, interpolation=InterpolationType.EACH_FRAME)
    # u_guess = inverse_dynamics_cycling(biorbd_model_path, q_guess, qdot_guess, qddotguess)
    # u_init.add("tau", u_guess, interpolation=InterpolationType.EACH_FRAME)

    return x_init


def set_u_bounds_and_init(bio_model, dynamics_type_str):
    u_bounds = BoundsList()
    u_init = InitialGuessList()
    if dynamics_type_str == "torque_driven":
        u_bounds.add(key="tau", min_bound=np.array([-50, -50, -0]), max_bound=np.array([50, 50, 0]), phase=0)
    elif dynamics_type_str == "muscle_driven":
        muscle_min, muscle_max, muscle_init = 0.0, 1.0, 0.5
        u_bounds.add(
            key="muscles",
            min_bound=np.array([muscle_min] * bio_model.nb_muscles),
            max_bound=np.array([muscle_max] * bio_model.nb_muscles),
            phase=0,
        )
        u_init.add(key="muscles", initial_guess=np.array([muscle_init] * bio_model.nb_muscles), phase=0)
    elif dynamics_type_str == "fes_driven":
        u_bounds, u_init = OcpFesMsk.set_u_bounds_fes(bio_model)
    return u_bounds, u_init


def set_bounds(model, x_init, n_shooting, turn_number, interpolation_type=InterpolationType.CONSTANT, cardinal=4):
    x_bounds, x_init_fes = OcpFesMsk.set_x_bounds_fes(model)
    for key in x_init_fes.keys():
        x_init[key] = x_init_fes[key]

    q_x_bounds = model.bounds_from_ranges("q")
    q_x_bounds.min[2] = q_x_bounds.min[2] * (turn_number / 3)
    qdot_x_bounds = model.bounds_from_ranges("qdot")

    if interpolation_type == InterpolationType.EACH_FRAME:
        x_min_bound = []
        x_max_bound = []
        for i in range(q_x_bounds.min.shape[0]):
            x_min_bound.append([q_x_bounds.min[i][0]] * (n_shooting + 1))
            x_max_bound.append([q_x_bounds.max[i][0]] * (n_shooting + 1))

        cardinal_node_list = [
            i * int(n_shooting / ((n_shooting / (n_shooting / turn_number)) * cardinal))
            for i in range(int((n_shooting / (n_shooting / turn_number)) * cardinal + 1))
        ]
        cardinal_node_list = [int(cardinal_node_list[i]) for i in range(len(cardinal_node_list))]
        slack = 10 * (np.pi / 180)
        for i in range(len(x_min_bound[0])):
            x_min_bound[0][i] = 0
            x_max_bound[0][i] = 1
            x_min_bound[1][i] = 1
            x_min_bound[2][i] = x_init["q"].init[2][-1]
            x_max_bound[2][i] = x_init["q"].init[2][0]
        for i in range(len(cardinal_node_list)):
            cardinal_index = cardinal_node_list[i]
            x_min_bound[2][cardinal_index] = (
                x_init["q"].init[2][cardinal_index]
                if i % cardinal == 0
                else x_init["q"].init[2][cardinal_index] - slack
            )
            x_max_bound[2][cardinal_index] = (
                x_init["q"].init[2][cardinal_index]
                if i % cardinal == 0
                else x_init["q"].init[2][cardinal_index] + slack
            )
            # x_min_bound[2][cardinal_index] = x_init["q"].init[2][cardinal_index] - slack
            # x_max_bound[2][cardinal_index] = x_init["q"].init[2][cardinal_index] + slack

        x_bounds.add(
            key="q", min_bound=x_min_bound, max_bound=x_max_bound, phase=0, interpolation=InterpolationType.EACH_FRAME
        )

    else:
        x_bounds.add(key="q", bounds=q_x_bounds, phase=0)

    # Modifying pedal speed bounds
    qdot_x_bounds.max[0] = [10, 10, 10]
    qdot_x_bounds.min[0] = [-10, -10, -10]
    qdot_x_bounds.max[1] = [10, 10, 10]
    qdot_x_bounds.min[1] = [-10, -10, -10]
    qdot_x_bounds.max[2] = [-2, -2, -2]
    qdot_x_bounds.min[2] = [-12, -12, -12]
    x_bounds.add(key="qdot", bounds=qdot_x_bounds, phase=0)
    return x_bounds, x_init


def set_constraints(bio_model, n_shooting, turn_number):
    constraints = ConstraintList()
    superimpose_marker_list = [
        i * int(n_shooting / ((n_shooting / (n_shooting / turn_number)) * 1))
        for i in range(int((n_shooting / (n_shooting / turn_number)) * 1 + 1))
    ]

    for i in superimpose_marker_list:
        constraints.add(
            ConstraintFcn.SUPERIMPOSE_MARKERS,
            first_marker="wheel_center",
            second_marker="global_wheel_center",
            node=i,
            axes=[Axis.X, Axis.Y],
        )
        constraints.add(
            ConstraintFcn.TRACK_MARKERS_VELOCITY,
            node=i,
            marker_index=bio_model.marker_index("wheel_center"),
            axes=[Axis.X, Axis.Y],
        )

    return constraints


def main():
    """
    Main function to configure and solve the optimal control problem.
    """
    # --- Configuration --- #
    dynamics_type = "fes_driven"  # Available options: "torque_driven", "muscle_driven", "fes_driven"
    model_path = "../../model_msk/simplified_UL_Seth_2D_cycling.bioMod"

    # NMPC parameters
    cycle_duration = 1
    cycle_len = 100
    n_cycles_to_advance = 1
    n_cycles_simultaneous = 3
    n_cycles = 3

    # Bike parameters
    turn_number = n_cycles_simultaneous
    pedal_config = {"x_center": 0.35, "y_center": 0.0, "radius": 0.1}

    # --- Load the appropriate model --- #
    if dynamics_type in ["torque_driven", "muscle_driven"]:
        model = BiorbdModel(model_path)
    elif dynamics_type == "fes_driven":
        # Define muscle dynamics for the FES-driven model
        muscles_model = [
            DingModelPulseWidthFrequencyWithFatigue(muscle_name="DeltoideusClavicle_A", sum_stim_truncation=10),
            DingModelPulseWidthFrequencyWithFatigue(muscle_name="DeltoideusScapula_P", sum_stim_truncation=10),
            DingModelPulseWidthFrequencyWithFatigue(muscle_name="TRIlong", sum_stim_truncation=10),
            DingModelPulseWidthFrequencyWithFatigue(muscle_name="BIC_long", sum_stim_truncation=10),
            DingModelPulseWidthFrequencyWithFatigue(muscle_name="BIC_brevis", sum_stim_truncation=10),
        ]
        stim_time = list(np.linspace(0, cycle_duration * n_cycles_simultaneous, 100)[:-1])
        model = FesMskModel(
            name=None,
            biorbd_path=model_path,
            muscles_model=muscles_model,
            stim_time=stim_time,
            activate_force_length_relationship=True,
            activate_force_velocity_relationship=True,
            activate_passive_force_relationship=True,
            activate_residual_torque=False,
            external_force_set=None,  # External forces will be added later
        )
        # Adjust n_shooting based on the stimulation time
        cycle_len = int(len(stim_time) / 3)
    else:
        raise ValueError(f"Dynamics type '{dynamics_type}' not recognized")

    resistive_torque = {
        "Segment_application": "wheel",
        "torque": np.array([0, 0, -1]),
    }

    nmpc = prepare_nmpc(
        model=model,
        cycle_duration=cycle_duration,
        cycle_len=cycle_len,
        n_cycles_to_advance=n_cycles_to_advance,
        n_cycles_simultaneous=n_cycles_simultaneous,
        total_n_cycles=n_cycles,
        turn_number=turn_number,
        pedal_config=pedal_config,
        dynamics_type=dynamics_type,
        external_force=resistive_torque,
    )
    nmpc.n_cycles_simultaneous = n_cycles_simultaneous

    def update_functions(_nmpc: MultiCyclicNonlinearModelPredictiveControl, cycle_idx: int, _sol: Solution):
        return cycle_idx < n_cycles  # True if there are still some cycle to perform

    # Add the penalty cost function plot
    nmpc.add_plot_penalty(CostType.ALL)
    # Solve the optimal control problem
    sol = nmpc.solve_fes_nmpc(
        update_functions,
        solver=Solver.IPOPT(show_online_optim=False, _max_iter=10000, show_options=dict(show_bounds=True)),
        total_cycles=n_cycles,
        external_force=resistive_torque,
        cycle_solutions=MultiCyclicCycleSolutions.ALL_CYCLES,
        get_all_iterations=True,
        cyclic_options={"states": {}},
        max_consecutive_failing=3,
    )

    # Display graphs and animate the solution
    sol[1][0].graphs(show_bounds=True)
    sol[1][1].graphs(show_bounds=True)
    print(sol[1][1].constraints)
    sol[0].graphs(show_bounds=True)
    print(sol[0].constraints)
    print(sol[0].parameters)
    print(sol[0].detailed_cost)
    sol[0].animate(viewer="pyorerun", n_frames=200, show_tracked_markers=True)


if __name__ == "__main__":
    main()
