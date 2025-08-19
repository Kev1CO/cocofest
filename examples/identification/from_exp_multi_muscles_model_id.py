import pickle

import numpy as np
import matplotlib.pyplot as plt
from casadi import exp

from bioptim import (
    OdeSolver,
    OptimalControlProgram,
    ObjectiveFcn,
    Node,
    ControlType,
    ObjectiveList,
    BoundsList,
    InitialGuessList,
    InterpolationType,
    Solver,
    SolutionMerge,
    CostType, DynamicsList, PhaseDynamics, VariableScaling, SelectionMapping,
    Dependency,
)

from cocofest.identification.identification_method import DataExtraction
from cocofest import (
    DingModelPulseWidthFrequency,
    FesMskModel,
    OcpFesId,
    OcpFesIdMultibody,
)

from examples.data_process.c3d_to_q import C3dToQ


def set_tau1_rest_all_model(models, value):
    for model in models:
        model.set_tau1_rest(model=model, tau1_rest=value)


def set_tau2_all_model(models, value):
    for model in models:
        model.set_tau2(model=model, tau2=value)


def set_km_rest_all_model(models, value):
    for model in models:
        model.set_km_rest(model=model, km_rest=value)


def set_a_scale_all_model(models, value):
    for model in models:
        model.set_a_scale(model=model, a_scale=value)


def set_pd0_all_model(models, value):
    for model in models:
        model.set_pd0(model=model, pd0=value)


def set_pdt_all_model(models, value):
    for model in models:
        model.set_pdt(model=model, pdt=value)


def set_parameters(parameters_to_identify: list, additional_key_settings: dict, n_phase: int, i, parameters, parameters_bounds, parameters_init):
    if i == 0:
        mapping = None
    else:
        mapping = SelectionMapping(nb_elements=n_phase, independent_indices=(0,), dependencies=(Dependency(dependent_index=i, reference_index=0, factor=1),))

    for param in parameters_to_identify:
        parameters.add(
            name=param + "_" + str(i),
            function=additional_key_settings[param]["function"],
            size=1,
            scaling=VariableScaling(
                param,
                [additional_key_settings[param]["scaling"]],
            ),
        )
        parameters_bounds.add(
            param + "_" + str(i),
            min_bound=np.array([additional_key_settings[param]["min_bound"]]),
            max_bound=np.array([additional_key_settings[param]["max_bound"]]),
            interpolation=InterpolationType.CONSTANT,
            phase=i
        )
        parameters_init.add(
            key=param + "_" + str(i),
            initial_guess=np.array([additional_key_settings[param]["initial_guess"]]),
            phase=i,
        )

    return parameters, parameters_bounds, parameters_init


def set_x_bounds(bio_models, x_init, x_bounds):
    for model in bio_models.muscles_dynamics_model:
        muscle_name = model.muscle_name
        variable_bound_list = [model.name_dof[i] + "_" + muscle_name for i in range(len(model.name_dof))]

        starting_bounds, min_bounds, max_bounds = (
            model.standard_rest_values(),
            model.standard_rest_values(),
            model.standard_rest_values(),
        )

        for j in range(len(variable_bound_list)):
            if variable_bound_list[j] == "Cn_" + muscle_name:
                max_bounds[j] = 10
            elif variable_bound_list[j] == "F_" + muscle_name:
                max_bounds[j] = 1000
            elif variable_bound_list[j] == "Tau1_" + muscle_name or variable_bound_list[j] == "Km_" + muscle_name:
                max_bounds[j] = 1
            elif variable_bound_list[j] == "A_" + muscle_name:
                min_bounds[j] = 0

        starting_bounds_min = np.concatenate((starting_bounds, min_bounds, min_bounds), axis=1)
        starting_bounds_max = np.concatenate((starting_bounds, max_bounds, max_bounds), axis=1)

        for j in range(len(variable_bound_list)):
            x_bounds.add(
                variable_bound_list[j],
                min_bound=np.array([starting_bounds_min[j]]),
                max_bound=np.array([starting_bounds_max[j]]),
                interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
            )

        for j in range(len(variable_bound_list)):
            x_init.add(variable_bound_list[j], model.standard_rest_values()[j])

    return x_bounds, x_init


def prepare_ocp(
    model,
    final_time: float,
    pulse_width_values: list,
    key_parameter_to_identify,
    q_target,
    time,
    use_sx=False,
):
    # Problem parameters
    n_shooting = q_target.shape[0]
    q_at_node = DataExtraction.force_at_node_in_ocp(time, q_target, n_shooting, final_time)

    numerical_data_time_series, stim_idx_at_node_list = model.muscles_dynamics_model[0].get_numerical_data_time_series(
        n_shooting, final_time
    )

    # Dynamics definition
    dynamics = DynamicsList()
    dynamics.add(
        model.declare_model_variables,
        dynamic_function=model.muscle_dynamic,
        expand_dynamics=True,
        expand_continuity=False,
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        numerical_data_timeseries=numerical_data_time_series,
        with_contact=False,
        with_passive_torque=True,
    )

    # States bounds and initial guess
    x_bounds = BoundsList()
    x_init = InitialGuessList()
    x_bounds, x_init = set_x_bounds(model, x_init, x_bounds)
    q_x_bounds = model.bounds_from_ranges("q")
    q_x_bounds.min[0][0] = q_at_node[0] - 0.1
    q_x_bounds.max[0][0] = q_at_node[0] + 0.1
    q_x_bounds.min[0][1] = 0
    q_x_bounds.min[0][2] = 0
    q_x_bounds.max[0][1] = 5
    q_x_bounds.max[0][2] = 5
    # q_x_bounds.max[0][1] = np.deg2rad(180-44)  # Participant's joint limit
    # q_x_bounds.max[0][2] = np.deg2rad(180-44)  # Participant's joint limit
    qdot_x_bounds = model.bounds_from_ranges("qdot")
    qdot_x_bounds.min[0][0] = 0
    qdot_x_bounds.max[0][0] = 0.1

    x_bounds.add(key="q", bounds=q_x_bounds, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT)
    x_bounds.add(key="qdot", bounds=qdot_x_bounds, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT)

    # Controls bounds and initial guess
    u_bounds = BoundsList()
    u_init = InitialGuessList()
    if isinstance(model.muscles_dynamics_model[0], DingModelPulseWidthFrequency):
        for muscle in model.muscles_dynamics_model:
            u_init.add(
                key="last_pulse_width_" + muscle.muscle_name,
                initial_guess=np.array([pulse_width_values]),
                interpolation=InterpolationType.EACH_FRAME,
            )
            u_bounds.add(
                "last_pulse_width_" + muscle.muscle_name,
                min_bound=np.array([pulse_width_values]),
                max_bound=np.array([pulse_width_values]),
                interpolation=InterpolationType.EACH_FRAME,
            )

    # Add objective functions
    target = np.array(q_at_node)[np.newaxis, :]

    objective_functions = ObjectiveList()
    objective_functions.add(
        ObjectiveFcn.Lagrange.TRACK_STATE,
        key="q",
        weight=10000,
        target=target,
        node=Node.ALL,
        quadratic=True,
    )

    x_init.add(key="q", initial_guess=target, interpolation=InterpolationType.EACH_FRAME)

    # Parameters definition
    additional_key_settings = OcpFesIdMultibody.set_default_values(msk_model=model)

    additional_key_settings["tau1_rest_BIClong"]["function"] = lambda models, value: set_tau1_rest_all_model(models, value)
    additional_key_settings["tau2_BIClong"]["function"] = lambda models, value: set_tau2_all_model(models, value)
    additional_key_settings["km_rest_BIClong"]["function"] = lambda models, value: set_km_rest_all_model(models, value)
    additional_key_settings["a_scale_BIClong"]["function"] = lambda models, value: set_a_scale_all_model(models, value)
    additional_key_settings["pd0_BIClong"]["function"] = lambda models, value: set_pd0_all_model(models, value)
    additional_key_settings["pdt_BIClong"]["function"] = lambda models, value: set_pdt_all_model(models, value)
    additional_key_settings["tau1_rest_BICshort"]["function"] = lambda models, value: set_tau1_rest_all_model(models, value)
    additional_key_settings["tau2_BICshort"]["function"] = lambda models, value: set_tau2_all_model(models, value)
    additional_key_settings["km_rest_BICshort"]["function"] = lambda models, value: set_km_rest_all_model(models, value)
    additional_key_settings["a_scale_BICshort"]["function"] = lambda models, value: set_a_scale_all_model(models, value)
    additional_key_settings["pd0_BICshort"]["function"] = lambda models, value: set_pd0_all_model(models, value)
    additional_key_settings["pdt_BICshort"]["function"] = lambda models, value: set_pdt_all_model(models, value)

    parameters, parameters_bounds, parameters_init = OcpFesId.set_parameters(
        parameter_to_identify=key_parameter_to_identify,
        parameter_setting=additional_key_settings,
        use_sx=use_sx,
    )

    # Update models with parameters
    for param_key in parameters:
        if parameters[param_key].function:
            param_scaling = parameters[param_key].scaling.scaling
            param_reduced = parameters[param_key].cx
            fes_models_for_param_key = [
                model.muscles_dynamics_model[i] for i in range(len(model.muscles_dynamics_model)) if model.muscles_dynamics_model[i].muscle_name in param_key
            ]
            # reshaped_param = casadi.MX(param_reduced * param_scaling).T
            parameters[param_key].function(
                fes_models_for_param_key, param_reduced * param_scaling, **parameters[param_key].kwargs
            )

    model.parameters = parameters.mx  #this command is necessary to update parameters

    return OptimalControlProgram(
        bio_model=[model],
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
        control_type=ControlType.CONSTANT,
        use_sx=use_sx,
        n_threads=20,
        ode_solver=OdeSolver.RK4(n_integration_steps=10),
    )


def main(plot=True, total_cycles=1):
    # Get experimental data
    converter = C3dToQ("/home/mickaelbegon/Documents/Stage_Florine/Data/P05/p05_motion_50Hz_62.c3d")
    converter.frequency_stimulation = 50
    data_dict = converter.get_sliced_time_Q_rad()
    time = data_dict["time"]
    stim_time = data_dict["stim_time"]
    Q_rad = data_dict["q"]

    # Get pulse width values
    with open("../data_process/seeds_pulse_width.pkl", "rb") as f:
        data = pickle.load(f)
    pulse_width_list = data[62]  # Example pulse width values for each phase
    pulse_width_control = []
    for i in range(total_cycles):
        pulse_width_control.append([pulse_width_list[i]] * Q_rad[i].shape[0])

    pulse_width_control = [item for sublist in pulse_width_control for item in sublist]
    Q_rad = np.concatenate(Q_rad[:total_cycles])
    time = np.concatenate(time[:total_cycles])
    stim_time = np.concatenate(stim_time[:total_cycles])

    plt.plot(time, Q_rad)
    plt.scatter(stim_time, [0] * len(stim_time), label="Stimulus", color="green")
    plt.show()

    # Define MSK models
    biclong_model = DingModelPulseWidthFrequency(muscle_name="BIClong", sum_stim_truncation=10)
    bicshort_model = DingModelPulseWidthFrequency(muscle_name="BICshort", sum_stim_truncation=10)
    model = FesMskModel(
        name=None,
        biorbd_path="../model_msk/p05_scaling_scaled.bioMod",
        muscles_model=[biclong_model, bicshort_model],
        stim_time=list(np.round(stim_time, 2)),
        activate_force_length_relationship=True,
        activate_force_velocity_relationship=True,
        activate_passive_force_relationship=True,
        activate_residual_torque=False,
        external_force_set=None,  # External forces will be added later
    )

    final_time = np.round(time[-1], 2)

    ocp = prepare_ocp(
        model,
        final_time,
        pulse_width_control,
        key_parameter_to_identify=[
            "tau1_rest_BIClong",
            "tau2_BIClong",
            "km_rest_BIClong",
            "a_scale_BIClong",
            # "pd0_BIClong",
            # "pdt_BIClong",
            "tau1_rest_BICshort",
            "tau2_BICshort",
            "km_rest_BICshort",
            "a_scale_BICshort",
            # "pd0_BICshort",
            # "pdt_BICshort",
        ],
        q_target=Q_rad,
        time=time,
    )

    ocp.add_plot_penalty(CostType.ALL)
    sol = ocp.solve(Solver.IPOPT(_max_iter=10000, _linear_solver="ma57")) #, _tol=1e-12))
    sol.graphs(show_bounds=True)
    identified_parameters = sol.parameters
    print("Identified parameters:")
    for key, value in identified_parameters.items():
        print(f"{key}: {value}")

    sol_Q = sol.decision_states(to_merge=[SolutionMerge.NODES, SolutionMerge.PHASES])["q"][0]
    sol_time = sol.decision_time(to_merge=[SolutionMerge.NODES, SolutionMerge.PHASES]).T[0]

    # Passive torque plot:
    passive_torque_plot = True
    if passive_torque_plot:
        omega = sol.decision_states(to_merge=[SolutionMerge.NODES, SolutionMerge.PHASES])["qdot"][0]

        k1 = 0.24395358  # k=1
        k2 = 0.00103129
        k3 = 0.00194639
        k4 = 24.41585281
        kc1 = 0.105713656
        kc2 = 0.19654403
        theta_max = 2.27074652
        theta_min = 0.49997778
        def sigmoide(x):
            return 1 / (1 + exp(-x))
        #c = [-kc1 * np.exp(kc2 * (sol_Q[i] - theta_min)) + kc3 * np.exp(kc4 * (sol_Q[i] - theta_max)) for i in range(sol_Q.shape[0])]
        #c=0.1

        theta = sol_Q
        c = [(sigmoide((theta[i] - theta_max) / kc2) + sigmoide(-(theta[i] - theta_min) / kc1))
             for i in range(sol_Q.shape[0])]
        #c = [ 0.1 for i in range(sol_Q.shape[0])]
        low = [k1 * exp(-k2 * (theta[i] - theta_min)) * sigmoide(-(theta[i] - theta_min)) for i in range(sol_Q.shape[0])]
        high = [- k3 * exp(k4 * (theta[i] - theta_max)) * sigmoide(theta[i] - theta_max) for i in range(sol_Q.shape[0])]
        damping = [- c[i] * omega[i] for i in range(sol_Q.shape[0])]
        tau = [k1 * exp(-k2 * (theta[i] - theta_min)) * sigmoide(-(theta[i] - theta_min)) - k3 * exp(k4 * (theta[i] - theta_max)) * sigmoide(theta[i] - theta_max) - c[i] * omega[i]
               for i in range(sol_Q.shape[0])]
        #tau = [k1*exp(-k2 * (theta[i] - theta_min)) - k3 * exp(k4 * (theta[i] - theta_max)) - c * omega[i] for i in range(sol_Q.shape[0])]
        # s = [1 / (1 + exp(-(omega[i]))) for i in range(sol_Q.shape[0])]
        # tau = [k1 * exp(-k2 * (theta[i] - theta_min)) * (1 - s[i]) - k3 * exp(k4 * (theta[i] - theta_max)) * s[i] - (c * omega[i])
        #             for i in range(sol_Q.shape[0])]
        #tau = [k1*exp(k2-k2*(theta[i]-theta_min))**4 if omega[i]<0 else - k3*exp(k4*(theta[i]-theta_max))**4 for i in range(sol_Q.shape[0])]#elastic - c * omega

        plt.figure()
        sc = plt.scatter(theta, np.array(tau).reshape(len(tau)), c=omega, cmap='viridis', s=20)
        plt.scatter(theta, np.array(low).reshape(len(low)), c='blue', label='Low torque')
        plt.scatter(theta, np.array(high).reshape(len(high)), c='red', label='High torque')
        plt.scatter(theta, np.array(damping).reshape(len(damping)), c='orange', label='Damping torque')
        plt.colorbar(sc, label='ω (rad/s)')
        plt.axvline(theta_min, linestyle=':', label='Extension limit')
        plt.axvline(theta_max, linestyle=':', label='Flexion limit')
        plt.xlabel('Elbow angle θ (rad)')
        plt.ylabel('Passive torque τ (N·m)')
        plt.title('Passive Torque vs Angle with Variable Speed Profile')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    if plot:
        # Plot the simulation and identification results
        fig, ax = plt.subplots()
        ax.set_title("Identification")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("q (radian)")

        ax.plot(time, Q_rad, label="Experimental q ", color='blue')
        ax.scatter(list(np.round(stim_time, 2)), [0]*len(stim_time), label="Stimulus", color="green")

        ax.plot(sol_time, sol_Q, label="Identified q", color='red')
        ax.legend()
        plt.show()


if __name__ == "__main__":
    main(total_cycles=1)
