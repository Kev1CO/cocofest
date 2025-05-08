import numpy as np
import matplotlib.pyplot as plt
import casadi

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
    ParameterList,
    Solver,
    SolutionMerge,
    CostType, DynamicsList, PhaseDynamics, PhaseTransitionList, PhaseTransitionFcn, VariableScaling, SelectionMapping,
    Dependency, ConstraintList,
)

from cocofest.identification.identification_method import DataExtraction
from cocofest import (
    DingModelPulseWidthFrequency,
    FesMskModel,
    OcpFesId,
    OcpFesMsk,
    CustomObjective, OcpFesIdMultibody, CustomConstraint,
)

from examples.data_process.c3d_to_q import C3DToQ


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


def set_x_bounds(bio_models, i, x_init, x_bounds):
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
                max_bounds[j] = model.fmax
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
                phase=i,
                interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
            )

        for j in range(len(variable_bound_list)):
            x_init.add(variable_bound_list[j], model.standard_rest_values()[j], phase=i)

    return x_bounds, x_init


def prepare_ocp(
    models,
    final_time: list,
    pulse_width_values: dict,
    key_parameter_to_identify,
    q_target,
    time,
    use_sx=False,
    n_phase: int = 2,
):

    n_shooting = ()
    q_at_node = []
    numerical_data_time_series = []
    stim_idx_at_node_list = []
    dynamics = DynamicsList()
    u_bounds = BoundsList()
    u_init = InitialGuessList()
    target = []
    objective_functions = ObjectiveList()
    phase_transition = PhaseTransitionList()
    x_bounds = BoundsList()
    x_init = InitialGuessList()

    for i in range(n_phase):

        # Problem parameters
        n_shooting += (models[i].muscles_dynamics_model[0].get_n_shooting(final_time[i]),)
        phase_time = np.array(time[i]) - time[i][0]
        q_at_node.append(DataExtraction.force_at_node_in_ocp(phase_time, q_target[i], n_shooting[i], final_time[i]))
        numerical_data_time, stim_idx_at_node = models[i].muscles_dynamics_model[0].get_numerical_data_time_series(
            n_shooting[i], final_time[i]
        )
        numerical_data_time_series.append(numerical_data_time)
        stim_idx_at_node_list.append(stim_idx_at_node)

        # Dynamics definition
        dynamics.add(
            models[i].declare_model_variables,
            dynamic_function=models[i].muscle_dynamic,
            expand_dynamics=True,
            expand_continuity=False,
            phase=i,
            phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
            numerical_data_timeseries=numerical_data_time_series[i],
            with_contact=False,
        )

        # States bounds and initial guess
        x_bounds, x_init = set_x_bounds(models[i], i, x_init, x_bounds)
        q_x_bounds = models[i].bounds_from_ranges("q")
        q_x_bounds.min[0][0] = q_at_node[i][0]
        q_x_bounds.max[0][0] = q_at_node[i][0]
        qdot_x_bounds = models[i].bounds_from_ranges("qdot")
        if i == 0:
            qdot_x_bounds.min[0][0] = 0
            qdot_x_bounds.max[0][0] = 0

        x_bounds.add(key="q", bounds=q_x_bounds, phase=i, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT)
        x_bounds.add(key="qdot", bounds=qdot_x_bounds, phase=i, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT)

        # Controls bounds and initial guess
        if isinstance(models[i].muscles_dynamics_model[0], DingModelPulseWidthFrequency):
            for muscle in models[i].muscles_dynamics_model:
                control_bounds = pulse_width_values[i]["last_pulse_width_" + muscle.muscle_name]
                u_init.add(
                    key="last_pulse_width_" + muscle.muscle_name,
                    initial_guess=np.array([control_bounds]),
                    phase=i,
                    interpolation=InterpolationType.EACH_FRAME,
                )
                u_bounds.add(
                    "last_pulse_width_" + muscle.muscle_name,
                    min_bound=np.array([control_bounds]),
                    max_bound=np.array([control_bounds]),
                    interpolation=InterpolationType.EACH_FRAME,
                    phase=i,
                )

        # Add objective functions
        target.append(np.array(q_at_node[i])[np.newaxis, :])
        objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_STATE,
            key="q",
            weight=1,
            target=target[i],
            node=Node.ALL,
            quadratic=True,
            index=[0],
            phase=i,
        )

        x_init.add(key="q", initial_guess=target[i], interpolation=InterpolationType.EACH_FRAME, phase=i)

        # Phase transition
        phase_transition.add(PhaseTransitionFcn.DISCONTINUOUS, phase_pre_idx=i)

    # Parameters definition
    ocp_fes_id = OcpFesId()
    additional_key_settings = ocp_fes_id.set_default_values(model=models[0].muscles_dynamics_model[0])

    additional_key_settings["tau1_rest"]["function"] = lambda models, value: set_tau1_rest_all_model(models, value)
    additional_key_settings["tau2"]["function"] = lambda models, value: set_tau2_all_model(models, value)
    additional_key_settings["km_rest"]["function"] = lambda models, value: set_km_rest_all_model(models, value)
    additional_key_settings["a_scale"]["function"] = lambda models, value: set_a_scale_all_model(models, value)
    additional_key_settings["pd0"]["function"] = lambda models, value: set_pd0_all_model(models, value)
    additional_key_settings["pdt"]["function"] = lambda models, value: set_pdt_all_model(models, value)

    parameters, parameters_bounds, parameters_init = ocp_fes_id.set_parameters(
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
                models[i].muscles_dynamics_model[0] for i in range(len(models))
            ]
            # reshaped_param = casadi.MX(param_reduced * param_scaling).T
            parameters[param_key].function(
                fes_models_for_param_key, param_reduced * param_scaling, **parameters[param_key].kwargs
            )

    models[0].parameters = parameters.mx
    models[1].parameters = parameters.mx




    #models[0] = OcpFesMsk.update_model(models[0], parameters=parameters, external_force_set=None)
    #models[1] = OcpFesMsk.update_model(models[1], parameters=parameters, external_force_set=None)

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
        control_type=ControlType.CONSTANT,
        use_sx=use_sx,
        n_threads=20,
        ode_solver=OdeSolver.RK4(n_integration_steps=10),
        phase_transitions=phase_transition,
    )


def main(plot=True):
    final_time = (1.55, 1.70)
    n_stim =(50, 50)
    stim_time = []
    n_phase = 2
    stim_time0 = list(np.linspace(0, 1, n_stim[0] + 1)[:-1])
    stim_time1 = list(np.linspace(1.55, 2.55, n_stim[1] + 1)[:-1])

    # Define model
    model_BIClong0 = DingModelPulseWidthFrequency(muscle_name="BIClong", sum_stim_truncation=10)
    #model_BICshort = DingModelPulseWidthFrequency(muscle_name="BICshort", sum_stim_truncation=10)

    model0 = FesMskModel(
        name=None,
        biorbd_path="../model_msk/arm26_biceps_1dof.bioMod",
        muscles_model=[model_BIClong0],
        stim_time=stim_time0,
        activate_force_length_relationship=True,
        activate_force_velocity_relationship=True,
        activate_passive_force_relationship=True,
        activate_residual_torque=False,
        external_force_set=None,  # External forces will be added later
    )

    model_BIClong1 = DingModelPulseWidthFrequency(muscle_name="BIClong", sum_stim_truncation=10)
    model1 = FesMskModel(
        name=None,
        biorbd_path="../model_msk/arm26_biceps_1dof.bioMod",
        muscles_model=[model_BIClong1],
        stim_time=stim_time1,
        activate_force_length_relationship=True,
        activate_force_velocity_relationship=True,
        activate_passive_force_relationship=True,
        activate_residual_torque=False,
        external_force_set=None
    )  # External forces will be added later

    model = [model0, model1]

    # Get experimental Q_rad
    converter = C3DToQ("C:\\Users\\flori_4ro0b8\\Documents\\Stage_S2M\\c3d_file\\essais_mvt_16.04.25\\Florine_mouv_50hz_250-300-350-400-450us_15mA_1s_1sr.c3d")
    Q_rad = converter.get_q_rad()
    Q_deg = converter.get_q_deg()
    time = converter.get_time()

    time0 = np.array(time[5:161]) - time[5]
    time1 = np.array(time[204:377]) - (time[204] - time0[-1])
    Q_rad0 = np.array(Q_rad[5:161])
    Q_rad1 = np.array(Q_rad[204:377])
    for i in range(len(Q_rad0)):
        if Q_rad0[i] < 0:
            Q_rad0[i] = 0
    for i in range(len(Q_rad1)):
        if Q_rad1[i] < 0:
            Q_rad1[i] = 0
    Q_rad = [Q_rad0, Q_rad1]
    time = [time0, time1]

    # plt.plot(time0, Q_rad0, color='blue')
    # plt.plot(time1, Q_rad1, color='red')
    # plt.show()

    pulse_width_values_BIClong0 = [0.00025] * 155
    pulse_width_values_BIClong1 = [0.0003] * 170
    #pulse_width_values_BICshort = sim_data["last_pulse_width_BICshort"]
    pulse_width_values0 = {
        "last_pulse_width_BIClong": pulse_width_values_BIClong0,
        #"last_pulse_width_BICshort": pulse_width_values_BICshort,
    }
    pulse_width_values1 = {"last_pulse_width_BIClong": pulse_width_values_BIClong1,}

    pulse_width_values = [pulse_width_values0, pulse_width_values1]

    #plt.plot(time, Q_rad)
    #plt.scatter(stim_time, [0]*len(stim_time), label="Stimulus", color="red")
    #plt.show()

    ocp = prepare_ocp(
        model,
        final_time,
        pulse_width_values,
        key_parameter_to_identify=[
            "tau1_rest",
            "tau2",
            "km_rest",
            "a_scale",
            "pd0",
            "pdt",
        ],
        q_target=Q_rad,
        time=time,
        n_phase=n_phase,
    )

    ocp.add_plot_penalty(CostType.ALL)
    sol = ocp.solve(Solver.IPOPT(_max_iter=1000, _tol=1e-12))  #ocp.nlp[0.parameters -> un seul param ou list de param tau1 ? sinon ajouter _i
    sol.graphs(show_bounds=True)
    identified_parameters = sol.parameters
    print("Identified parameters:")
    for key, value in identified_parameters.items():
        print(f"{key}: {value}")

    sol_time0 = sol.decision_time(to_merge=SolutionMerge.NODES)[0]
    sol_time1 = sol.decision_time(to_merge=SolutionMerge.NODES)[1]
    sol_Q0 = sol.decision_states(to_merge=SolutionMerge.NODES)[0]["q"][0]
    sol_Q1 = sol.decision_states(to_merge=SolutionMerge.NODES)[1]["q"][0]

    sol_time = np.concatenate((sol_time0, sol_time1))
    sol_Q = np.concatenate((sol_Q0, sol_Q1))

    if plot:
        # Plot the simulation and identification results
        fig, ax = plt.subplots()
        ax.set_title("Identification")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("q (radian)")

        ax.plot(sol_time, sol_Q, label="Identified q")
        ax.plot(np.concatenate((time0, time1)), np.concatenate((Q_rad0, Q_rad1)), label="Experimental q")
        #ax.plot(time, Q_rad, label="Experimental q")
        #ax.scatter(stim_time, [0]*len(stim_time), label="Stimulus", color="red")

        ax.legend()
        plt.show()
        # sol.graphs(show_bounds=True)
        sol.animate()

    print(sol_time)


if __name__ == "__main__":
    main()
