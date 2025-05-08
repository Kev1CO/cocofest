import numpy as np
import matplotlib.pyplot as plt
import pickle

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
    CostType,
)

from cocofest import (
    DingModelPulseWidthFrequency,
    FesMskModel,
    OcpFesId,
    OcpFesMsk,
    CustomObjective, OcpFesIdMultibody,
)


def prepare_ocp_simulation(model: FesMskModel, final_time: float, msk_info: dict):
    muscle_models = model.muscles_dynamics_model
    n_shooting = muscle_models[0].get_n_shooting(final_time)
    numerical_data_time_series, stim_idx_at_node_list = muscle_models[0].get_numerical_data_time_series(
        n_shooting, final_time
    )
    dynamics = OcpFesMsk.declare_dynamics(
        model,
        numerical_time_series=numerical_data_time_series,
        with_contact=False
    )

    x_bounds, x_init = OcpFesMsk.set_x_bounds(model, msk_info)
    u_bounds, u_init = OcpFesMsk.set_u_bounds(model, msk_info["with_residual_torque"], max_bound=0.0006)

    objective_functions = ObjectiveList()
    objective_functions.add(
        CustomObjective.minimize_overall_muscle_force_production,
        custom_type=ObjectiveFcn.Lagrange,
        weight=1,
        quadratic=True,
    )
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, target=120, weight=5, node=Node.MID, quadratic=True,
                            key='q')

    model = OcpFesMsk.update_model(model, parameters=ParameterList(use_sx=False), external_force_set=None)

    return OptimalControlProgram(
        bio_model=[model],
        dynamics=dynamics,
        n_shooting=n_shooting,
        phase_time=final_time,
        objective_functions=objective_functions,
        x_init=x_init,
        x_bounds=x_bounds,
        u_init=u_init,
        u_bounds=u_bounds,
        control_type=ControlType.CONSTANT,
        use_sx=False,
        n_threads=20,
    )


def simulate_data(model: FesMskModel, msk_info: dict, final_time: float):
    ocp = prepare_ocp_simulation(model, final_time, msk_info=msk_info)
    sol = ocp.solve(Solver.IPOPT(_max_iter=1000))
    state = sol.decision_states(to_merge=SolutionMerge.NODES)
    Q = state["q"]
    time = sol.decision_time(to_merge=SolutionMerge.NODES).T[0]
    last_pulse_width_BIClong = sol.decision_controls(to_merge=SolutionMerge.NODES)["last_pulse_width_BIClong"][0]
    last_pulse_width_BICshort = sol.decision_controls(to_merge=SolutionMerge.NODES)["last_pulse_width_BICshort"][0]

    data = {
        "time": time,
        "q": Q[0],
        "last_pulse_width_BICshort": last_pulse_width_BICshort,
        "last_pulse_width_BIClong": last_pulse_width_BIClong,
    }
    return data


def save_in_pkl(data, pkl_path):
    with open(pkl_path, "wb") as file:
        pickle.dump(data, file)


def read_pkl(pkl_path):
    with open(pkl_path, "rb") as file:
        data = pickle.load(file)
    return data

def prepare_ocp(
    model,
    final_time,
    pulse_width_values: dict,
    key_parameter_to_identify,
    q_target,
    use_sx=False,
):
    fes_model = model.muscles_dynamics_model
    n_shooting = fes_model[0].get_n_shooting(final_time)
    numerical_data_time_series, stim_idx_at_node_list = fes_model[0].get_numerical_data_time_series(
        n_shooting, final_time
    )

    dynamics = OcpFesMsk.declare_dynamics(
        model,
        numerical_time_series=numerical_data_time_series,
        with_contact=False
    )

    x_bounds, x_init = OcpFesMsk.set_x_bounds_fes(model)
    q_x_bounds = model.bounds_from_ranges("q")
    qdot_x_bounds = model.bounds_from_ranges("qdot")
    x_bounds.add(key="q", bounds=q_x_bounds, phase=0)
    x_bounds.add(key="qdot", bounds=qdot_x_bounds, phase=0)

    u_bounds = BoundsList()  # Controls bounds
    u_init = InitialGuessList()  # Controls initial guess
    if isinstance(fes_model[0], DingModelPulseWidthFrequency):
        for muscle in model.muscles_dynamics_model:
            control_bounds = pulse_width_values["last_pulse_width_" + muscle.muscle_name]
            u_init.add(
                key="last_pulse_width_" + muscle.muscle_name,
                initial_guess=np.array([control_bounds]),
                phase=0,
                interpolation=InterpolationType.EACH_FRAME,
            )
            u_bounds.add(
                "last_pulse_width_" + muscle.muscle_name,
                min_bound=np.array([control_bounds]),
                max_bound=np.array([control_bounds]),
                interpolation=InterpolationType.EACH_FRAME,
            )

    target = np.array(q_target)[np.newaxis, :]
    objective_functions = ObjectiveList()
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_STATE,
        key="q",
        weight=1,
        target=target,
        node=Node.ALL,
        quadratic=True,
        index=[0],
    )

    x_init.add(key="q", initial_guess=target, interpolation=InterpolationType.EACH_FRAME, phase=0)
    additional_key_settings = OcpFesIdMultibody.set_default_values(msk_model=model)

    parameters, parameters_bounds, parameters_init = OcpFesId.set_parameters(
        parameter_to_identify=key_parameter_to_identify,
        parameter_setting=additional_key_settings,
        use_sx=use_sx,
    )

    model = OcpFesMsk.update_model(model, parameters=parameters, external_force_set=None)

    for param_key in parameters:
        if parameters[param_key].function:
            param_scaling = parameters[param_key].scaling.scaling
            param_reduced = parameters[param_key].cx
            fes_model_for_param_key = [
                fes_model[i] for i in range(len(fes_model)) if fes_model[i].muscle_name in param_key
            ][0]
            parameters[param_key].function(
                fes_model_for_param_key, param_reduced * param_scaling, **parameters[param_key].kwargs
            )

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


def main(plot=True):
    final_time = 1.6
    n_stim = 33
    stim_time = list(np.linspace(0, 1, n_stim + 1)[:-1])
    model_BIClong = DingModelPulseWidthFrequency(muscle_name="BIClong", sum_stim_truncation=10)
    # model_BIClong.a_scale = 5000
    # model_BIClong.tau1_rest = 0.07
    # model_BIClong.tau2 = 0.002
    # model_BIClong.km_rest = 0.200
    # model_BIClong.pd0 = 0.000200
    # model_BIClong.pdt = 0.000250
    model_BICshort = DingModelPulseWidthFrequency(muscle_name="BICshort", sum_stim_truncation=10)
    #model_BICshort.a_scale = 4800
    #model_BICshort.tau1_rest = 0.05
    #model_BICshort.tau2 = 0.001
    #model_BICshort.km_rest = 0.100
    #model_BICshort.pd0 = 0.000100
    #model_BICshort.pdt = 0.000150

    model = FesMskModel(
        name=None,
        biorbd_path="../model_msk/arm26_allbiceps_1dof.bioMod",
        muscles_model=[model_BIClong, model_BICshort],
        stim_time=stim_time,
        activate_force_length_relationship=True,
        activate_force_velocity_relationship=True,
        activate_passive_force_relationship=True,
        activate_residual_torque=False,
        external_force_set=None,  # External forces will be added later
    )

    msk_info = {
        "with_residual_torque": False,
        "bound_type": "start_end",
        "bound_data": [[20], [20]],
    }

    #sim_data = simulate_data(model, msk_info, final_time)
    sim_data = read_pkl("simulation_data/simulation_data_multibody_default.pkl")
    pulse_width_values_BIClong = sim_data["last_pulse_width_BIClong"]
    pulse_width_values_BICshort = sim_data["last_pulse_width_BICshort"]
    pulse_width_values = {
        "last_pulse_width_BIClong": pulse_width_values_BIClong,
        "last_pulse_width_BICshort": pulse_width_values_BICshort,
    }

    ocp = prepare_ocp(
        model,
        final_time,
        pulse_width_values,
        key_parameter_to_identify=[
            "tau1_rest_BICshort",
            "tau2_BICshort",
            "km_rest_BICshort",
            "a_scale_BICshort",
            "pd0_BICshort",
            "pdt_BICshort",
            "tau1_rest_BIClong",
            "tau2_BIClong",
            "km_rest_BIClong",
            "a_scale_BIClong",
            "pd0_BIClong",
            "pdt_BIClong",
        ],
        q_target=sim_data["q"],
    )

    ocp.add_plot_penalty(CostType.ALL)
    sol = ocp.solve(Solver.IPOPT(_max_iter=1000))
    identified_parameters = sol.parameters
    print("Identified parameters:")
    for key, value in identified_parameters.items():
        print(f"{key}: {value}")

    sol_time = sol.decision_time(to_merge=SolutionMerge.NODES).T[0]
    sol_Q = sol.decision_states(to_merge=SolutionMerge.NODES)["q"][0]

    # Plot the simulation and identification results
    fig, ax = plt.subplots()
    ax.set_title("Identification")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("q (radian)")

    ax.plot(sol_time, sol_Q, label="Identified q")
    ax.plot(sim_data["time"], sim_data["q"], label="Simulated q")

    ax.legend()
    plt.show()
    # sol.graphs(show_bounds=True)
    sol.animate()


if __name__ == "__main__":
    main()