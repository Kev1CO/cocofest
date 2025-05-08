import numpy as np
import matplotlib.pyplot as plt

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

from cocofest.identification.identification_method import DataExtraction
from cocofest import (
    DingModelPulseWidthFrequency,
    FesMskModel,
    OcpFesId,
    OcpFesMsk,
    CustomObjective, OcpFesIdMultibody,
)

from examples.data_process.c3d_to_q import C3DToQ


def prepare_ocp(
    model,
    final_time,
    pulse_width_values: dict,
    key_parameter_to_identify,
    q_target,
    time,
    use_sx=False,
):
    fes_model = model.muscles_dynamics_model
    n_shooting = fes_model[0].get_n_shooting(final_time)

    q_at_node = DataExtraction.force_at_node_in_ocp(
        time, q_target, n_shooting, final_time
    )
    # plt.plot(np.linspace(0, 1.6, len(q_at_node)), q_at_node)
    # plt.show()
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

    target = np.array(q_at_node)[np.newaxis, :]
    objective_functions = ObjectiveList()
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_STATE,
        key="q",
        weight=1000,
        target=target,
        node=Node.ALL,
        quadratic=True,
        index=[0],
    )

    x_init.add(key="q", initial_guess=target, interpolation=InterpolationType.EACH_FRAME, phase=0)
    ocp_fes_id = OcpFesId()
    additional_key_settings = ocp_fes_id.set_default_values(model=model.muscles_dynamics_model[0])

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
                fes_model[i] for i in range(len(fes_model)) #if fes_model[i].muscle_name in param_key
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
    final_time = 4
    n_stim = 50
    stim_time = list(np.linspace(0, 1, n_stim + 1)[:-1]) + list(np.linspace(2.04, 3.04, n_stim + 1)[:-1])
    model_BIClong = DingModelPulseWidthFrequency(muscle_name="BIClong", sum_stim_truncation=10)
    # model_BIClong.a_scale = 4210
    # model_BIClong.tau1_rest = 0.054
    # model_BIClong.tau2 = 0.001
    # model_BIClong.km_rest = 0.159
    # model_BIClong.pd0 = 0.000118
    # model_BIClong.pdt = 0.000090
    #model_BICshort = DingModelPulseWidthFrequency(muscle_name="BICshort", sum_stim_truncation=10)
    #model_BICshort.a_scale = 4800
    #model_BICshort.tau1_rest = 0.05
    #model_BICshort.tau2 = 0.001
    #model_BICshort.km_rest = 0.100
    #model_BICshort.pd0 = 0.000100
    #model_BICshort.pdt = 0.000150

    model = FesMskModel(
        name=None,
        biorbd_path="../model_msk/arm26_biceps_1dof.bioMod",
        muscles_model=[model_BIClong],
        stim_time=stim_time,
        activate_force_length_relationship=True,
        activate_force_velocity_relationship=True,
        activate_passive_force_relationship=True,
        activate_residual_torque=False,
        external_force_set=None,  # External forces will be added later
    )

    converter = C3DToQ("C:\\Users\\flori_4ro0b8\\Documents\\Stage_S2M\\c3d_file\\essais_mvt_16.04.25\\Florine_mouv_50hz_250-300-350-400-450us_15mA_1s_1sr.c3d")
    Q_rad = converter.get_q_rad()
    Q_deg = converter.get_q_deg()
    time = converter.get_time()

    msk_info = {
        "with_residual_torque": False,
        "bound_type": "start_end",
        "bound_data": [[Q_deg[5]], [Q_deg[160]]],  # end à 20
    }
    time = np.array(time[5:377]) - time[5]
    Q_rad1 = np.array(Q_rad[5:161]) - Q_rad[5]
    Q_rad2 = np.array(Q_rad[161:377]) - Q_rad[205]
    for i in range(len(Q_rad1)):
        if Q_rad1[i] < 0:
            Q_rad1[i] = 0
    for i in range(len(Q_rad2)):
        if Q_rad2[i] < 0:
            Q_rad2[i] = 0
    Q_rad = np.concatenate((Q_rad1, Q_rad2))
    pulse_width_values_BIClong = [0.00025] * 100 + [0.0003] * 100
    #pulse_width_values_BICshort = sim_data["last_pulse_width_BICshort"]
    pulse_width_values = {
        "last_pulse_width_BIClong": pulse_width_values_BIClong,
        #"last_pulse_width_BICshort": pulse_width_values_BICshort,
    }
    # plt.plot(time, Q_rad)
    # plt.scatter(stim_time, [0]*len(stim_time), label="Stimulus", color="red")
    # plt.show()

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
    )

    ocp.add_plot_penalty(CostType.ALL)
    sol = ocp.solve(Solver.IPOPT(_max_iter=0, _tol=1e-12))
    identified_parameters = sol.parameters
    print("Identified parameters:")
    for key, value in identified_parameters.items():
        print(f"{key}: {value}")

    sol_time = sol.decision_time(to_merge=SolutionMerge.NODES).T[0]
    sol_Q = sol.decision_states(to_merge=SolutionMerge.NODES)["q"][0]

    if plot:
        # Plot the simulation and identification results
        fig, ax = plt.subplots()
        ax.set_title("Identification")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("q (radian)")

        ax.plot(sol_time, sol_Q, label="Identified q")
        ax.plot(time, Q_rad, label="Experimental q")
        ax.scatter(stim_time, [0]*len(stim_time), label="Stimulus", color="red")

        ax.legend()
        plt.show()
        # sol.graphs(show_bounds=True)
        sol.animate()


if __name__ == "__main__":
    main()
