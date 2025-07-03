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

from examples.data_process.c3d_to_q import C3dToQ


def slicing_data(q: list[list], time: list[list], stim_time: list[list]):
    start = q[1][0]
    end_index = 0
    for i in range(100, len(q[0])):
        if q[0][i] < start:
            end = q[0][i]
            end_time = time[0][i]
            end_index = i
            break
    new_q = np.concatenate((q[0][:end_index], q[1][:]))
    new_stim_time = np.concatenate((stim_time[0], stim_time[1] - time[1][0] + time[0][end_index]))
    new_time = np.concatenate((time[0][:end_index], time[1][:] - time[1][0] + time[0][end_index]))
    # for j in range(len(q)-1):
    #     for i in range(50, len(q[j])):
    #         if q[j][i] < start:
    #             end = q[j][i]
    #             end_time = time[j][i]
    #             end_index = i
    #             break
    #     if j == 0:
    #         new_q = np.concatenate((q[j][:end_index], q[j+1][:]))
    #         new_stim_time = np.concatenate((stim_time[j], stim_time[j+1] - time[j+1][0] + time[j][end_index]))
    #         new_time = np.concatenate((time[j][:end_index], time[j+1][:] - time[j+1][0] + time[j][end_index]))
    #     else:
    #         new_q = np.concatenate((new_q, q[j + 1][:]))
    #         new_stim_time = np.concatenate((new_stim_time, stim_time[j + 1] - time[j + 1][0] + time[j][end_index]))
    #         new_time = np.concatenate((new_time, time[j + 1][:] - time[j + 1][0] + time[j][end_index]))

    return new_q, new_time, new_stim_time


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
        with_contact=False,
        with_passive_torque=True,
    )

    x_bounds, x_init = OcpFesMsk.set_x_bounds_fes(model)
    q_x_bounds = model.bounds_from_ranges("q")
    # q_x_bounds.min[0][0] = q_at_node[0]
    # q_x_bounds.max[0][0] = q_at_node[0]
    q_x_bounds.min[0] = [-1, -1, -1]
    # q_x_bounds.max[0][0] = np.deg2rad(180-44)
    # q_x_bounds.max[0][1] = np.deg2rad(180-44)
    # q_x_bounds.max[0][2] = np.deg2rad(180-44)
    qdot_x_bounds = model.bounds_from_ranges("qdot")
    # qdot_x_bounds.min[0][0] = 0
    # qdot_x_bounds.max[0][0] = 0
    x_bounds.add(key="q", bounds=q_x_bounds, phase=0)
    x_bounds.add(key="qdot", bounds=qdot_x_bounds, phase=0)

    u_bounds = BoundsList()  # Controls bounds
    u_init = InitialGuessList()  # Controls initial guess
    if isinstance(fes_model[0], DingModelPulseWidthFrequency):
        # u_init.add(
        #     key="tau",
        #     initial_guess=np.array([0]),
        #     phase=0,
        #     interpolation=InterpolationType.CONSTANT,
        # )
        # u_bounds.add(
        #     "tau",
        #     min_bound=np.array([0]),
        #     max_bound=np.array([3]),
        #     interpolation=InterpolationType.CONSTANT,
        # )
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
     #objective_functions.add(
     #    ObjectiveFcn.Lagrange.,
     #    key="q",
     #    weight=1000,
     #    target=target,
     #    node=Node.ALL,
     #    quadratic=True,
     #    index=[0],
     #)
    objective_functions.add(
        ObjectiveFcn.Mayer.MINIMIZE_STATE,
        key="q",
        weight=1000,
        target=target,
        node=Node.ALL,
        quadratic=True,
        index=[0],
    )
    # objective_functions.add(
    #     ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,
    #     key="tau",
    #     weight=100000,
    # )

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
    converter = C3dToQ("/home/mickaelbegon/Documents/Stage_Florine/lucie_50Hz_250-300-350-400-450x2_22mA.c3d")
    exp_data = converter.get_sliced_time_Q_rad()
    time = exp_data["time"]
    Q_rad = exp_data["q"]
    stim_time = exp_data["stim_time"]
    final_time = np.round(time[1][-1], 2)

    # plt.plot(time[0], Q_rad[0], label="before slicing", color="green", alpha=0.5)
    # plt.plot(time[1], Q_rad[1], color="green", alpha=0.5)
    # plt.scatter(stim_time[0], [0]*len(stim_time[0]), label="Stimulus", color="purple", alpha=0.5)
    # plt.scatter(stim_time[1], [0] * len(stim_time[1]), color="purple", alpha=0.5)

    Q_rad, time, stim_time = slicing_data(Q_rad, time, stim_time)

    #stim_time = np.concatenate((stim_time[0], stim_time[1]))
    #Q_rad = np.concatenate((Q_rad[0], Q_rad[1]))
    #time = np.concatenate((time[0], time[1]))

    # plt.plot(time, Q_rad, label="after slicing", color="blue", alpha=0.5)
    # plt.scatter(stim_time, [0]*len(stim_time), label="Stimulus", color="red", alpha=0.5)
    # plt.legend()
    # plt.show()

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
        stim_time=list(np.round(stim_time, 2)),
        activate_force_length_relationship=True,
        activate_force_velocity_relationship=True,
        activate_passive_force_relationship=True,
        # activate_residual_torque=True,
        activate_residual_torque=False,
        external_force_set=None,  # External forces will be added later
    )

    pulse_width_values_BIC_long = [0.00025] * 141 + [0.0003] * 180
    #pulse_width_values_BICshort = sim_data["last_pulse_width_BICshort"]
    pulse_width_values = {
        "last_pulse_width_BIClong": pulse_width_values_BIC_long,
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
    sol = ocp.solve(Solver.IPOPT(_max_iter=10000, _linear_solver="ma57"))
    sol.graphs(show_bounds=True)
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
        # sol.animate(viewer="pyorerun", n_frames=100)


if __name__ == "__main__":
    main()
