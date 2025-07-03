from bioptim import FatigueList, ConfigureProblem, DynamicsFunctions, DynamicsList, DynamicsFcn, PhaseDynamics, \
    OptimalControlProgram, BoundsList, InitialGuessList, OdeSolver, ControlType, ObjectiveList, ObjectiveFcn, Node, \
    CostType, Solver, SolutionMerge, BiorbdModel, InterpolationType
from cocofest import FesMskModel, DingModelPulseWidthFrequency, OcpFesMsk, OcpFesId
from cocofest.identification.identification_method import DataExtraction

import numpy as np
import matplotlib.pyplot as plt
from examples.data_process.c3d_to_q import C3dToQ
from cocofest.models.muscle_driven_passive_torque import MuscleDrivenPassiveTorque



def slicing(q, time, stim_time):
    sliced_q = []
    sliced_time = []
    for i in range(len(stim_time)):
        last_stim = stim_time[i][-1]
        next_stim = stim_time[i + 1][0] if i + 1 < len(stim_time) else time[-1]
        time_between = np.where((time >= last_stim) & (time <= next_stim))[0]
        sliced_q.append(q[time_between])
        sliced_time.append(time[time_between])

    return sliced_q, sliced_time


def prepare_ocp(model_path,
    final_time: float,
    key_parameter_to_identify,
    q_target,
    time,
    use_sx=False,
    ):

    n_shooting = q_target.shape[0]

    dynamics = DynamicsList()
    dynamics.add(
        MuscleDrivenPassiveTorque.declare_dynamics,
        dynamic_function=MuscleDrivenPassiveTorque.muscles_driven,
        expand_dynamics=True,
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        numerical_data_timeseries=None,
        with_passive_torque=True,
    )

    additional_key_settings = {
        "k1": {
            "initial_guess": 5,
            "min_bound": 0.1,#1,
            "max_bound": 6, #10,
            "function": MuscleDrivenPassiveTorque.set_k1,
            "scaling": 1
        },
        "k2": {
            "initial_guess": 1,
            "min_bound": 0.1,#1,
            "max_bound": 6, #10,
            "function": MuscleDrivenPassiveTorque.set_k2,
            "scaling": 1
        },
        "k3": {
            "initial_guess": 2,
            "min_bound": 0.1,#1,
            "max_bound": 6, #10,
            "function": MuscleDrivenPassiveTorque.set_k3,
            "scaling": 1
        },
        "k4": {
            "initial_guess": 1,
            "min_bound": 0.1,#1,
            "max_bound": 6, #10,
            "function": MuscleDrivenPassiveTorque.set_k4,
            "scaling": 1
        },
        "kc1": {
            "initial_guess": 1,
            "min_bound": 0.1,#1,
            "max_bound": 10, #10,
            "function": MuscleDrivenPassiveTorque.set_kc1,
            "scaling": 1
        },
        "kc2": {
            "initial_guess": 1,
            "min_bound": 0.01,#1,
            "max_bound": 6, #10,
            "function": MuscleDrivenPassiveTorque.set_kc2,
            "scaling": 1
        },
        "kc3": {
            "initial_guess": 1,
            "min_bound": 0.1,  # 1,
            "max_bound": 10,  # 10,
            "function": MuscleDrivenPassiveTorque.set_kc3,
            "scaling": 1
        },
        "kc4": {
            "initial_guess": 1,
            "min_bound": 0.01,  # 1,
            "max_bound": 10,  # 10,
            "function": MuscleDrivenPassiveTorque.set_kc4,
            "scaling": 1
        },
        "theta_c": {
            "initial_guess": 5,
            "min_bound": 0.1,#1,
            "max_bound": 100, #10,
            "function": MuscleDrivenPassiveTorque.set_theta_c,
            "scaling": 1
        },
        "theta_max": {
            "initial_guess": 5,
            "min_bound": 0,  # 1,
            "max_bound": 4,  # 10,
            "function": MuscleDrivenPassiveTorque.set_theta_max,
            "scaling": 1
        },
        "theta_min": {
            "initial_guess": 5,
            "min_bound": 0,  # 1,
            "max_bound": 4,  # 10,
            "function": MuscleDrivenPassiveTorque.set_theta_min,
            "scaling": 1
        },
        "e_min": {
            "initial_guess": 1,
            "min_bound": 0.01,#1,
            "max_bound": 15, #15,
            "function": MuscleDrivenPassiveTorque.set_e_min,
            "scaling": 1
        },
        "e_max": {
            "initial_guess": 5,
            "min_bound": 0.01,#1,
            "max_bound": 5, #15,
            "function": MuscleDrivenPassiveTorque.set_e_max,
            "scaling": 1
        },
    }

    parameters, parameters_bounds, parameters_init = OcpFesId.set_parameters(
        parameter_to_identify=key_parameter_to_identify,
        parameter_setting=additional_key_settings,
        use_sx=use_sx,
    )

    model = BiorbdModel(model_path, parameters=parameters)

    x_bounds, x_init = BoundsList(), InitialGuessList()
    q_x_bounds = model.bounds_from_ranges("q")
    qdot_x_bounds = model.bounds_from_ranges("qdot")
    init_q = np.array(q_target.tolist() + [0])
    x_init.add(key="q", initial_guess=init_q.reshape(1, len(init_q)), interpolation=InterpolationType.EACH_FRAME)

    x_bounds.add(key="q", bounds=q_x_bounds)
    q_x_bounds.min[0] = [q_target[0], -3, -3]
    q_x_bounds.max[0] = [q_target[0], 4, 4]

    x_bounds.add(key="qdot", bounds=qdot_x_bounds)

    # tau_min, tau_max, tau_init = -1.0, 1.0, 0.0
    u_bounds = BoundsList()
    # u_bounds["tau"] = [tau_min] * model.nb_tau, [tau_max] * model.nb_tau
    u_bounds.add(key="muscles", min_bound=[0, 0], max_bound=[0, 0])

    u_init = InitialGuessList()
    u_init["muscles"] = [0, 0]

    target = np.array(q_target)[np.newaxis, :]
    objective_functions = ObjectiveList()
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_STATE,
        key="q",
        weight=1000,
        target=target,
        node=Node.ALL_SHOOTING,
        quadratic=True,
        index=[0],
    )


    return OptimalControlProgram(
        bio_model=model,
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
    converter = C3dToQ("/home/mickaelbegon/Documents/Stage_Florine/P07/p07_motion_50Hz_15.c3d")
    converter.frequency_stimulation = 50
    time = converter.get_time()
    q_rad = converter.get_q_rad()
    stim_time = converter.get_sliced_stim_time()["stim_time"]

    plt.plot(time, q_rad)
    plt.show()

    q_rad, time = slicing(q_rad, time, stim_time)

    for i in range(len(q_rad)):
        plt.plot(time[i], q_rad[i])
        plt.scatter(stim_time[i], [0] * len(stim_time[i]))
    plt.show()

    time = time[0]
    time = time - time[0]  # Start time at 0
    q_rad = q_rad[0]

    final_time = time[-1]

    model_path = "../model_msk/p07_scaling_scaled.bioMod"

    ocp = prepare_ocp(
        model_path,
        final_time,
        key_parameter_to_identify=[
            "k1",
            "k2",
            "k3",
            "k4",
            # "kc1",
            # "kc2",
            # "e_min",
            # "e_max",
            # "kc3",
            # "kc4",
            # "theta_max",
            # "theta_min"
        ],
        q_target=q_rad,
        time=time,
    )

    ocp.add_plot_penalty(CostType.ALL)
    sol = ocp.solve(Solver.IPOPT(_max_iter=1000, _linear_solver="ma57"))
    # sol.graphs(show_bounds=True)
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
        ax.plot(time, q_rad, label="Experimental q")

        ax.legend()
        plt.show()

if __name__ == "__main__":
    main()