from typing import final

from bioptim import FatigueList, ConfigureProblem, DynamicsFunctions, DynamicsList, DynamicsFcn, PhaseDynamics, \
    OptimalControlProgram, BoundsList, InitialGuessList, OdeSolver, ControlType, ObjectiveList, ObjectiveFcn, Node, \
    CostType, Solver, SolutionMerge, BiorbdModel, InterpolationType, VariableScaling, ParameterList, \
    PhaseTransitionList, PhaseTransitionFcn
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

def set_parameters(parameters_to_identify: list, additional_key_settings: dict,n_phase, parameters_bounds, parameters_init):
    for i in range(n_phase):
        for param in parameters_to_identify:

            parameters_bounds.add(
                param,
                min_bound=np.array([additional_key_settings[param]["min_bound"]]),
                max_bound=np.array([additional_key_settings[param]["max_bound"]]),
                interpolation=InterpolationType.CONSTANT,
                phase=i
            )
            parameters_init.add(
                key=param,
                initial_guess=np.array([additional_key_settings[param]["initial_guess"]]),
                phase=i,
            )

    return parameters_bounds, parameters_init



def prepare_ocp(model_path,
    final_time: list,
    key_parameter_to_identify,
    q_target,
    bounds_list: list,
    use_sx=False,
    ):
    additional_key_settings = {
        "k1": {
            "initial_guess": 1,
            "min_bound": 0.001,
            "max_bound": 10,
            "function": MuscleDrivenPassiveTorque.set_k1,
            "scaling": 1
        },
        "k2": {
            "initial_guess": 1,
            "min_bound": 0.001,
            "max_bound": 10,
            "function": MuscleDrivenPassiveTorque.set_k2,
            "scaling": 1
        },
        "k3": {
            "initial_guess": 1,
            "min_bound": 0.001,
            "max_bound": 10,
            "function": MuscleDrivenPassiveTorque.set_k3,
            "scaling": 1
        },
        "k4": {
            "initial_guess": 1,
            "min_bound": 0.001,
            "max_bound": 100,
            "function": MuscleDrivenPassiveTorque.set_k4,
            "scaling": 1
        },
        "kc1": {
            "initial_guess": 0.1,
            "min_bound": 0.01,
            "max_bound": 2,
            "function": MuscleDrivenPassiveTorque.set_kc1,
            "scaling": 1
        },
        "kc2": {
            "initial_guess": 0.1,
            "min_bound": 0.01,
            "max_bound": 2,
            "function": MuscleDrivenPassiveTorque.set_kc2,
            "scaling": 1
        },
        "kc3": {
            "initial_guess": 1,
            "min_bound": 0.1,
            "max_bound": 10,
            "function": MuscleDrivenPassiveTorque.set_kc3,
            "scaling": 1
        },
        "kc4": {
            "initial_guess": 1,
            "min_bound": 0.01,
            "max_bound": 10,
            "function": MuscleDrivenPassiveTorque.set_kc4,
            "scaling": 1
        },
        "theta_c": {
            "initial_guess": 5,
            "min_bound": 0.1,
            "max_bound": 100,
            "function": MuscleDrivenPassiveTorque.set_theta_c,
            "scaling": 1
        },
        "theta_max": {
            "initial_guess": 5,
            "min_bound": 2,
            "max_bound": 4,
            "function": MuscleDrivenPassiveTorque.set_theta_max,
            "scaling": 1
        },
        "theta_min": {
            "initial_guess": 5,
            "min_bound": 0,
            "max_bound": 0.5,  #TODO:
            "function": MuscleDrivenPassiveTorque.set_theta_min,
            "scaling": 1
        },
        "e_min": {
            "initial_guess": 1,
            "min_bound": 0.01,
            "max_bound": 15,
            "function": MuscleDrivenPassiveTorque.set_e_min,
            "scaling": 1
        },
        "e_max": {
            "initial_guess": 5,
            "min_bound": 0.01,
            "max_bound": 5,
            "function": MuscleDrivenPassiveTorque.set_e_max,
            "scaling": 1
        },
    }
    n_shooting = []
    dynamics = DynamicsList()
    x_bounds, x_init = BoundsList(), InitialGuessList()
    u_init = InitialGuessList()
    u_bounds = BoundsList()
    objective_functions = ObjectiveList()
    phase_transitions = PhaseTransitionList()
    models = []

    ocp_fes_id = OcpFesId()
    parameters, parameters_bounds, parameters_init = ocp_fes_id.set_parameters(
        parameter_to_identify=key_parameter_to_identify,
        parameter_setting=additional_key_settings,
        use_sx=use_sx,
    )

    for i in range(len(q_target)):

        n_shooting.append(q_target[i].shape[0])

        muscle_driven = MuscleDrivenPassiveTorque()
        dynamics.add(
            muscle_driven.declare_dynamics,
            dynamic_function=muscle_driven.muscles_driven,
            expand_dynamics=True,
            phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
            numerical_data_timeseries=None,
            with_passive_torque=True,
            phase=i,
        )

        models.append(BiorbdModel(model_path[i], parameters=parameters))

        q_x_bounds = models[i].bounds_from_ranges("q")
        qdot_x_bounds = models[i].bounds_from_ranges("qdot")
        init_q = np.array(q_target[i].tolist() + [0])
        x_init.add(key="q", initial_guess=init_q.reshape(1, len(init_q)), interpolation=InterpolationType.EACH_FRAME, phase=i)

        x_bounds.add(key="q", bounds=q_x_bounds, phase=i)
        q_x_bounds.min[0] = [q_target[i][0], 0, 0]
        q_x_bounds.max[0] = [q_target[i][0], 4, 4]
        x_bounds.add(key="qdot", bounds=qdot_x_bounds, phase=i)

        u_bounds.add(key="muscles", min_bound=[0, 0], max_bound=[0, 0], phase=i)

        u_init["muscles"] = [0, 0]

        target = np.array(q_target[i])[np.newaxis, :]

        if bounds_list[i]=="up":
            objective_functions.add(
                ObjectiveFcn.Lagrange.MINIMIZE_STATE,
                key="q",
                weight=2,
                target=target,
                node=Node.ALL_SHOOTING,
                quadratic=True,
                index=[0],
                phase=i,
            )
        else:
            objective_functions.add(
                ObjectiveFcn.Lagrange.MINIMIZE_STATE,
                key="q",
                weight=1,
                target=target,
                node=Node.ALL_SHOOTING,
                quadratic=True,
                index=[0],
                phase=i,
            )
        if i < len(q_target) - 1:
            phase_transitions.add(PhaseTransitionFcn.DISCONTINUOUS, phase_pre_idx=i)

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
        control_type=ControlType.CONSTANT, #None si on ne declare pas de control
        use_sx=use_sx,
        n_threads=20,
        ode_solver=OdeSolver.RK4(n_integration_steps=4),
        phase_transitions=phase_transitions,
    )

def main(model_paths, data_low, data_up, bounds_list, plot=True):
    q_rad = data_low["q_rad"]
    time = data_low["time"]
    stim_time = data_low["stim_time"]
    q_rad, time_low = slicing(q_rad, time, stim_time)

    q_target = q_rad + [data_up["q_rad"][0]] + [data_up["q_rad"][1]]

    time_low = time_low
    time_low[0] = np.array(time_low[0]) - time_low[0][0]  # Start at 0
    for i in range(len(time_low) - 1):
        time_low[i+1] = time_low[i+1] + (time_low[i][-1] - time_low[i+1][0])
    time_up = [np.array(data_up["time"][0]) + (time_low[-1][-1] - data_up["time"][0][0])]
    time_up += [np.array(data_up["time"][1]) + (time_up[0][-1] - data_up["time"][1][0])]
    time = time_low + time_up

    final_time = []
    for i in range(len(q_target)):
        final_time.append(time[i][-1] - time[i][0])

    for i in range(len(q_target)):
        plt.plot(time[i], q_target[i])
    plt.show()

    ocp = prepare_ocp(
        model_path=model_paths,
        final_time=final_time,
        key_parameter_to_identify=[
            "k1",
            "k2",
            "k3",
            "k4",
            "kc1",
            "kc2",
            # "e_min",
            # "e_max",
            # "kc3",
            # "kc4",
            "theta_max",
            "theta_min"
        ],
        q_target=q_target,
        bounds_list=bounds_list,
    )

    ocp.add_plot_penalty(CostType.ALL)
    sol = ocp.solve(Solver.IPOPT(_max_iter=1000, _linear_solver="ma57"))
    sol.graphs(show_bounds=True)
    identified_parameters = sol.parameters
    print("Identified parameters:")
    for key, value in identified_parameters.items():
        print(f"{key}: {value}")

    sol_Q = sol.decision_states(to_merge=[SolutionMerge.NODES, SolutionMerge.PHASES])["q"][0]
    sol_time = sol.decision_time(to_merge=[SolutionMerge.NODES, SolutionMerge.PHASES]).T[0]

    if plot:
        # Plot the simulation and identification results
        fig, ax = plt.subplots()
        ax.set_title("Identification")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("q (radian)")

        ax.plot(sol_time, sol_Q, label="Identified q")
        for i in range(len(q_target)):
            ax.plot(time[i], q_target[i], label=f"Experimental q {i+1}")

        ax.legend()
        plt.show()

if __name__ == "__main__":
    converter = C3dToQ("/home/mickaelbegon/Documents/Stage_Florine/Data/P05/p05_motion_50Hz_62.c3d")
    converter.frequency_stimulation = 50
    time = converter.get_time()
    q_rad = converter.get_q_rad()
    stim_time = converter.get_sliced_stim_time()["stim_time"]
    data_low = {"time": time, "q_rad": q_rad, "stim_time": stim_time}

    converter = C3dToQ("/home/mickaelbegon/Documents/Stage_Florine/Data/P05/p05_limit_upper.c3d")
    time = converter.get_time()
    q_rad = converter.get_q_rad()

    data_up = {"time": time, "q_rad": q_rad}

    converter = C3dToQ("/home/mickaelbegon/Documents/Stage_Florine/Data/P05/p05_limit_upper1.c3d")
    time = converter.get_time()
    q_rad = converter.get_q_rad()

    data_up["time"] = [data_up["time"]] + [time]
    data_up["q_rad"] = [data_up["q_rad"]] + [q_rad]

    models = ["../model_msk/p05_scaling_scaled.bioMod", "../model_msk/p05_scaling_scaled.bioMod", "../model_msk/p05_scaling_scaled.bioMod", "../model_msk/p05_scaling_scaled.bioMod", "../model_msk/p05_scaling_scaled.bioMod", "../model_msk/p05_scaling_scaled_modified.bioMod", "../model_msk/p05_scaling_scaled_modified.bioMod"]

    main(data_low=data_low, data_up=data_up, model_paths=models, bounds_list = ["low", "low", "low", "low", "low", "up", "up"], plot=True)