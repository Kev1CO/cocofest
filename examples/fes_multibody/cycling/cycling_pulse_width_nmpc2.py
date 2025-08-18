"""
This example will perform an optimal control program moving time horizon for a hand cycling motion driven by FES.
"""

import os
import pickle

from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.ma.extras import average

from bioptim import (
    Axis,
    BiorbdModel,
    BoundsList,
    ConstraintList,
    ConstraintFcn,
    ContactType,
    CostType,
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
    Node,
    VariableScalingList,
)
from cocofest import (
    CustomObjective,
    DingModelPulseWidthFrequencyWithFatigue,
    FesMskModel,
    inverse_kinematics_cycling,
    OcpFesMsk,
    FesNmpcMsk,
)


class MyCyclicNMPC(FesNmpcMsk):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nodes_per_cycle = self.cycle_len * (
            self.nlp[0].dynamics_type.ode_solver.polynomial_degree + 1
            if isinstance(self.nlp[0].dynamics_type.ode_solver, OdeSolver.COLLOCATION)
            else 1
        )
        self.initial_wheel_position = self.nlp[0].x_bounds["q"].min[-1][0]
        self.actual_cycle = 0
        self.pedal_turn_in_one_cycle = 2 * np.pi  # One mhe cycle simulates on pedal turn
        self.polynomial_order = (
            self.nlp[0].dynamics_type.ode_solver.polynomial_degree + 1
            if isinstance(self.nlp[0].dynamics_type.ode_solver, OdeSolver.COLLOCATION)
            else 1
        )
        self.debugg_bounds = False


    def advance_window_bounds_states(self, sol, n_cycles_simultaneous=None, **extra):
        # --- Get states results --- #
        states = sol.decision_states(to_merge=SolutionMerge.NODES)
        states_keys = states.keys()
        # --- Store previous state bounds for debugg purpose --- #
        if self.debugg_bounds:
            self.previous_bounds = {}
            for key in states_keys:
                xb = self.nlp[0].x_bounds[key]
                self.previous_bounds[key] = {
                    "min": xb.min[:, :self.nodes_per_cycle].copy(),
                    "max": xb.max[:, :self.nodes_per_cycle].copy(),
                }

        # --- States are bounded to match the last node of the cycle to ensure continuity between window --- #
        for key in states_keys:
            for i in range(states[key].shape[0]):
                # --- Only doing wheel to prevent overconstraining the system --- #
                if key == "q" or key == "qdot":
                    if i == 2:
                        self.nlp[0].x_bounds[key].min[i, 0] = states[key][i][self.nodes_per_cycle]
                        self.nlp[0].x_bounds[key].max[i, 0] = states[key][i][self.nodes_per_cycle]
                    if key == "q" and i == 2:
                        self.nlp[0].x_bounds[key].min[i, 0] = self.nlp[0].x_bounds["q"].min[
                                                                  i, 0] + self.pedal_turn_in_one_cycle  # - 0.05
                        self.nlp[0].x_bounds[key].max[i, 0] = self.nlp[0].x_bounds["q"].max[
                                                                  i, 0] + self.pedal_turn_in_one_cycle  # + 0.05
                else:
                    self.nlp[0].x_bounds[key].min[i, 0] = states[key][i][self.nodes_per_cycle]
                    self.nlp[0].x_bounds[key].max[i, 0] = states[key][i][self.nodes_per_cycle]
        # --- Inform the past cycle stimulation time into the new one --- #
        self.update_stim()
        return True

    def advance_window_initial_guess_states(self, sol, n_cycles_simultaneous=None):
        # --- Get states results --- #
        states = sol.decision_states(to_merge=SolutionMerge.NODES)
        states_keys = states.keys()
        cyclical_keys = [s for s in states if any(s.startswith(prefix) for prefix in ("Cn_", "F_", "q", "qdot"))]
        continuous_keys = [s for s in states if any(s.startswith(prefix) for prefix in ("A_", "Tau1_", "Km_"))]
        # --- Set initial guesses for cyclical and continuous states --- #
        for key in states_keys:
            for i in range(states[key].shape[0]):
                if key in cyclical_keys:
                    if key == "q" and i == states[key].shape[0] - 1:
                        # Special case for the wheel position
                        self.set_init_cyclical_wheel(states, key, i)
                    else:
                        self.set_init_cyclical(states, key, i)
                elif key in continuous_keys:
                    self.set_init_continuous(states, key, i)
        self._correct_init_guess_to_fit_bounds(
            corrected_input="states")  # This function is called to move init guess within the bounds if not in bounds

        # --- Print bounds and initial guesses for debugg purpose --- #
        if self.debugg_bounds:
            for key in states.keys():
                # if key=="q" or key=="qdot":
                self.plot_initial_guess(data=self.nlp[0].x_init[key].init,
                                    current_bounds=self.nlp[0].x_bounds[key],
                                    past_bounds=self.previous_bounds[key],
                                    key=key)
        return True


    def advance_window_initial_guess_controls(self, sol, n_cycles_simultaneous=None):
        # --- Get control results --- #
        controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
        controls_keys = controls.keys()

        # --- Store previous control bounds for debugg purpose --- #
        if self.debugg_bounds:
            self.previous_bounds = {}
            for key in controls_keys:
                ub = self.nlp[0].u_bounds[key]
                self.previous_bounds[key] = {
                    "min": ub.min[:, :self.nodes_per_cycle].copy(),
                    "max": ub.max[:, :self.nodes_per_cycle].copy(),
                }

        # --- Set intial guess for controls --- #
        for key in controls.keys():
            self.set_init_cyclical(controls, key, 0, False)
        self._correct_init_guess_to_fit_bounds(corrected_input="controls")  # This function is called to move init guess within the bounds if not in bounds

        # --- Print bounds and initial guesses for debugg purpose --- #
        if self.debugg_bounds:
            for key in controls_keys:
                self.plot_initial_guess(data=self.nlp[0].u_init[key].init,
                                        current_bounds=self.nlp[0].u_bounds[key],
                                        past_bounds=self.previous_bounds[key],
                                        key=key)
        return True

    def set_init_continuous(self, states, key, i):
        n_plus_one_cycles = states[key][i][self.nodes_per_cycle:-1]
        last_cycle = states[key][i][-self.nodes_per_cycle-1:]
        delta = n_plus_one_cycles[-1] - last_cycle[0]
        shifted_last_cycle = states[key][i][-self.nodes_per_cycle - 1:] + delta
        values = np.concatenate((n_plus_one_cycles, shifted_last_cycle))
        self.nlp[0].x_init[key].init[:, :] = values
        return True

    def set_init_cyclical(self, data, key, i, state=True):
        n_plus_one_cycles = data[key][i][self.nodes_per_cycle:-1]
        last_cycle = data[key][i][-self.nodes_per_cycle - 1:]
        values = np.concatenate((n_plus_one_cycles, last_cycle))
        if state:
            self.nlp[0].x_init[key].init[i, :] = values
        else:
            self.nlp[0].u_init[key].init[i, :] = values
        return True

    def set_init_cyclical_wheel(self, states, key, i):
        shifted_n_plus_one_cycles = states[key][i][self.nodes_per_cycle:-1] + self.pedal_turn_in_one_cycle
        last_cycle = states[key][i][-self.nodes_per_cycle - 1:]
        values = np.concatenate((shifted_n_plus_one_cycles, last_cycle))
        self.nlp[0].x_init[key].init[i, :] = values
        return True

    def _correct_init_guess_to_fit_bounds(self, corrected_input="states"):
        corrected_data_input = self.nlp[0].x_init if corrected_input == "states" else self.nlp[0].u_init if corrected_input == "controls" else None
        corrected_bound_input = self.nlp[0].x_bounds if corrected_input == "states" else self.nlp[0].u_bounds if corrected_input == "controls" else None
        if corrected_data_input is None or corrected_bound_input is None:
            raise ValueError("Input must be either 'states' or 'controls'.")
        # This function is called to move init guess within the bounds if not in bounds
        for key in corrected_data_input.keys():
            data = corrected_data_input[key].init
            bounds = corrected_bound_input[key]
            for i in range(data.shape[0]):
                if bounds.min.shape == data.shape:
                    min_bounds = bounds.min[:, :][i]
                    max_bounds = bounds.max[:, :][i]
                else:
                    min_bounds = [bounds.min[i][0], *[bounds.min[i][1]] * (data.shape[1] - 2), bounds.min[i][2]]
                    max_bounds = [bounds.max[i][0], *[bounds.max[i][1]] * (data.shape[1] - 2), bounds.max[i][2]]

                for j in range(data.shape[1]):
                    if data[:, :][i][j] < min_bounds[j]:
                        corrected_data_input[key].init[i, j] = min_bounds[j]
                    if data[:, :][i][j] > max_bounds[j]:
                        corrected_data_input[key].init[i, j] = max_bounds[j]


    def plot_initial_guess(self, data, current_bounds, past_bounds, key):
        for i in range(data.shape[0]):
            if current_bounds.min.shape == data.shape:
                current_min_bounds = current_bounds.min[:, :][i]
                current_max_bounds = current_bounds.max[:, :][i]
            else:
                current_min_bounds = [current_bounds.min[i][0], *[current_bounds.min[i][1]] * (data.shape[1] - 2), current_bounds.min[i][2]]
                current_max_bounds = [current_bounds.max[i][0], *[current_bounds.max[i][1]] * (data.shape[1] - 2), current_bounds.max[i][2]]

            if past_bounds["min"].shape[1] == self.nodes_per_cycle:
                past_min_bounds = past_bounds["min"][i]
                past_max_bounds = past_bounds["max"][i]
            else:
                past_min_bounds = [past_bounds["min"][i][0], *[past_bounds["min"][i][1]] * (self.nodes_per_cycle - 1)]
                past_max_bounds = [past_bounds["max"][i][0], *[past_bounds["max"][i][1]] * (self.nodes_per_cycle - 1)]

            fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]})
            fig.suptitle("Bounds and initial guess of " + key + " " + "index n°" + str(i), size=14, weight="bold")

            current_time_index = list(np.linspace(0, self.n_cycles_simultaneous, data[:,:][i].shape[0]))
            axs[0].plot(current_time_index,data[:, :][i], label="Intial guess", color="black", lw=3)
            axs[0].plot(current_time_index,current_min_bounds, linestyle="-", label="Current bound", color="grey", lw=1)
            axs[0].plot(current_time_index,current_max_bounds, linestyle="-", color="grey", lw=1)

            past_time_index = np.linspace(-1, 0, self.nodes_per_cycle)
            axs[0].plot(past_time_index, past_min_bounds, linestyle="-", label="Pervious bound", color="lightcoral", lw=1)
            axs[0].plot(past_time_index, past_max_bounds, linestyle="-", color="lightcoral", lw=1)

            labeled = False
            for j in range(data.shape[1]):
                if data[:, :][i][j] < current_min_bounds[j] or data[:, :][i][j] > current_max_bounds[j]:
                    axs[0].scatter(current_time_index[j], data[:, :][i][j], color="red", s=10, label="out of bounds" if not labeled else None)
                    labeled = True
            axs[0].legend()

            axs[1].plot(past_time_index, past_max_bounds, linestyle="-", color="lightcoral", lw=1)
            axs[1].set_ylim([0, 1])
            axs[1].axvspan(-1, 0, color='lightcoral', alpha=0.5)
            axs[1].text(-0.5, 0.5, 'Cycle n-1', ha='center', va='center', size=15, weight="bold")

            for j in range(self.n_cycles_simultaneous):
                axs[1].axvspan(j, j + 1, color='lightgreen', alpha=0.5 - 0.05 * j)
                axs[1].text(j + 0.5, 0.5, f'Cycle n{f"+{j}" if j > 0 else ""}', ha='center', va='center', size=15, weight="bold")

            axs[1].set_ylim(0, 1)
            axs[1].set_yticks([])
            axs[1].set_xlabel("Time (s)", size=15, weight="bold" )

            plt.subplots_adjust(wspace=0, hspace=0)
            plt.show()


# WIP
# def task_performance_coefficient_cost_fun(controller: PenaltyController):
#     F0 = vertcat(
#             *[controller.model.muscles_dynamics_model[i].fmax for i in range(len(controller.model.muscles_dynamics_model))]
#         )
#     force_state = vertcat(
#         *[controller.states["F_" + controller.model.muscles_dynamics_model[i].muscle_name].cx for i in
#           range(len(controller.model.muscles_dynamics_model))]
#     )
#     fatigue_state = vertcat(
#             *[controller.states["A_" + controller.model.muscles_dynamics_model[i].muscle_name].cx for i in range(len(controller.model.muscles_dynamics_model))]
#         )
#     rested_state = vertcat(
#             *[controller.model.muscles_dynamics_model[i].a_scale for i in range(len(controller.model.muscles_dynamics_model))]
#         )
#
#     return (1-(fatigue_state/rested_state)) - (force_state/F0)

#-------------------#
#   OCP functions   #
#-------------------#

def prepare_nmpc(
    model: BiorbdModel | FesMskModel,
    mhe_info: dict,
    cycling_info: dict,
    simulation_conditions: dict,
):
    # --- Initialize parameters from dictionaries --- #
        # --- MHE info --- #
    cycle_duration = mhe_info["cycle_duration"]
    cycle_len = mhe_info["cycle_len"]
    n_cycles_to_advance = mhe_info["n_cycles_to_advance"]
    n_cycles_simultaneous = mhe_info["n_cycles_simultaneous"]
    ode_solver = mhe_info["ode_solver"]
    use_sx = mhe_info["use_sx"]
    window_n_shooting = cycle_len * n_cycles_simultaneous
    window_cycle_duration = cycle_duration * n_cycles_simultaneous
        # --- Cycling info --- #
    turn_number = cycling_info["turn_number"]
    pedal_config = cycling_info["pedal_config"]
    external_force = cycling_info["resistive_torque"]
        # --- Cost function info --- #
    minimize_force = simulation_conditions["minimize_force"]
    minimize_fatigue = simulation_conditions["minimize_fatigue"]
    minimize_control = simulation_conditions["minimize_control"]
    cost_fun_weight = simulation_conditions["cost_fun_weight"]
        # --- Pickle file info --- #
    initial_guess_path = simulation_conditions["init_guess_file_path"]

    # --- Set dynamics --- #
        # --- External force numerical time series --- #
    numerical_time_series, external_force_set = set_external_forces(
        n_shooting=window_n_shooting, external_force_dict=external_force, force_name="external_torque"
    )
        # --- Stimulation instant numerical time series --- #
    numerical_data_time_series, stim_idx_at_node_list = model.muscles_dynamics_model[0].get_numerical_data_time_series(
        window_n_shooting, window_cycle_duration
    )
    numerical_time_series.update(numerical_data_time_series)
        # --- Dynamics --- #
    dynamics = set_dynamics(model=model, numerical_time_series=numerical_time_series, ode_solver=ode_solver)

    # --- Set states --- #
        # --- Set q (position and speed) initial guesses --- #
    x_init = set_q_qdot_init(n_shooting=window_n_shooting,
                             pedal_config=pedal_config,
                             turn_number=turn_number,
                             ode_solver=ode_solver,
                             init_file_path=initial_guess_path)

        # --- Set bounds and FES initial guesses --- #
    x_bounds, x_init = set_x_bounds(
        model=model,
        x_init=x_init,
        n_shooting=window_n_shooting,
        n_cycle_simultanous=n_cycles_simultaneous,
        ode_solver=ode_solver,
        init_file_path = initial_guess_path,
    )

        # --- Set states scaling --- #
    x_scaling = set_x_scaling(bio_model=model)  # Less efficient

    # --- Set controls --- #
    u_bounds, u_init, u_scaling = set_u_bounds_and_init(model, window_n_shooting, init_file_path=initial_guess_path)

    # --- Set constraints --- #
    constraints = set_constraints(model,
                                  end_first_cycle_node=cycle_len,
                                  pedal_target=x_init["q"].init[2][0],
                                  pedal_speed_target=x_init["qdot"].init[2][0])

    # --- Set objective --- #
    objective_functions = set_objective_functions(minimize_force, minimize_fatigue, minimize_control, cost_fun_weight,
                                                  target=x_init["q"].init[2][-1],
                                                  )

    # --- Update model for resisitive torque --- #
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
        x_init=x_init,
        # x_scaling=x_scaling,
        u_bounds=u_bounds,
        u_init=u_init,
        u_scaling=u_scaling,
        n_threads=32,
        use_sx=use_sx,
    )


def set_external_forces(n_shooting, external_force_dict, force_name):
    external_force_set = ExternalForceSetTimeSeries(nb_frames=n_shooting)
    external_force_array = np.array(external_force_dict["torque"])
    reshape_values_array = np.tile(external_force_array[:, np.newaxis], (1, n_shooting))
    external_force_set.add_torque(
        segment=external_force_dict["Segment_application"], values=reshape_values_array, force_name=force_name
    )  # warning forloop different force name
    numerical_time_series = {"external_forces": external_force_set.to_numerical_time_series()}
    return numerical_time_series, external_force_set


def set_dynamics(model, numerical_time_series, ode_solver):
    dynamics = DynamicsList()
    dynamics.add(
        dynamics_type=model.declare_model_variables,
        dynamic_function=model.muscle_dynamic,
        expand_dynamics=True,
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        numerical_data_timeseries=numerical_time_series,
        contact_type=[ContactType.RIGID_EXPLICIT],  # empty list for no contact
        phase=0,
        ode_solver=ode_solver,
    )
    return dynamics


def set_q_qdot_init(n_shooting: int, pedal_config: dict, turn_number: int, ode_solver: OdeSolver, init_file_path: str)-> InitialGuessList:
    x_init = InitialGuessList()
    if init_file_path:
        with open(init_file_path, "rb") as file:
            data = pickle.load(file)
        q_guess = data["states"]["q"]
        qdot_guess = data["states"]["qdot"]
        x_init.add("q", q_guess, interpolation=InterpolationType.ALL_POINTS)
        x_init.add("qdot", qdot_guess, interpolation=InterpolationType.ALL_POINTS)
    else:
        # --- Chose the biorbd model to init the inverse kinematics --- #
        biorbd_model_path = "../../model_msk/Wu_Shoulder_Model_mod_kev_inverse_dyn.bioMod"
        # biorbd_model_path = "../../model_msk/simplified_UL_Seth_2D_cycling_for_inverse_kinematics.bioMod"
        n_shooting = n_shooting * (ode_solver.polynomial_degree + 1) if isinstance(ode_solver,
                                                                                   OdeSolver.COLLOCATION) else n_shooting
        # --- Run inverse kinematics --- #
        q_guess, qdot_guess, qddotguess = inverse_kinematics_cycling(
            biorbd_model_path,
            n_shooting,
            x_center=pedal_config["x_center"],
            y_center=pedal_config["y_center"],
            radius=pedal_config["radius"],
            ik_method="trf",
            cycling_number=turn_number,
        )
        # --- Set q and qdot initial guesses values obtained by inverse kinematics --- #
        if isinstance(ode_solver, OdeSolver.COLLOCATION):
            x_init.add("q", q_guess, interpolation=InterpolationType.ALL_POINTS)
            x_init.add("qdot", qdot_guess, interpolation=InterpolationType.ALL_POINTS)
        elif isinstance(ode_solver, OdeSolver.RK1 | OdeSolver.RK2 | OdeSolver.RK4):
            x_init.add("q", q_guess, interpolation=InterpolationType.EACH_FRAME)
            x_init.add("qdot", qdot_guess, interpolation=InterpolationType.EACH_FRAME)
        else:
            raise RuntimeError("ode_solver must be COLLOCATION or RK4")

    return x_init


def set_x_bounds(model, x_init: InitialGuessList, n_shooting: int, n_cycle_simultanous: int, ode_solver: OdeSolver, init_file_path: str)-> tuple[BoundsList, InitialGuessList]:
    # --- Set interpolation type according to ode_solver type --- #
    interpolation_type = InterpolationType.EACH_FRAME
    if isinstance(ode_solver, OdeSolver.COLLOCATION):
        n_shooting = n_shooting * (ode_solver.polynomial_degree + 1)
        interpolation_type = InterpolationType.ALL_POINTS

    # --- Initialize default FES bounds and intial guess --- #
    x_bounds, x_init_fes = OcpFesMsk.set_x_bounds_fes(model)

    # --- Getting initial guesses from initialization file if entered --- #
    states = None
    if init_file_path:
        with open(init_file_path, "rb") as file:
            data = pickle.load(file)
            states = data["states"]

    # --- Setting FES initial guesses --- #
    for key in x_init_fes.keys():
        initial_guess = states[key] if states else np.array([[x_init_fes[key].init[0][0]] * (n_shooting + 1)])
        x_init.add(key=key, initial_guess=initial_guess, phase=0, interpolation=interpolation_type)

    # --- Setting q bounds --- #
    q_x_bounds = model.bounds_from_ranges("q")

        # --- First: enter general bound values in radiant --- #
    arm_q = [0, 1.5]  # Arm min_max q bound in radiant
    forarm_q = [0.5, 2.5]  # Forarm min_max q bound in radiant
    slack = 0.05  # Wheel rotation slack
    wheel_q = [x_init["q"].init[2][-1] - slack, x_init["q"].init[2][0] + slack]  # Wheel min_max q bound in radiant

        # --- Second: set general bound values in radiant, CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT mandatory for qdot --- #
    q_x_bounds.min[0] = [arm_q[0], arm_q[0], arm_q[0]]
    q_x_bounds.max[0] = [arm_q[1], arm_q[1], arm_q[1]]
    q_x_bounds.min[1] = [forarm_q[0], forarm_q[0], forarm_q[0]]
    q_x_bounds.max[1] = [forarm_q[1], forarm_q[1], forarm_q[1]]
    q_x_bounds.min[2] = [x_init["q"].init[2][0], wheel_q[0] - 2, x_init["q"].init[2][-1]-slack]
    q_x_bounds.max[2] = [x_init["q"].init[2][0], wheel_q[1] + 2, x_init["q"].init[2][-1]+slack]

    x_bounds.add(key="q", bounds=q_x_bounds, phase=0, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT)

    # --- Setting qdot bounds --- #
    qdot_x_bounds = model.bounds_from_ranges("qdot")

        # --- First: enter general bound values in radiant --- #
    arm_qdot = [-10, 10]  # Arm min_max qdot bound in radiant
    forarm_qdot = [-14, 10]  # Forarm min_max qdot bound in radiant
    wheel_qdot = [-2 * np.pi - 2, -2 * np.pi + 2]  # Wheel min_max qdot bound in radiant

        # --- Second: set general bound values in radiant, CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT mandatory for qdot --- #
    qdot_x_bounds.min[0] = [arm_qdot[0], arm_qdot[0], arm_qdot[0]]
    qdot_x_bounds.max[0] = [arm_qdot[1], arm_qdot[1], arm_qdot[1]]
    qdot_x_bounds.min[1] = [forarm_qdot[0], forarm_qdot[0], forarm_qdot[0]]
    qdot_x_bounds.max[1] = [forarm_qdot[1], forarm_qdot[1], forarm_qdot[1]]
    qdot_x_bounds.min[2] = [-2*np.pi, wheel_qdot[0], wheel_qdot[0]]
    qdot_x_bounds.max[2] = [-2*np.pi, wheel_qdot[1], wheel_qdot[1]]

    x_bounds.add(key="qdot", bounds=qdot_x_bounds, phase=0, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT)

    return x_bounds, x_init


def set_x_scaling(bio_model)->VariableScalingList:
    x_scaling = VariableScalingList()
    model_list = bio_model.muscles_dynamics_model
    prefix_key_list = ["Cn_", "A_", "Tau1_", "Km_"]
    # scaling_value_list = [1/100, 1000, 1/100, 1/10]
    scaling_value_list = [100, 1/1000, 100, 10]
    for i in range(len(model_list)):
        for j in range(len(prefix_key_list)):
            key = prefix_key_list[j] + model_list[i].muscle_name
            x_scaling.add(key=key, scaling=[scaling_value_list[j]])
    return x_scaling


def set_u_bounds_and_init(bio_model, n_shooting, init_file_path):
    u_bounds, u_init = OcpFesMsk.set_u_bounds_fes(bio_model)
    u_init = InitialGuessList()  # Controls initial guess
    models = bio_model.muscles_dynamics_model
    if init_file_path:
        with open(init_file_path, "rb") as file:
            data = pickle.load(file)
        u_guess = data["controls"]
    else:
        u_guess = None
    for model in models:
        key = "last_pulse_width_" + str(model.muscle_name)
        if u_guess:
            initial_guess = data["controls"][key]
        else:
            initial_guess = np.array([[model.pd0] * n_shooting])
        u_init.add(
            key=key,
            initial_guess=initial_guess,
            phase=0,
            interpolation=InterpolationType.EACH_FRAME,
        )
    u_scaling = VariableScalingList()
    for model in bio_model.muscles_dynamics_model:
        key = "last_pulse_width_" + str(model.muscle_name)
        u_scaling.add(key=key, scaling=[1 / 400])

    return (
        u_bounds,
        u_init,
        u_scaling,
    )


def set_constraints(bio_model, end_first_cycle_node, pedal_target, pedal_speed_target):
    constraints = ConstraintList()
    # --- Constraining wheel center position to a fix position --- #
    constraints.add(
        ConstraintFcn.TRACK_MARKERS_VELOCITY,
        node=Node.START,
        marker_index=bio_model.marker_index("wheel_center"),
        axes=[Axis.X, Axis.Y],
    )
    constraints.add(
        ConstraintFcn.SUPERIMPOSE_MARKERS,
        first_marker="wheel_center",
        second_marker="global_wheel_center",
        node=Node.START,
        axes=[Axis.X, Axis.Y],
    )

    return constraints


def set_objective_functions(minimize_force, minimize_fatigue, minimize_control, cost_fun_weight, target):
    objective_functions = ObjectiveList()
    # --- Set main cost function --- #
    if minimize_force:
        objective_functions.add(
            CustomObjective.minimize_overall_muscle_force_production,
            custom_type=ObjectiveFcn.Lagrange,
            node=Node.ALL,
            weight=10000 * cost_fun_weight[0],
            quadratic=True,
        )
    if minimize_fatigue:
        objective_functions.add(
            CustomObjective.minimize_overall_muscle_fatigue,
            custom_type=ObjectiveFcn.Lagrange,
            node=Node.ALL,
            weight=10000 * cost_fun_weight[1],
            quadratic=True,
        )
    if minimize_control:
        objective_functions.add(
            CustomObjective.minimize_overall_stimulation_charge,
            custom_type=ObjectiveFcn.Lagrange,
            node=Node.ALL,
            weight=10000 * cost_fun_weight[2],
            quadratic=True,
        )

    # --- Set cost function for initial_guess ocp --- #
    if not any([minimize_force, minimize_fatigue, minimize_control]):
        objective_functions.add(
            ObjectiveFcn.Mayer.MINIMIZE_STATE,
            key="q",
            index=2,
            node=Node.END,
            weight=1e6,
            target=target,
            quadratic=True,
        )

    # --- Set regulation cost function --- #
    else:
        objective_functions.add(
            ObjectiveFcn.Mayer.MINIMIZE_STATE,
            key="q",
            index=2,
            node=Node.END,
            weight=1e-2,
            target=target,
            quadratic=True,
        )

    # WIP
    # objective_functions.add(
    #     task_performance_coefficient_cost_fun,
    #     custom_type=ObjectiveFcn.Mayer,
    #     node=Node.ALL,
    #     weight=1e-3,
    #     quadratic=True,
    # )

    return objective_functions


def updating_model(model: FesMskModel, external_force_set, parameters=None)->FesMskModel:
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
    return model


#--------------------------#
#   Simulation functions   #
#--------------------------#

def set_fes_model(model_path, stim_time):
    # Set FES model (set to Ding et al. 2007 + fatigue, for now)
    dummy_biomodel = BiorbdModel(model_path)
    muscle_name_list = dummy_biomodel.muscle_names
    muscles_model = [DingModelPulseWidthFrequencyWithFatigue(
                    muscle_name=muscle,
                    sum_stim_truncation=6
                ) for muscle in muscle_name_list]

    parameter_dict = {"Delt_ant":{"Fmax": 144, "a_scale": 2988.4, "alpha_a":-3.3 * 10e-2},
                      "Delt_post":{"Fmax": 29, "a_scale": 692.4, "alpha_a":-2.7 * 10e-2},
                      "Biceps": {"Fmax": 130, "a_scale": 3769.8, "alpha_a": -3.8 * 10e-2},
                      "Triceps": {"Fmax": 323, "a_scale": 5357.3, "alpha_a": -3.4 * 10e-2},
                      }

    for model in muscles_model:
        muscle_name = model.muscle_name
        model.a_scale = parameter_dict[muscle_name]["a_scale"]
        model.a_rest = parameter_dict[muscle_name]["a_scale"]
        model.fmax = parameter_dict[muscle_name]["Fmax"]
        model.alpha_a = parameter_dict[muscle_name]["alpha_a"]

    # Create MSK FES-driven model
    fes_model = FesMskModel(
        name=None,
        biorbd_path=model_path,
        muscles_model=muscles_model,
        stim_time=stim_time,
        activate_force_length_relationship=True,
        activate_force_velocity_relationship=True,
        activate_passive_force_relationship=True,
        activate_residual_torque=False,
        external_force_set=None,  # External forces will be added later (resistive_torque)
    )
    return fes_model

def create_simulation_list(
    n_cycles_simultaneous: list[int],
    stimulation:           list[int],
    cost_fun_weight:       list[tuple[float, float, float]],
    ode_solver:            OdeSolver(),
) -> list[dict]:

    def make_file_paths(
        n_cycles: int,
        w_force:  float,
        w_fatigue: float,
        w_control: float,
        ode_solver: OdeSolver,
    ) -> tuple[str, str]:

        parts = []
        if w_force:   parts.append(f"{int(w_force*100)}_force")
        if w_fatigue: parts.append(f"{int(w_fatigue*100)}_fatigue")
        if w_control: parts.append(f"{int(w_control*100)}_control")
        weight_suffix = "_".join(parts)

        if isinstance(ode_solver, OdeSolver.COLLOCATION):
            solver_suffix = f"collocation_{ode_solver.polynomial_degree}_{ode_solver.method}"
        elif isinstance(ode_solver, OdeSolver.RK4):
            solver_suffix = f"rk4_{ode_solver.n_integration_steps}"
        else:
            raise RuntimeError("ode_solver must be COLLOCATION or RK4")

        full_suffix = f"{weight_suffix}_{solver_suffix}_with_init"
        pkl = str(Path("result") / f"{n_cycles}_cycle" / f"{n_cycles}_min_{full_suffix}.pkl")
        init = str(Path("result/initial_guess") / f"{n_cycles}_initial_guess_{solver_suffix}.pkl")
        init = init if os.path.exists(init) else None
        if init is None:
            print("No initial guess file for n_cycle: " + str(n_cycles) + " and solver: " + str(solver_suffix))
        return pkl, init

    sims = []
    for n_cycles, stim, (w_f, w_fat, w_c) in product(
        n_cycles_simultaneous, stimulation, cost_fun_weight
    ):
        pkl_path, init_path = make_file_paths(n_cycles, w_f, w_fat, w_c, ode_solver)
        sims.append({
            "n_cycles_simultaneous": n_cycles,
            "stimulation":           stim,
            "minimize_force":        bool(w_f),
            "minimize_fatigue":      bool(w_fat),
            "minimize_control":      bool(w_c),
            "cost_fun_weight":       [w_f, w_fat, w_c],
            "pickle_file_path":      pkl_path,
            "init_guess_file_path":  init_path,
        })
    return sims

def save_sol_in_pkl(sol, simulation_conditions, is_initial_guess=False):
    solution = sol[0] if not is_initial_guess else sol[1][0]
    time = solution.stepwise_time(to_merge=[SolutionMerge.NODES]).T[0]
    states = solution.stepwise_states(to_merge=[SolutionMerge.NODES])
    controls = solution.stepwise_controls(to_merge=[SolutionMerge.NODES])
    stim_time = solution.ocp.nlp[0].model.muscles_dynamics_model[0].stim_time
    solving_time_per_ocp = [sol[1][i].solver_time_to_optimize for i in range(len(sol[1]))]
    objective_values_per_ocp = [float(sol[1][i].cost) for i in range(len(sol[1]))]
    iter_per_ocp = [sol[1][i].iterations for i in range(len(sol[1]))]
    average_solving_time_per_iter_list = [solving_time_per_ocp[i] / iter_per_ocp[i] for i in range(len(sol[1]))]
    total_average_solving_time_per_iter = average(average_solving_time_per_iter_list)
    number_of_turns_before_failing = len(sol[1]) - 1 + simulation_conditions["n_cycles_simultaneous"]
    convergence_status = [sol[1][i].status for i in range(len(sol[1]))]
    dictionary = {
        "time": time,
        "states": states,
        "controls": controls,
        "stim_time": stim_time,
        "solving_time_per_ocp": solving_time_per_ocp,
        "objective_values_per_ocp": objective_values_per_ocp,
        "number_of_turns_before_failing": number_of_turns_before_failing,
        "convergence_status": convergence_status,
        "iter_per_ocp": iter_per_ocp,
        "average_solving_time_per_iter_list": average_solving_time_per_iter_list,
        "total_average_solving_time_per_iter": total_average_solving_time_per_iter,
        "total_n_shooting": solution.ocp.n_shooting,
        "n_shooting_per_cycle": int(solution.ocp.n_shooting / (simulation_conditions["n_cycles_simultaneous"] - 1)),
        "polynomial_order": solution.ocp.nlp[0].dynamics_type.ode_solver.polynomial_degree,
    }
    pickle_file_name = simulation_conditions["pickle_file_path"]
    with open(pickle_file_name, "wb") as file:
        pickle.dump(dictionary, file)
    print(simulation_conditions["pickle_file_path"])

def run_initial_guess(mhe_info, cycling_info, model_path, stimulation, n_cycles_simultaneous, save_sol=True):
    init_guess_mhe_info = {
        "cycle_duration": mhe_info["cycle_duration"],
        "n_cycles_to_advance": mhe_info["n_cycles_to_advance"],
        "n_cycles": 1,
        "ode_solver": mhe_info["ode_solver"],
        "use_sx": mhe_info["use_sx"]
    }

    ode_solver = mhe_info["ode_solver"]
    rk_name = {
        OdeSolver.RK1: "rk1",
        OdeSolver.RK2: "rk2",
        OdeSolver.RK4: "rk4",
    }.get(type(ode_solver))
    if isinstance(ode_solver, OdeSolver.COLLOCATION):
        solver_suffix = f"collocation_{ode_solver.polynomial_degree}_{ode_solver.method}"
    elif rk_name is not None:
        solver_suffix = f"{rk_name}_{ode_solver.n_integration_steps}"
    else:
        raise RuntimeError("ode_solver must either be COLLOCATION or RK")

    for i in range(len(n_cycles_simultaneous)):
        simulation_conditions = {
            "n_cycles_simultaneous": n_cycles_simultaneous[i],
            "stimulation": stimulation[i],
            "minimize_force": False,
            "minimize_fatigue": False,
            "minimize_control": False,
            "cost_fun_weight": [0, 0, 0],
            "pickle_file_path": Path(
                "result") / "initial_guess" / f"{n_cycles_simultaneous[i]}_initial_guess_{solver_suffix}.pkl",
            "init_guess_file_path": None,
        }

        run_optim(mhe_info=init_guess_mhe_info,
                  cycling_info=cycling_info,
                  simulation_conditions=simulation_conditions,
                  model_path=model_path,
                  save_sol=save_sol,
                  run_initial_guess=True)

def run_optim(mhe_info, cycling_info, simulation_conditions, model_path, save_sol, run_initial_guess=False):
    # --- Set FES model --- #
    stim_time = list(
        np.linspace(
            0,
            mhe_info["cycle_duration"] * simulation_conditions["n_cycles_simultaneous"],
            simulation_conditions["stimulation"],
            endpoint=False,
        )
    )
    model = set_fes_model(model_path, stim_time)

    mhe_info["cycle_len"] = int(len(stim_time) / simulation_conditions["n_cycles_simultaneous"])
    mhe_info["n_cycles_simultaneous"] = simulation_conditions["n_cycles_simultaneous"]
    cycling_info["turn_number"] = simulation_conditions["n_cycles_simultaneous"] # One turn per cycle

    nmpc = prepare_nmpc(
        model=model,
        mhe_info=mhe_info,
        cycling_info=cycling_info,
        simulation_conditions=simulation_conditions,
    )
    nmpc.n_cycles_simultaneous = simulation_conditions["n_cycles_simultaneous"]

    def update_functions(_nmpc: MultiCyclicNonlinearModelPredictiveControl, cycle_idx: int, _sol: Solution):
        print("Optimized window n°" + str(cycle_idx))
        return cycle_idx < mhe_info["n_cycles"]  # True if there are still some cycle to perform

    # Add the penalty cost function plot
    nmpc.add_plot_penalty(CostType.ALL)

    # Set solver for the optimal control problem
    solver = Solver.IPOPT(show_online_optim=False, _max_iter=10000, show_options=dict(show_bounds=True))
    solver.set_warm_start_init_point("yes")
    solver.set_mu_init(1e-2)
    solver.set_tol(1e-6)
    solver.set_dual_inf_tol(1e-6)
    solver.set_constr_viol_tol(1e-6)
    # solver.set_linear_solver("ma57")

    # Solve the optimal control problem
    sol = nmpc.solve_fes_nmpc(
        update_functions,
        solver=solver,
        total_cycles=mhe_info["n_cycles"],
        external_force=cycling_info["resistive_torque"],
        cycle_solutions=MultiCyclicCycleSolutions.ALL_CYCLES,
        get_all_iterations=True,
        cyclic_options={"states": {}},
        max_consecutive_failing=1,
    )

    sol[0].animate(viewer="pyorerun")
    plot_mhe_graphs(sol[0])
    for i in range(len(sol[1])):
        sol[1][i].graphs(show_bounds=True)

    # Saving the data in a pickle file
    if save_sol:
        save_sol_in_pkl(sol, simulation_conditions, is_initial_guess=run_initial_guess)


def plot_mhe_graphs(sol):
    def plot_bounds(key, index, subplot_index, data, time):
        current_bounds = sol.ocp.nlp[0].x_bounds[key]
        if current_bounds.min.shape[1] == states[key].shape[1]:
            current_min_bounds = current_bounds.min[:, :][index][:-1]
            current_max_bounds = current_bounds.max[:, :][index][:-1]
        else:
            current_min_bounds = [current_bounds.min[index][0],
                                  *[current_bounds.min[index][1]] * (states[key].shape[1] - 3),
                                  current_bounds.min[index][2]]
            current_max_bounds = [current_bounds.max[index][0],
                                  *[current_bounds.max[index][1]] * (states[key].shape[1] - 3),
                                  current_bounds.max[index][2]]

        axs[subplot_index_list[subplot_index][0], subplot_index_list[subplot_index][1]].plot(time, current_min_bounds)
        axs[subplot_index_list[subplot_index][0], subplot_index_list[subplot_index][1]].plot(time, current_max_bounds)

        labeled = False
        for j in range(data.shape[0]):
            if data[j] < current_min_bounds[j] or data[j] > current_max_bounds[j]:
                axs[subplot_index_list[index][0], subplot_index_list[index][1]].scatter(time[j], data[j], color="red", s=10,
                               label="out of bounds" if not labeled else None)
                labeled = True

    states = sol.stepwise_states(to_merge=SolutionMerge.NODES)
    controls = sol.stepwise_controls(to_merge=SolutionMerge.NODES)
    time = sol.stepwise_time(to_merge=SolutionMerge.NODES).T[0]

    q_key = ["q"]
    qdot_key = ["qdot"]
    cn_key = [key for key in states.keys() if "Cn_" in key]
    f_key = [key for key in states.keys() if "F_" in key]
    a_key = [key for key in states.keys() if "A_" in key]
    tau1_key = [key for key in states.keys() if "Tau1_" in key]
    km_key = [key for key in states.keys() if "Km_" in key]
    control_key = list(controls.keys())

    key_list = [q_key, qdot_key, cn_key, f_key, a_key, tau1_key, km_key]
    subplot_index_list = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y_axis_labels = [
        "rad", "rad/s", "(-)", "N", "N/s", "s", "(-)"]
    j = 0
    for key in key_list:
        fig, axs = plt.subplots(2, 2)
        if key == ["q"] or key == ["qdot"]:
            for i in range(3):
                axs[subplot_index_list[i][0], subplot_index_list[i][1]].plot(time[:-1], states[key[0]][i][:-1])
                axs[subplot_index_list[i][0], subplot_index_list[i][1]].set_title(key[0] + f" index_{i}")
                plot_bounds(key[0], index=i, subplot_index=i, data=states[key[0]][i][:-1], time=time[:-1])
        else:
            for i in range(len(key)):
                axs[subplot_index_list[i][0], subplot_index_list[i][1]].plot(time[:-1], states[key[i]][0][:-1])
                axs[subplot_index_list[i][0], subplot_index_list[i][1]].set_title(key[i])
                plot_bounds(key[i], index=0, subplot_index=i, data=states[key[i]][0][:-1], time=time[:-1])

        for ax in axs.flat:
            ax.set(xlabel='Time (s)', ylabel=y_axis_labels[j])
        j += 1
        plt.show()

    fig, axs = plt.subplots(2, 2)
    for i in range(len(control_key)):
        control_val = [v for v in controls[control_key[i]][0] for _ in range(4)]
        axs[subplot_index_list[i][0], subplot_index_list[i][1]].plot(time[:-1], control_val)
        axs[subplot_index_list[i][0], subplot_index_list[i][1]].set_title(control_key[i])

    for ax in axs.flat:
        ax.set(xlabel='Time (s)', ylabel="pulse width (us)")

    plt.show()

def main():
    # --- Simulation configuration --- #
    save_sol = True
    get_initial_guess = False

    # --- Model choice --- #
    # model_path = "../../model_msk/simplified_UL_Seth_2D_cycling.bioMod"
    model_path = "../../model_msk/Wu_Shoulder_Model_mod_kev_v2.bioMod"

    # --- MHE parameters --- #
    n_cycles_simultaneous = [2]  # , 3, 4, 5]
    ode_solver = OdeSolver.COLLOCATION(polynomial_degree=3, method="radau")
    # ode_solver = OdeSolver.RK4(n_integration_steps=5)
    mhe_info = {
        "cycle_duration": 1,
        "n_cycles_to_advance": 1,
        "n_cycles": 6,
        "ode_solver": ode_solver,
        "use_sx": False
    }

    # --- Bike parameters --- #
    resistive_torque = -0.5
    cycling_info = {"pedal_config": {"x_center": 0.35, "y_center": 0.0, "radius": 0.1},
                    "resistive_torque": {"Segment_application": "wheel", "torque": np.array([0, 0, resistive_torque])}}

    # --- Build cost function weight parameters --- #
    cost_function_weight = [
        (1, 0, 0), (0, 1, 0), (0, 0, 1),
        (0.75, 0.25, 0), (0.5, 0.5, 0), (0.25, 0.75, 0),
        (0.75, 0, 0.25), (0.5, 0, 0.5), (0.25, 0, 0.75),
        (0, 0.75, 0.25), (0, 0.5, 0.5), (0, 0.25, 0.75),
        (1/3, 1/3, 1/3),
    ]  # (min_force, min_fatigue, min_control)

    # --- Build simulation list --- #
    stimulation_frequency = 30
    stimulation = [stimulation_frequency * i for i in n_cycles_simultaneous]

    # --- Build the simulation conditions list --- #
    simulation_conditions_list = create_simulation_list(n_cycles_simultaneous=n_cycles_simultaneous,
                                                        stimulation=stimulation,
                                                        cost_fun_weight=cost_function_weight,
                                                        ode_solver=mhe_info["ode_solver"])

    # --- Run the initial guess optimization --- #
    if get_initial_guess:
        run_initial_guess(
            mhe_info=mhe_info,
            cycling_info=cycling_info,
            model_path=model_path,
            stimulation=stimulation,
            n_cycles_simultaneous=n_cycles_simultaneous,
            save_sol=save_sol
        )

    # --- Run the optimization --- #
    for i in range(len(simulation_conditions_list)):
        run_optim(mhe_info=mhe_info,
                  cycling_info=cycling_info,
                  simulation_conditions=simulation_conditions_list[i],
                  model_path=model_path,
                  save_sol=save_sol)

if __name__ == "__main__":
    main()
