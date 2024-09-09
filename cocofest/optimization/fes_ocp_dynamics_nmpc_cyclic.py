import math

import numpy as np
from bioptim import (
    SolutionMerge,
    ObjectiveList,
    ObjectiveFcn,
    OdeSolver,
    Node,
    OptimalControlProgram,
    ControlType,
    TimeAlignment,
)

from .fes_ocp import OcpFes
from .fes_ocp_dynamics import OcpFesMsk
from ..models.fes_model import FesModel
from ..custom_objectives import CustomObjective
from .fes_ocp_nmpc_cyclic import OcpFesNmpcCyclic
from ..dynamics.warm_start import get_initial_guess


class OcpFesDynamicsNmpcCyclic(OcpFesNmpcCyclic):
    def __init__(
        self,
        models: list[FesModel] = None,
        n_stim: int = None,
        n_shooting: int = None,
        final_time: int | float = None,
        pulse_event: dict = None,
        pulse_duration: dict = None,
        pulse_intensity: dict = None,
        bound_type: list = None,
        bound_data: list = None,
        fes_muscle_models: list[FesModel] = None,
        with_residual_torque: bool = None,
        activate_force_length_relationship: bool = None,
        activate_force_velocity_relationship: bool = None,
        minimize_muscle_fatigue: bool = None,
        minimize_muscle_force: bool = None,
        custom_constraint: list = None,
        n_total_cycles: int = None,
        n_simultaneous_cycles: int = None,
        n_cycle_to_advance: int = None,
        cycle_to_keep: str = None,
        objective: dict = None,
        use_sx: bool = True,
        ode_solver: OdeSolver = OdeSolver.RK4(n_integration_steps=1),
        n_threads: int = 1,
    ):
        super(OcpFesNmpcCyclic, self).__init__()
        self.models = models
        self.n_stim = n_stim
        self.n_shooting = n_shooting
        self.final_time = final_time
        self.pulse_event = pulse_event
        self.pulse_duration = pulse_duration
        self.pulse_intensity = pulse_intensity
        self.objective = objective
        self.n_total_cycles = n_total_cycles
        self.n_simultaneous_cycles = n_simultaneous_cycles
        self.n_cycle_to_advance = n_cycle_to_advance
        self.cycle_to_keep = cycle_to_keep
        self.use_sx = use_sx
        self.ode_solver = ode_solver
        self.n_threads = n_threads
        self.bound_type = bound_type
        self.bound_data = bound_data
        self.fes_muscle_models = fes_muscle_models
        self.with_residual_torque = with_residual_torque
        self.activate_force_length_relationship = activate_force_length_relationship
        self.activate_force_velocity_relationship = activate_force_velocity_relationship
        self.minimize_muscle_fatigue = minimize_muscle_fatigue
        self.minimize_muscle_force = minimize_muscle_force
        self.custom_constraint = custom_constraint
        self.warm_start = False
        self.ocp = None
        self._nmpc_sanity_check()
        self.states = []
        self.parameters = []
        self.previous_stim = []
        self.result = {"time": {}, "states": {}, "parameters": {}}
        self.temp_last_node_time = 0
        self.first_node_in_phase = 0
        self.last_node_in_phase = 0

    def prepare_nmpc(self):
        (pulse_event, pulse_duration, pulse_intensity, objective) = OcpFes._fill_dict(
            self.pulse_event, self.pulse_duration, self.pulse_intensity, self.objective
        )

        time_min = pulse_event["min"]
        time_max = pulse_event["max"]
        time_bimapping = pulse_event["bimapping"]
        frequency = pulse_event["frequency"]
        round_down = pulse_event["round_down"]
        pulse_mode = pulse_event["pulse_mode"]

        fixed_pulse_duration = pulse_duration["fixed"]
        pulse_duration_min = pulse_duration["min"]
        pulse_duration_max = pulse_duration["max"]
        pulse_duration_bimapping = pulse_duration["bimapping"]
        key_in_dict = "similar_for_all_muscles" in pulse_duration
        pulse_duration_similar_for_all_muscles = pulse_duration["similar_for_all_muscles"] if key_in_dict else False

        fixed_pulse_intensity = pulse_intensity["fixed"]
        pulse_intensity_min = pulse_intensity["min"]
        pulse_intensity_max = pulse_intensity["max"]
        pulse_intensity_bimapping = pulse_intensity["bimapping"]
        key_in_dict = "similar_for_all_muscles" in pulse_intensity
        pulse_intensity_similar_for_all_muscles = pulse_intensity["similar_for_all_muscles"] if key_in_dict else False

        force_tracking = objective["force_tracking"]
        end_node_tracking = objective["end_node_tracking"]
        cycling_objective = objective["cycling"]
        custom_objective = objective["custom"]
        key_in_dict = "q_tracking" in objective
        q_tracking = objective["q_tracking"] if key_in_dict else None

        OcpFes._sanity_check(
            models=[self.models[i].bio_stim_model[1] for i in range(len(self.models))],
            n_stim=self.n_stim * self.n_simultaneous_cycles,
            n_shooting=self.n_shooting,
            final_time=self.final_time,
            pulse_mode=pulse_mode,
            frequency=frequency,
            time_min=time_min,
            time_max=time_max,
            time_bimapping=time_bimapping,
            fixed_pulse_duration=fixed_pulse_duration,
            pulse_duration_min=pulse_duration_min,
            pulse_duration_max=pulse_duration_max,
            pulse_duration_bimapping=pulse_duration_bimapping,
            fixed_pulse_intensity=fixed_pulse_intensity,
            pulse_intensity_min=pulse_intensity_min,
            pulse_intensity_max=pulse_intensity_max,
            pulse_intensity_bimapping=pulse_intensity_bimapping,
            custom_objective=custom_objective,
            use_sx=self.use_sx,
            ode_solver=self.ode_solver,
            n_threads=self.n_threads,
        )

        OcpFesMsk._sanity_check_fes_models_inputs(
            biorbd_model_path=self.models[0].bio_model.path,
            bound_type=self.bound_type,
            bound_data=self.bound_data,
            fes_muscle_models=self.models[0].muscles_dynamics_model,
            force_tracking=force_tracking,
            end_node_tracking=end_node_tracking,
            cycling_objective=cycling_objective,
            q_tracking=q_tracking,
            with_residual_torque=self.with_residual_torque,
            activate_force_length_relationship=self.activate_force_length_relationship,
            activate_force_velocity_relationship=self.activate_force_velocity_relationship,
            minimize_muscle_fatigue=self.minimize_muscle_fatigue,
            minimize_muscle_force=self.minimize_muscle_force,
        )

        OcpFes._sanity_check_frequency(n_stim=self.n_stim, final_time=self.final_time, frequency=frequency, round_down=round_down)

        OcpFesMsk._sanity_check_muscle_model(biorbd_model_path=self.models[0].bio_model.path, fes_muscle_models=self.models[0].muscles_dynamics_model)

        n_stim, final_time = OcpFes._build_phase_parameter(
            n_stim=self.n_stim, final_time=self.final_time, frequency=frequency, pulse_mode=pulse_mode, round_down=round_down
        )

        force_fourier_coef = [] if force_tracking else None
        if force_tracking:
            for i in range(len(force_tracking[1])):
                force_fourier_coef.append(OcpFes._build_fourier_coefficient([force_tracking[0], force_tracking[1][i]]))

        q_fourier_coef = [] if q_tracking else None
        if q_tracking:
            for i in range(len(q_tracking[1])):
                q_fourier_coef.append(OcpFes._build_fourier_coefficient([q_tracking[0], q_tracking[1][i]]))

        n_shooting = [self.n_shooting] * n_stim * self.n_simultaneous_cycles
        final_time_phase = OcpFes._build_phase_time(
            final_time=final_time * self.n_simultaneous_cycles,
            n_stim=n_stim * self.n_simultaneous_cycles,
            pulse_mode=pulse_mode,
            time_min=time_min,
            time_max=time_max,
        )
        (
            parameters,
            parameters_bounds,
            parameters_init,
            parameter_objectives,
            constraints,
        ) = OcpFesMsk._build_parameters(
            model=self.models[0].muscles_dynamics_model,
            n_stim=n_stim * self.n_simultaneous_cycles,
            time_min=time_min,
            time_max=time_max,
            time_bimapping=time_bimapping,
            fixed_pulse_duration=fixed_pulse_duration,
            pulse_duration_min=pulse_duration_min,
            pulse_duration_max=pulse_duration_max,
            pulse_duration_bimapping=pulse_duration_bimapping,
            pulse_duration_similar_for_all_muscles=pulse_duration_similar_for_all_muscles,
            fixed_pulse_intensity=fixed_pulse_intensity,
            pulse_intensity_min=pulse_intensity_min,
            pulse_intensity_max=pulse_intensity_max,
            pulse_intensity_bimapping=pulse_intensity_bimapping,
            pulse_intensity_similar_for_all_muscles=pulse_intensity_similar_for_all_muscles,
            use_sx=self.use_sx,
        )

        constraints = OcpFesMsk._set_constraints(constraints, self.custom_constraint)

        if len(constraints) == 0 and len(parameters) == 0:
            raise ValueError(
                "This is not an optimal control problem,"
                " add parameter to optimize or use the IvpFes method to build your problem"
            )

        dynamics = OcpFesMsk._declare_dynamics(self.models, n_stim * self.n_simultaneous_cycles)
        initial_state = (
            get_initial_guess(self.models[0].bio_model.path, final_time, n_stim, n_shooting, objective) if self.warm_start else None
        )

        x_bounds, x_init = OcpFesMsk._set_bounds(
            self.models,
            self.models[0].muscles_dynamics_model,
            self.bound_type,
            self.bound_data,
            n_stim * self.n_simultaneous_cycles,
            initial_state,
        )
        u_bounds, u_init = OcpFesMsk._set_controls(self.models, n_stim * self.n_simultaneous_cycles, self.with_residual_torque)
        muscle_force_key = ["F_" + self.models[0].muscles_dynamics_model[i].muscle_name for i in range(len(self.models[0].muscles_dynamics_model))]
        objective_functions = OcpFesMsk._set_objective(
            n_stim,
            n_shooting,
            force_fourier_coef,
            end_node_tracking,
            cycling_objective,
            custom_objective,
            q_fourier_coef,
            self.minimize_muscle_fatigue,
            self.minimize_muscle_force,
            muscle_force_key,
            time_min,
            time_max,
            n_simultaneous_cycles=self.n_simultaneous_cycles
        )

        self.ocp = OptimalControlProgram(
            bio_model=self.models,
            dynamics=dynamics,
            n_shooting=n_shooting,
            phase_time=final_time_phase,
            objective_functions=objective_functions,
            x_init=x_init,
            x_bounds=x_bounds,
            u_init=u_init,
            u_bounds=u_bounds,
            constraints=constraints,
            parameters=parameters,
            parameter_bounds=parameters_bounds,
            parameter_init=parameters_init,
            parameter_objectives=parameter_objectives,
            # control_type=self.control_type,
            use_sx=self.use_sx,
            ode_solver=self.ode_solver,
            n_threads=self.n_threads,
        )
        return self.ocp

    def update_states_bounds(self, sol_states):
        state_keys = list(self.ocp.nlp[0].states.keys())
        index_to_keep = 1 * self.n_stim - 1  # todo: update this when more simultaneous cycles than 3
        for key in state_keys:
            self.ocp.nlp[0].x_bounds[key].max[0][0] = sol_states[index_to_keep][key][0][-1]
            self.ocp.nlp[0].x_bounds[key].min[0][0] = sol_states[index_to_keep][key][0][-1]
            for j in range(index_to_keep, len(self.ocp.nlp)):
                self.ocp.nlp[j].x_init[key].init[0][0] = sol_states[j][key][0][0]

    def update_stim(self, sol):
        if "pulse_apparition_time" in sol.decision_parameters():
            stimulation_time = sol.decision_parameters()["pulse_apparition_time"]
        else:
            stimulation_time = [0] + list(np.cumsum(sol.ocp.phase_time[: self.n_stim - 1]))

        stim_prev = list(np.array(stimulation_time) - self.final_time)
        self.previous_stim = list(np.array(self.previous_stim) - self.final_time) if self.previous_stim else stim_prev

        for j in range(len(self.ocp.nlp)):
            self.ocp.nlp[j].model.time_stim_prev = self.previous_stim + self.ocp.nlp[j].model.time_stim_prev
            # TODO: Does not seem to be taken into account by the next model force estimation

    def store_results(self, sol_time, sol_states, sol_parameters, index):
        if self.cycle_to_keep == "middle":
            # Get the middle phase index to keep
            phase_to_keep = int(math.ceil(self.n_simultaneous_cycles / 2))
            self.first_node_in_phase = self.n_stim * (phase_to_keep - 1)
            self.last_node_in_phase = self.n_stim * phase_to_keep

            # Initialize the dict if it's the first iteration
            if index == 0:
                self.result["time"] = [None] * self.n_total_cycles
                [
                    self.result["states"].update({state_key: [None] * self.n_total_cycles})
                    for state_key in list(sol_states[0].keys())
                ]
                [
                    self.result["parameters"].update({key_parameter: [None] * self.n_total_cycles})
                    for key_parameter in list(sol_parameters.keys())
                ]

            # Store the results
            phase_size = np.array(sol_time).shape[0]
            node_size = np.array(sol_time).shape[1]
            sol_time = list(np.array(sol_time).reshape(phase_size * node_size))[
                self.first_node_in_phase * node_size : self.last_node_in_phase * node_size
            ]
            sol_time = list(dict.fromkeys(sol_time))  # Remove duplicate time
            if index == 0:
                updated_sol_time = [t - sol_time[0] for t in sol_time]
            else:
                updated_sol_time = [t - sol_time[0] + self.temp_last_node_time for t in sol_time]
            self.temp_last_node_time = updated_sol_time[-1]
            self.result["time"][index] = updated_sol_time[:-1]

            for state_key in list(sol_states[0].keys()):
                middle_states_values = sol_states[self.first_node_in_phase : self.last_node_in_phase]
                middle_states_values = [
                    list(middle_states_values[i][state_key][0])[:-1] for i in range(len(middle_states_values))
                ]  # Remove the last node duplicate
                middle_states_values = [j for sub in middle_states_values for j in sub]
                self.result["states"][state_key][index] = middle_states_values

            for key_parameter in list(sol_parameters.keys()):
                self.result["parameters"][key_parameter][index] = sol_parameters[key_parameter][
                    self.first_node_in_phase : self.last_node_in_phase
                ]
        return

    def solve(self, solver):
        for i in range(self.n_total_cycles // self.n_cycle_to_advance):
            sol = self.ocp.solve(solver)
            sol.graphs()
            sol_states = sol.decision_states(to_merge=[SolutionMerge.NODES])
            self.update_states_bounds(sol_states)
            sol_time = sol.decision_time(to_merge=SolutionMerge.NODES, time_alignment=TimeAlignment.STATES)
            sol_parameters = sol.decision_parameters()
            self.store_results(sol_time, sol_states, sol_parameters, i)
            self.update_stim(sol)
            # Todo uncomment when the model is updated to take into account the past stimulation

    @staticmethod
    def _set_objective(
        n_stim,
        n_shooting,
        force_fourier_coefficient,
        end_node_tracking,
        custom_objective,
        time_min,
        time_max,
        n_simultaneous_cycles,
    ):
        # Creates the objective for our problem
        objective_functions = ObjectiveList()
        if custom_objective:
            if len(custom_objective) != n_stim:
                raise ValueError(
                    "The number of custom objective must be equal to the stimulation number of a single cycle"
                )
            for i in range(len(custom_objective)):
                for j in range(n_simultaneous_cycles):
                    objective_functions.add(custom_objective[i + j * n_stim][0])

        if force_fourier_coefficient is not None:
            for phase in range(n_stim):
                for i in range(n_shooting[phase]):
                    for j in range(n_simultaneous_cycles):
                        objective_functions.add(
                            CustomObjective.track_state_from_time,
                            custom_type=ObjectiveFcn.Mayer,
                            node=i,
                            fourier_coeff=force_fourier_coefficient,
                            key="F",
                            quadratic=True,
                            weight=1,
                            phase=phase + j * n_stim,
                        )

        if end_node_tracking:
            if isinstance(end_node_tracking, int | float):
                for i in range(n_simultaneous_cycles):
                    objective_functions.add(
                        ObjectiveFcn.Mayer.MINIMIZE_STATE,
                        node=Node.END,
                        key="F",
                        quadratic=True,
                        weight=1,
                        target=end_node_tracking,
                        phase=n_stim - 1 + i * n_stim,
                    )

        if time_min and time_max:
            for i in range(n_stim):
                for j in range(n_simultaneous_cycles):
                    objective_functions.add(
                        ObjectiveFcn.Mayer.MINIMIZE_TIME,
                        weight=0.001 / n_shooting[i],
                        min_bound=time_min,
                        max_bound=time_max,
                        quadratic=True,
                        phase=i + j * n_stim,
                    )

        return objective_functions

    def _nmpc_sanity_check(self):
        if not isinstance(self.n_total_cycles, int):
            raise TypeError("n_total_cycles must be an integer")
        if not isinstance(self.n_simultaneous_cycles, int):
            raise TypeError("n_simultaneous_cycles must be an integer")
        if not isinstance(self.n_cycle_to_advance, int):
            raise TypeError("n_cycle_to_advance must be an integer")
        if not isinstance(self.cycle_to_keep, str):
            raise TypeError("cycle_to_keep must be a string")

        if self.n_cycle_to_advance > self.n_simultaneous_cycles:
            raise ValueError("The number of n_simultaneous_cycles must be higher than the number of n_cycle_to_advance")

        if self.n_total_cycles % self.n_cycle_to_advance != 0:
            raise ValueError("The number of n_total_cycles must be a multiple of the number n_cycle_to_advance")

        if self.cycle_to_keep not in ["first", "middle", "last"]:
            raise ValueError("cycle_to_keep must be either 'first', 'middle' or 'last'")
        if self.cycle_to_keep != "middle":
            raise NotImplementedError("Only 'middle' cycle_to_keep is implemented")

        if self.n_simultaneous_cycles != 3:
            raise NotImplementedError("Only 3 simultaneous cycles are implemented yet work in progress")
            # Todo add more simultaneous cycles
