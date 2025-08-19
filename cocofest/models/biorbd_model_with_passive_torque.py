import numpy as np

from casadi import vertcat, MX, SX, exp, Function
from bioptim import (
    BiorbdModel,
    ExternalForceSetTimeSeries,
    OptimalControlProgram,
    NonLinearProgram,
    ConfigureProblem,
    DynamicsFunctions,
    DynamicsEvaluation,
    ParameterList,
)
from .hill_coefficients import (
    muscle_force_length_coefficient,
    muscle_force_velocity_coefficient,
    muscle_passive_force_coefficient,
)

from .state_configure import StateConfigure

class BioRbdModelWithPassiveTorque(BiorbdModel):
    def __init__(
            self,
            muscles_model: list = None,
    ):
        self.muscles_dynamics_model = muscles_model

    def muscle_dynamic(
        self,
        time: MX | SX,
        states: MX | SX,
        controls: MX | SX,
        parameters: MX | SX,
        algebraic_states: MX | SX,
        numerical_data_timeseries: MX | SX,
        nlp: NonLinearProgram,
        muscle_models: list,
        state_name_list=None,
        with_passive_torque=False,
    ) -> DynamicsEvaluation:
        """
        The custom dynamics function that provides the derivative of the states: dxdt = f(t, x, u, p, s)

        Parameters
        ----------
        time: MX | SX
            The time of the system
        states: MX | SX
            The state of the system
        controls: MX | SX
            The controls of the system
        parameters: MX | SX
            The parameters acting on the system
        algebraic_states: MX | SX
            The stochastic variables of the system
        numerical_data_timeseries: MX | SX
            A list of values to pass to the dynamics at each node. Experimental external forces should be included here.
        nlp: NonLinearProgram
            A reference to the phase
        muscle_models: list[FesModel]
            The list of the muscle models
        state_name_list: list[str]
            The states names list
        with_passive_torque: bool
            Whether to include the passive torque in the dynamics
        Returns
        -------
        The derivative of the states in the tuple[MX | SX] format
        """

        q = DynamicsFunctions.get(nlp.states["q"], states)
        qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
        tau = DynamicsFunctions.get(nlp.controls["tau"], controls) if "tau" in nlp.controls.keys() else 0

        muscles_tau, dxdt_muscle_list = self.muscles_joint_torque(
            time,
            states,
            controls,
            parameters,
            algebraic_states,
            numerical_data_timeseries,
            nlp,
            muscle_models,
            state_name_list,
            q,
            qdot,
        )

        dq = DynamicsFunctions.compute_qdot(nlp, q, qdot)
        total_torque = muscles_tau + tau if self.activate_residual_torque else muscles_tau
        if with_passive_torque:
            passive_torque = self.get_passive_torque(theta=q, theta_dot=qdot)
            total_torque = total_torque + passive_torque
            # total_torque = total_torque + nlp.model.passive_joint_torque()(q, qdot, nlp.parameters.cx)
        external_forces = nlp.get_external_forces(states, controls, algebraic_states, numerical_data_timeseries)
        with_contact = (
            True if nlp.model.bio_model.contact_names != () else False
        )  # TODO: Add a better way of with_contact=True
        ddq = nlp.model.forward_dynamics(with_contact=with_contact)(
            q, qdot, total_torque, external_forces, parameters
        )  # q, qdot, tau, external_forces, parameters
        dxdt = vertcat(dxdt_muscle_list, dq, ddq)

        return DynamicsEvaluation(dxdt=dxdt, defects=None)

    @staticmethod
    def get_passive_torque(theta, theta_dot):
        k1 = 0.24395358 #k=1
        k2 = 0.00103129
        k3 = 0.00194639
        k4 = 24.41585281
        kc1 = 1.05713656
        kc2 = 0.19654403
        theta_max = 2.27074652
        theta_min = 0.49997778
        #c = - kc1 * np.exp(-kc2 * (theta - theta_min)) + kc3 * np.exp(kc4 * (theta - theta_max))
        def sigmoide(x):
            return 1 / (1 + exp(-x))
        #c=0.1
        c = (sigmoide((theta - theta_max) / kc2) + sigmoide(-(theta - theta_min) / kc1))
        passive_torque = k1 * exp(-k2 * (theta - theta_min)) * sigmoide(-(theta - theta_min)) - k3 * exp(k4 * (theta - theta_max)) * sigmoide(theta - theta_max) #- c * theta_dot
        #(k1 * exp(-k2 * (theta - theta_min)) - k3 * exp(k4 * (theta - theta_max)) - (c * theta_dot))
        #k1 * exp(-k2 * (theta - theta_min)) * (1 - s) - k3 * exp(k4 * (theta - theta_max)) * s - (c * theta_dot))
        return passive_torque

    def declare_model_variables(
        self,
        ocp: OptimalControlProgram,
        nlp: NonLinearProgram,
        with_contact: bool = False,
        with_passive_torque: bool = False,
        numerical_data_timeseries: dict[str, np.ndarray] = None,
    ):
        """
        Tell the program which variables are states and controls.
        The user is expected to use the ConfigureProblem.configure_xxx functions.
        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        numerical_data_timeseries: dict[str, np.ndarray]
            A list of values to pass to the dynamics at each node. Experimental external forces should be included here.

        """

        state_name_list = [] #StateConfigure().configure_all_muscle_states(self.muscles_dynamics_model, ocp, nlp)
        ConfigureProblem.configure_q(ocp, nlp, as_states=True, as_controls=False)
        state_name_list.append("q")
        ConfigureProblem.configure_qdot(ocp, nlp, as_states=True, as_controls=False)
        state_name_list.append("qdot")
        # for muscle_model in self.muscles_dynamics_model:
        #     if isinstance(muscle_model, DingModelPulseWidthFrequency):
        #         StateConfigure().configure_last_pulse_width(ocp, nlp, muscle_name=str(muscle_model.muscle_name))
        #     if isinstance(muscle_model, DingModelPulseIntensityFrequency):
        #         StateConfigure().configure_pulse_intensity(
        #             ocp, nlp, muscle_name=str(muscle_model.muscle_name), truncation=muscle_model.sum_stim_truncation
        #         )
        # if self.activate_residual_torque:
        #     ConfigureProblem.configure_tau(ocp, nlp, as_states=False, as_controls=True)

        ConfigureProblem.configure_dynamics_function(
            ocp,
            nlp,
            dyn_func=self.muscle_dynamic,
            muscle_models=self.muscles_dynamics_model,
            state_name_list=state_name_list,
            with_passive_torque=with_passive_torque,
        )

        # if with_contact:
        #     ConfigureProblem.configure_contact_function(
        #         ocp, nlp, self.forces_from_fes_driven, state_name_list=state_name_list
        #     )

    @staticmethod
    def muscles_joint_torque(
        time: MX | SX,
        states: MX | SX,
        controls: MX | SX,
        parameters: MX | SX,
        algebraic_states: MX | SX,
        numerical_data_timeseries: MX | SX,
        nlp: NonLinearProgram,
        muscle_models: list,
        state_name_list=None,
        q: MX | SX = None,
        qdot: MX | SX = None,
    ):

        dxdt_muscle_list = vertcat()
        muscle_forces = vertcat()
        muscle_idx_list = []

        Q = nlp.model.q
        Qdot = nlp.model.qdot

        updatedModel = nlp.model.model.UpdateKinematicsCustom(Q, Qdot)
        nlp.model.model.updateMuscles(updatedModel, Q, Qdot)
        updated_muscle_length_jacobian = nlp.model.model.musclesLengthJacobian(updatedModel, Q, False).to_mx()
        updated_muscle_length_jacobian = Function("musclesLengthJacobian", [Q, Qdot], [updated_muscle_length_jacobian])(
            q, qdot
        )

        bio_muscle_names_at_index = []
        for i in range(len(nlp.model.model.muscles())):
            bio_muscle_names_at_index.append(nlp.model.model.muscle(i).name().to_string())

        for muscle_model in muscle_models:
            muscle_states_idxs = [
                i for i in range(len(state_name_list)) if muscle_model.muscle_name in state_name_list[i]
            ]

            muscle_states = vertcat(*[states[i] for i in muscle_states_idxs])

            muscle_parameters_idxs = [
                i for i in range(parameters.shape[0]) if muscle_model.muscle_name in str(parameters[i])
            ]
            muscle_parameters = vertcat(*[parameters[i] for i in muscle_parameters_idxs])

            muscle_controls = controls
            # muscle_control_idxs = [i for i in range(controls.shape[0]) if muscle_model.muscle_name in str(controls[i])]
            # if isinstance(muscle_model, DingModelPulseWidthFrequency):
            #     muscle_controls = controls[muscle_control_idxs[0]]
            # if isinstance(muscle_model, DingModelPulseIntensityFrequency):
            #     muscle_controls = controls[muscle_control_idxs]

            muscle_idx = bio_muscle_names_at_index.index(muscle_model.muscle_name)

            muscle_force_length_coeff = (
                muscle_force_length_coefficient(
                    model=updatedModel,
                    muscle=nlp.model.model.muscle(muscle_idx),
                    q=Q,
                )
                if nlp.model.activate_force_velocity_relationship
                else 1
            )
            muscle_force_length_coeff = Function("muscle_force_length_coeff", [Q, Qdot], [muscle_force_length_coeff])(
                q, qdot
            )

            muscle_force_velocity_coeff = (
                muscle_force_velocity_coefficient(
                    model=updatedModel,
                    muscle=nlp.model.model.muscle(muscle_idx),
                    q=Q,
                    qdot=Qdot,
                )
                if nlp.model.activate_force_velocity_relationship
                else 1
            )
            muscle_force_velocity_coeff = Function(
                "muscle_force_velocity_coeff", [Q, Qdot], [muscle_force_velocity_coeff]
            )(q, qdot)

            muscle_passive_force_coeff = (
                muscle_passive_force_coefficient(
                    model=updatedModel,
                    muscle=nlp.model.model.muscle(muscle_idx),
                    q=Q,
                )
                if nlp.model.activate_passive_force_relationship
                else 0
            )
            muscle_passive_force_coeff = Function(
                "muscle_passive_force_coeff", [Q, Qdot], [muscle_passive_force_coeff]
            )(q, qdot)

            external_force_in_numerical_data_timeseries = (
                True if "external_force" in str(numerical_data_timeseries) else False
            )
            fes_numerical_data_timeseries = (
                numerical_data_timeseries[3 : numerical_data_timeseries.shape[0]]
                if external_force_in_numerical_data_timeseries
                else numerical_data_timeseries
            )
            muscle_dxdt = muscle_model.dynamics(
                time,
                muscle_states,
                muscle_controls,
                muscle_parameters,
                algebraic_states,
                fes_numerical_data_timeseries,
                nlp,
                fes_model=muscle_model,
                force_length_relationship=muscle_force_length_coeff,
                force_velocity_relationship=muscle_force_velocity_coeff,
                passive_force_relationship=muscle_passive_force_coeff,
            ).dxdt

            dxdt_muscle_list = vertcat(dxdt_muscle_list, muscle_dxdt)
            muscle_idx_list.append(muscle_idx)

            muscle_forces = vertcat(
                muscle_forces,
                DynamicsFunctions.get(nlp.states["F_" + muscle_model.muscle_name], states),
            )

        muscle_moment_arm_matrix = updated_muscle_length_jacobian[
            muscle_idx_list, :
        ]  # reorganize the muscle moment arm matrix according to the muscle index list
        muscle_joint_torques = -muscle_moment_arm_matrix.T @ muscle_forces

        return muscle_joint_torques, dxdt_muscle_list