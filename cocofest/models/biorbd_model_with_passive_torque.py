from typing import Callable
import numpy as np

from casadi import vertcat, MX, SX, Function, exp, if_else, fabs
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

from .state_configure import StateConfigure

class BioRbdModelWithPassiveTorque(BiorbdModel):
    def __init__(self):
        pass
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

        # You can directly call biorbd function (as for ddq) or call bioptim accessor (as for dq)
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

    def get_passive_torque(self, theta, theta_dot):
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
        #theta_dot = if_else(theta_dot > 0, theta_dot, 0)  # Ensure that the velocity is positive
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

        state_name_list = StateConfigure().configure_all_muscle_states(self.muscles_dynamics_model, ocp, nlp)
        ConfigureProblem.configure_q(ocp, nlp, as_states=True, as_controls=False)
        state_name_list.append("q")
        ConfigureProblem.configure_qdot(ocp, nlp, as_states=True, as_controls=False)
        state_name_list.append("qdot")
        for muscle_model in self.muscles_dynamics_model:
            if isinstance(muscle_model, DingModelPulseWidthFrequency):
                StateConfigure().configure_last_pulse_width(ocp, nlp, muscle_name=str(muscle_model.muscle_name))
            if isinstance(muscle_model, DingModelPulseIntensityFrequency):
                StateConfigure().configure_pulse_intensity(
                    ocp, nlp, muscle_name=str(muscle_model.muscle_name), truncation=muscle_model.sum_stim_truncation
                )
        if self.activate_residual_torque:
            ConfigureProblem.configure_tau(ocp, nlp, as_states=False, as_controls=True)

        ConfigureProblem.configure_dynamics_function(
            ocp,
            nlp,
            dyn_func=self.muscle_dynamic,
            muscle_models=self.muscles_dynamics_model,
            state_name_list=state_name_list,
            with_passive_torque=with_passive_torque,
        )

        if with_contact:
            ConfigureProblem.configure_contact_function(
                ocp, nlp, self.forces_from_fes_driven, state_name_list=state_name_list
            )
