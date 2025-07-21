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

from ..models.fes_model import FesModel
from ..models.ding2003 import DingModelFrequency
from ..models.ding2007 import DingModelPulseWidthFrequency
from ..models.hmed2018 import DingModelPulseIntensityFrequency
from .state_configure import StateConfigure
from .hill_coefficients import (
    muscle_force_length_coefficient,
    muscle_force_velocity_coefficient,
    muscle_passive_force_coefficient,
)


class FesMskModel(BiorbdModel):
    def __init__(
        self,
        name: str = None,
        biorbd_path: str = None,
        muscles_model: list[FesModel] = None,
        stim_time: list[float] = None,
        previous_stim: dict = None,
        activate_force_length_relationship: bool = False,
        activate_force_velocity_relationship: bool = False,
        activate_passive_force_relationship: bool = False,
        activate_residual_torque: bool = False,
        parameters: ParameterList = None,
        external_force_set: ExternalForceSetTimeSeries = None,
    ):
        """
        The custom model that will be used in the optimal control program for the FES-MSK models

        Parameters
        ----------
        name: str
            The model name
        biorbd_path: str
            The path to the biorbd model
        muscles_model: DingModelFrequency
            The muscle model that will be used in the model
        activate_force_length_relationship: bool
            If the force-length relationship should be activated
        activate_force_velocity_relationship: bool
            If the force-velocity relationship should be activated
        activate_passive_force_relationship: bool
            If the passive force relationship should be activated
        activate_residual_torque: bool
            If the residual torque should be activated
        parameters: ParameterList
            The parameters that will be used in the model
        """
        super().__init__(biorbd_path, parameters=parameters, external_force_set=external_force_set)
        self.bio_model = BiorbdModel(biorbd_path, parameters=parameters, external_force_set=external_force_set)
        self._name = name
        self.biorbd_path = biorbd_path

        self._model_sanity(
            muscles_model,
            activate_force_length_relationship,
            activate_force_velocity_relationship,
        )
        self.muscles_dynamics_model = muscles_model
        for i in range(len(self.muscles_dynamics_model)):
            self.muscles_dynamics_model[i].stim_time = (
                stim_time if stim_time else self.muscles_dynamics_model[i].stim_time
            )
            self.muscles_dynamics_model[i].previous_stim = (
                previous_stim if previous_stim else self.muscles_dynamics_model[i].previous_stim
            )
            self.muscles_dynamics_model[i].all_stim = self.muscles_dynamics_model[i].previous_stim["time"] + stim_time

        self.bio_stim_model = [self.bio_model] + self.muscles_dynamics_model

        self.activate_force_length_relationship = activate_force_length_relationship
        self.activate_force_velocity_relationship = activate_force_velocity_relationship
        self.activate_passive_force_relationship = activate_passive_force_relationship
        self.activate_residual_torque = activate_residual_torque
        self.parameters_list = parameters
        self.external_forces_set = external_force_set

    # ---- Absolutely needed methods ---- #
    def serialize(self) -> tuple[Callable, dict]:
        return (
            FesMskModel,
            {
                "name": self._name,
                "biorbd_path": self.bio_model.path,
                "muscles_model": self.muscles_dynamics_model,
                "stim_time": self.muscles_dynamics_model[0].stim_time,
                "previous_stim": self.muscles_dynamics_model[0].previous_stim,
                "activate_force_length_relationship": self.activate_force_length_relationship,
                "activate_force_velocity_relationship": self.activate_force_velocity_relationship,
                "activate_passive_force_relationship": self.activate_passive_force_relationship,
                "activate_residual_torque": self.activate_residual_torque,
                "parameters": self.parameters_list,
                "external_force_set": self.external_forces_set,
            },
        )

    # ---- Needed for the example ---- #
    @property
    def name_dof(self) -> tuple[str]:
        return self.bio_model.name_dof

    def muscle_name_dof(self, index: int = 0) -> list[str]:
        return self.muscles_dynamics_model[index].name_dof(with_muscle_name=True)

    @property
    def nb_state(self) -> int:
        nb_state = 0
        for muscle_model in self.muscles_dynamics_model:
            nb_state += muscle_model.nb_state
        nb_state += self.bio_model.nb_q
        return nb_state

    @property
    def name(self) -> None | str:
        return self._name

    def get_passive_torque(self, theta, theta_dot):

        k1 = 0.31936218
        k2 = 0.00102681
        k3 = 0.01222197
        k4 = 9.99959244
        theta_max = 2.02248865
        theta_min = 0.49999636
        k1 = 0.43919893
        k2 = 0.00101457
        k3 = 0.00100078
        k4 = 14.05330065
        theta_max = 2.00000496
        theta_min = 0.49999865
        k1 = 0.00742484
        k2 = 0.00223747
        k3 = 0.00100167
        k4 = 25.471912
        theta_max = 2.24729024
        theta_min = 0.18412647
        k1 = 0.05191881
        k2 = 0.00114811
        k3 = 0.0010015
        k4 = 21.29188453
        theta_max = 2.18734733
        theta_min = 0.40881427
        k1 = 0.26208257
        k2 = 9.99998504
        k3 = 3.39228857
        k4 = 0.00100025
        theta_max = 2.0000011
        theta_min = 0.49999987
        k1 = 0.24395358
        k2 = 0.00103129
        k3 = 0.000194639
        k4 = 24.41585281
        kc1 = 1.05713656
        kc2 = 0.19654403
        theta_max = 2.27074652
        theta_min = 0.49997778
        k1 = 0.30217815 #k=0.9
        k2 = 0.00102294
        k3 = 0.00564691
        k4 = 24.12280173
        kc1 = 1.84481101
        kc2 = 0.12915622
        theta_max = 2.30904925
        theta_min = 0.49997775
        k1 = 0.23845744 #k=0.85
        k2 = 0.0010318
        k3 = 0.00100285
        k4 = 22.53819414
        kc1 = 1.11427573
        kc2 = 0.16773932
        theta_max = 2.21780163
        theta_min = 0.49997263
        k1 = 0.23556646 #k=0.8 max
        k2 = 0.00103206
        k3 = 0.00100176
        k4 = 21.54613035
        kc1 = 1.14183095
        kc2 = 0.15589504
        theta_max = 2.20185039
        theta_min = 0.49996896
        k1 = 0.3623444 #k=0.85
        k2 = 0.00101858
        k3 = 0.00430086
        k4 = 23.91321625
        kc1 = 1.99987948
        kc2 = 0.1570301
        theta_max = 2.2971145
        theta_min = 0.49998832
        k1 = 0.41858272  # k=8
        k2 = 0.00101626
        k3 = 0.00378704
        k4 = 23.94017168
        kc1 = 1.99989723
        kc2 = 0.23019732
        theta_max = 2.29635469
        theta_min = 0.49999104
        k1 = 0.33655981  # k=0.5
        k2 = 0.00102635
        k3 = 0.00100043
        k4 = 23.62753997
        kc1 = 1.99997016
        kc2 = 1.99978018
        theta_max = 2.29025762
        theta_min = 0.49998263
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
        # passive_torque = 0
        return passive_torque

    def muscle_dynamic(
        self,
        time: MX | SX,
        states: MX | SX,
        controls: MX | SX,
        parameters: MX | SX,
        algebraic_states: MX | SX,
        numerical_data_timeseries: MX | SX,
        nlp: NonLinearProgram,
        muscle_models: list[FesModel],
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

    @staticmethod
    def muscles_joint_torque(
        time: MX | SX,
        states: MX | SX,
        controls: MX | SX,
        parameters: MX | SX,
        algebraic_states: MX | SX,
        numerical_data_timeseries: MX | SX,
        nlp: NonLinearProgram,
        muscle_models: list[FesModel],
        state_name_list=None,
        q: MX | SX = None,
        qdot: MX | SX = None,
    ):

        dxdt_muscle_list = vertcat()
        muscle_forces = vertcat()
        muscle_idx_list = []

        Q = nlp.model.bio_model.q
        Qdot = nlp.model.bio_model.qdot

        updatedModel = nlp.model.bio_model.model.UpdateKinematicsCustom(Q, Qdot)
        nlp.model.bio_model.model.updateMuscles(updatedModel, Q, Qdot)
        updated_muscle_length_jacobian = nlp.model.bio_model.model.musclesLengthJacobian(updatedModel, Q, False).to_mx()
        updated_muscle_length_jacobian = Function("musclesLengthJacobian", [Q, Qdot], [updated_muscle_length_jacobian])(
            q, qdot
        )

        bio_muscle_names_at_index = []
        for i in range(len(nlp.model.bio_model.model.muscles())):
            bio_muscle_names_at_index.append(nlp.model.bio_model.model.muscle(i).name().to_string())

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
            muscle_control_idxs = [i for i in range(controls.shape[0]) if muscle_model.muscle_name in str(controls[i])]
            if isinstance(muscle_model, DingModelPulseWidthFrequency):
                muscle_controls = controls[muscle_control_idxs[0]]
            if isinstance(muscle_model, DingModelPulseIntensityFrequency):
                muscle_controls = controls[muscle_control_idxs]

            muscle_idx = bio_muscle_names_at_index.index(muscle_model.muscle_name)

            muscle_force_length_coeff = (
                muscle_force_length_coefficient(
                    model=updatedModel,
                    muscle=nlp.model.bio_model.model.muscle(muscle_idx),
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
                    muscle=nlp.model.bio_model.model.muscle(muscle_idx),
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
                    muscle=nlp.model.bio_model.model.muscle(muscle_idx),
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

    @staticmethod
    def forces_from_fes_driven(
        time: MX.sym,
        states: MX.sym,
        controls: MX.sym,
        parameters: MX.sym,
        algebraic_states: MX.sym,
        numerical_timeseries: MX.sym,
        nlp,
        with_passive_torque: bool = False,
        with_ligament: bool = False,
        state_name_list=None,
    ) -> MX:
        """
        Contact forces of a forward dynamics driven by muscles activations and joint torques with contact constraints.

        Parameters
        ----------
        time: MX.sym
            The time of the system
        states: MX.sym
            The state of the system
        controls: MX.sym
            The controls of the system
        parameters: MX.sym
            The parameters of the system
        algebraic_states: MX.sym
            The algebraic states of the system
        numerical_timeseries: MX.sym
            The numerical timeseries of the system
        nlp: NonLinearProgram
            The definition of the system
        with_passive_torque: bool
            If the dynamic with passive torque should be used
        with_ligament: bool
            If the dynamic with ligament should be used
        state_name_list: list[str]
            The states names list
        Returns
        ----------
        MX.sym
            The contact forces that ensure no acceleration at these contact points
        """

        q = nlp.get_var_from_states_or_controls("q", states, controls)
        qdot = nlp.get_var_from_states_or_controls("qdot", states, controls)
        residual_tau = nlp.get_var_from_states_or_controls("tau", states, controls) if "tau" in nlp.controls else None
        # mus_activations = nlp.get_var_from_states_or_controls("muscles", states, controls)
        # muscles_tau = DynamicsFunctions.compute_tau_from_muscle(nlp, q, qdot, mus_activations)

        muscles_joint_torque, _ = FesMskModel.muscles_joint_torque(
            time,
            states,
            controls,
            parameters,
            algebraic_states,
            numerical_timeseries,
            nlp,
            nlp.model.muscles_dynamics_model,
            state_name_list,
            q,
            qdot,
        )

        tau = muscles_joint_torque + residual_tau if residual_tau is not None else muscles_joint_torque
        tau = tau + nlp.model.passive_joint_torque()(q, qdot, nlp.parameters.cx) if with_passive_torque else tau
        tau = tau + nlp.model.ligament_joint_torque()(q, qdot, nlp.parameters.cx) if with_ligament else tau

        external_forces = nlp.get_external_forces(states, controls, algebraic_states, numerical_timeseries[0:3])

        return nlp.model.contact_forces()(q, qdot, tau, external_forces, nlp.parameters.cx)

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

    @staticmethod
    def _model_sanity(
        muscles_model,
        activate_force_length_relationship,
        activate_force_velocity_relationship,
    ):
        if not isinstance(muscles_model, list):
            for muscle_model in muscles_model:
                if not isinstance(muscle_model, FesModel):
                    raise TypeError(
                        f"The current model type used is {type(muscles_model)}, it must be a FesModel type."
                        f"Current available models are: DingModelFrequency, DingModelFrequencyWithFatigue,"
                        f"DingModelPulseWidthFrequency, DingModelPulseWidthFrequencyWithFatigue,"
                        f"DingModelPulseIntensityFrequency, DingModelPulseIntensityFrequencyWithFatigue"
                    )

            raise TypeError("The given muscles_model must be a list of FesModel")

        if not isinstance(activate_force_length_relationship, bool):
            raise TypeError("The activate_force_length_relationship must be a boolean")

        if not isinstance(activate_force_velocity_relationship, bool):
            raise TypeError("The activate_force_velocity_relationship must be a boolean")


def _check_numerical_timeseries_format(numerical_timeseries: np.ndarray, n_shooting: int, phase_idx: int):
    """Check if the numerical_data_timeseries is of the right format"""
    if type(numerical_timeseries) is not np.ndarray:
        raise RuntimeError(
            f"Phase {phase_idx} has numerical_data_timeseries of type {type(numerical_timeseries)} "
            f"but it should be of type np.ndarray"
        )
    if numerical_timeseries is not None and numerical_timeseries.shape[2] != n_shooting + 1:
        raise RuntimeError(
            f"Phase {phase_idx} has {n_shooting}+1 shooting points but the numerical_data_timeseries "
            f"has {numerical_timeseries.shape[2]} shooting points."
            f"The numerical_data_timeseries should be of format dict[str, np.ndarray] "
            f"where the list is the number of shooting points of the phase "
        )
