from bioptim import FatigueList, ConfigureProblem, DynamicsFunctions, DynamicsEvaluation
from bioptim.dynamics.configure_problem import _check_contacts_in_biorbd_model

import numpy as np
from casadi import exp, if_else, horzcat, sign


class MuscleDrivenPassiveTorque:
    def __init__(self):
        # Default values for the parameters used in the passive torque calculation
        k1_default = 5
        k2_default = 1
        k3_default = 5
        k4_default = 2
        kc1_default = 1
        kc2_default = 5
        e_min_default = 1
        e_max_default = 5

        theta_c_default = 2
        theta_max_default = 2.344
        theta_min_default = 0.101

        self.k1 = k1_default
        self.k2 = k2_default
        self.k3 = k3_default
        self.k4 = k4_default
        self.kc1 = kc1_default
        self.kc2 = kc2_default
        self.theta_c = theta_c_default
        self.e_min = e_min_default
        self.e_max = e_max_default
        self.theta_max = theta_max_default
        self.theta_min = theta_min_default


    @staticmethod
    def declare_dynamics(
            ocp,
            nlp,
            with_excitations: bool = False,
            fatigue: FatigueList = None,
            with_residual_torque: bool = False,
            with_contact: bool = False,
            with_passive_torque: bool = False,
            with_ligament: bool = False,
            with_friction: bool = False,
            numerical_data_timeseries: dict[str, np.ndarray] = None,
    ):
        """
        Configure the dynamics for a muscle driven program.
        If with_excitations is set to True, then the muscle activations are computed from the muscle dynamics.
        The tau from muscle is computed using the muscle activations.
        If with_residual_torque is set to True, then tau are used as supplementary force in the
        case muscles are too weak.

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the phase
        with_excitations: bool
            If the dynamic should include the muscle dynamics
        fatigue: FatigueList
            The list of fatigue parameters
        with_residual_torque: bool
            If the dynamic should be added with residual torques
        with_contact: bool
            If the dynamic with contact should be used
        with_passive_torque: bool
            If the dynamic with passive torque should be used
        with_ligament: bool
            If the dynamic with ligament should be used
        with_friction: bool
            If the dynamic with joint friction should be used (friction = coefficients * qdot)
        numerical_data_timeseries: dict[str, np.ndarray]
            A list of values to pass to the dynamics at each node. Experimental external forces should be included here.
        """
        _check_contacts_in_biorbd_model(with_contact, nlp.model.nb_contacts, nlp.phase_idx)

        if fatigue is not None and "tau" in fatigue and not with_residual_torque:
            raise RuntimeError("Residual torques need to be used to apply fatigue on torques")

        ConfigureProblem.configure_q(ocp, nlp, True, False)
        ConfigureProblem.configure_qdot(ocp, nlp, True, False, True)
        ConfigureProblem.configure_qddot(ocp, nlp, False, False, True)

        if with_residual_torque:
            ConfigureProblem.configure_tau(ocp, nlp, False, True, fatigue=fatigue)
        ConfigureProblem.configure_muscles(ocp, nlp, with_excitations, True, fatigue=fatigue)

        # if nlp.dynamics_type.dynamic_function:
        #     ConfigureProblem.configure_dynamics_function(ocp, nlp, DynamicsFunctions.custom)
        # else:
        ConfigureProblem.configure_dynamics_function(
            ocp,
            nlp,
            dyn_func=MuscleDrivenPassiveTorque.muscles_driven,
            with_contact=with_contact,
            fatigue=fatigue,
            with_residual_torque=with_residual_torque,
            with_passive_torque=with_passive_torque,
            with_ligament=with_ligament,
            with_friction=with_friction,
        )

        if with_contact:
            ConfigureProblem.configure_contact_function(
                ocp,
                nlp,
                DynamicsFunctions.forces_from_muscle_driven,
            )
        ConfigureProblem.configure_soft_contact_function(ocp, nlp)

    @staticmethod
    def muscles_driven(
        time,
        states,
        controls,
        parameters,
        algebraic_states,
        numerical_timeseries,
        nlp,
        with_contact: bool,
        with_passive_torque: bool = False,
        with_ligament: bool = False,
        with_friction: bool = False,
        with_residual_torque: bool = False,
        fatigue=None,
    ) -> DynamicsEvaluation:
        """
        Forward dynamics driven by muscle.

        Parameters
        ----------
        time: MX.sym | SX.sym
            The time of the system
        states: MX.sym | SX.sym
            The state of the system
        controls: MX.sym | SX.sym
            The controls of the system
        parameters: MX.sym | SX.sym
            The parameters of the system
        algebraic_states: MX.sym | SX.sym
            The algebraic states of the system
        numerical_timeseries: MX.sym | SX.sym
            The numerical timeseries of the system
        nlp: NonLinearProgram
            The definition of the system
        with_contact: bool
            If the dynamic with contact should be used
        with_passive_torque: bool
            If the dynamic with passive torque should be used
        with_ligament: bool
            If the dynamic with ligament should be used
        with_friction: bool
            If the dynamic with friction should be used
        fatigue: FatigueDynamicsList
            To define fatigue elements
        with_residual_torque: bool
            If the dynamic should be added with residual torques

        Returns
        ----------
        DynamicsEvaluation
            The derivative of the states and the defects of the implicit dynamics
        """

        q = nlp.get_var_from_states_or_controls("q", states, controls)
        qdot = nlp.get_var_from_states_or_controls("qdot", states, controls)
        residual_tau = (
            DynamicsFunctions.__get_fatigable_tau(nlp, states, controls, fatigue) if with_residual_torque else None
        )
        mus_activations = nlp.get_var_from_states_or_controls("muscles", states, controls)
        fatigue_states = None
        if fatigue is not None and "muscles" in fatigue:
            mus_fatigue = fatigue["muscles"]
            fatigue_name = mus_fatigue.suffix[0]

            # Sanity check
            n_state_only = sum([m.models.state_only for m in mus_fatigue])
            if 0 < n_state_only < len(fatigue["muscles"]):
                raise NotImplementedError(
                    f"{fatigue_name} list without homogeneous state_only flag is not supported yet"
                )
            apply_to_joint_dynamics = sum([m.models.apply_to_joint_dynamics for m in mus_fatigue])
            if 0 < apply_to_joint_dynamics < len(fatigue["muscles"]):
                raise NotImplementedError(
                    f"{fatigue_name} list without homogeneous apply_to_joint_dynamics flag is not supported yet"
                )

            dyn_suffix = mus_fatigue[0].models.models[fatigue_name].dynamics_suffix()
            fatigue_suffix = mus_fatigue[0].models.models[fatigue_name].fatigue_suffix()
            for m in mus_fatigue:
                for key in m.models.models:
                    if (
                        m.models.models[key].dynamics_suffix() != dyn_suffix
                        or m.models.models[key].fatigue_suffix() != fatigue_suffix
                    ):
                        raise ValueError(f"{fatigue_name} must be of all same types")

            if n_state_only == 0:
                mus_activations = DynamicsFunctions.get(nlp.states[f"muscles_{dyn_suffix}"], states)

            if apply_to_joint_dynamics > 0:
                fatigue_states = DynamicsFunctions.get(nlp.states[f"muscles_{fatigue_suffix}"], states)
        muscles_tau = DynamicsFunctions.compute_tau_from_muscle(nlp, q, qdot, mus_activations, fatigue_states)

        tau = muscles_tau + residual_tau if residual_tau is not None else muscles_tau
        tau = tau + MuscleDrivenPassiveTorque.get_passive_torque(
            k1=parameters[0],
             k2=parameters[1],
             k3=parameters[2],
             k4=parameters[3],
             # kc1=parameters[4],
             # kc2=parameters[5],
            # kc3=parameters[6],
            # kc4=parameters[7],
            theta_max=parameters[4],
            theta_min=parameters[5],
            theta=q,
             theta_dot=qdot) if with_passive_torque else tau

        tau = tau + nlp.model.ligament_joint_torque()(q, qdot, nlp.parameters.cx) if with_ligament else tau
        tau = tau - nlp.model.friction_coefficients @ qdot if with_friction else tau

        dq = DynamicsFunctions.compute_qdot(nlp, q, qdot)

        external_forces = nlp.get_external_forces(states, controls, algebraic_states, numerical_timeseries)
        ddq = DynamicsFunctions.forward_dynamics(nlp, q, qdot, tau, with_contact, external_forces)
        dxdt = nlp.cx(nlp.states.shape, ddq.shape[1])
        dxdt[nlp.states["q"].index, :] = horzcat(*[dq for _ in range(ddq.shape[1])])
        dxdt[nlp.states["qdot"].index, :] = ddq

        has_excitation = True if "muscles" in nlp.states else False
        if has_excitation:
            mus_excitations = DynamicsFunctions.get(nlp.controls["muscles"], controls)
            dmus = DynamicsFunctions.compute_muscle_dot(nlp, mus_excitations, mus_activations)
            dxdt[nlp.states["muscles"].index, :] = horzcat(*[dmus for _ in range(ddq.shape[1])])

        if fatigue is not None and "muscles" in fatigue:
            dxdt = fatigue["muscles"].dynamics(dxdt, nlp, states, controls)

        return DynamicsEvaluation(dxdt=dxdt, defects=None)

    @staticmethod
    def get_passive_torque(k1, k2, k3, k4, theta, theta_min, theta_max, theta_dot):
        md = MuscleDrivenPassiveTorque()
        #c = - kc1 * np.exp(-kc2 * (theta - theta_min)) + kc3 * np.exp(kc4 * (theta - theta_max))
        c = 0.1
        # theta_dot = if_else(theta_dot > 0, theta_dot, 0)  # Ensure that the velocity is positive
        s = 1 / (1 + exp(-0.5*theta_dot))
        passive_torque = k1 * exp(-k2 * (theta - theta_min)) * (1 - s) - k3 * exp(k4 * (theta - theta_max)) * s #- (c * theta_dot)
        #k1 * exp(-k2 * (theta - theta_min)) * (1 - sign(theta_dot)) / 2 - k3 * exp(k4 * (theta - theta_max)) * (1 + sign(theta_dot)) / 2)
        #if_else(theta_dot < 0, k1 * exp(-k2 * (theta - theta_min)), -k3 * exp(k4 * (theta - theta_max))) - (c * theta_dot))
        #k1 * exp(-k2 * (theta - theta_min)) - k3 * exp(k4 * (theta - theta_max)) - (c * theta_dot)
        return passive_torque

    def set_k1(self, k1):
        self.k1 = k1

    def set_k2(self, k2):
        self.k2 = k2

    def set_k3(self, k3):
        self.k3 = k3

    def set_k4(self, k4):
        self.k4 = k4

    def set_kc1(self, kc1):
        self.kc1 = kc1

    def set_kc2(self, kc2):
        self.kc2 = kc2

    def set_theta_c(self, theta_c):
        self.theta_c = theta_c

    def set_e_min(self, e_min):
        self.e_min = e_min

    def set_e_max(self, e_max):
        self.e_max = e_max

    def set_kc3(self, kc3):
        self.kc3 = kc3

    def set_kc4(self, kc4):
        self.kc4 = kc4

    def set_theta_max(self, theta_max):
        self.theta_max = theta_max

    def set_theta_min(self, theta_min):
        self.theta_min = theta_min