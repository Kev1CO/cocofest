"""
Exponential passive joint torque of the elbow, and the bioptim models carrying it.

The passive torque follows the classical double-exponential formulation: a flexion term that grows when the joint
approaches its extension limit theta_min, an extension term that grows when it approaches its flexion limit
theta_max, and a viscous term that is only active close to those two limits.

    tau_p(theta, theta_dot) =  k1 * exp(-k2 * (theta - theta_min)) * s(-(theta - theta_min))
                             - k3 * exp( k4 * (theta - theta_max)) * s(  theta - theta_max )
                             - c(theta) * theta_dot
    with  c(theta) = s(-(theta - theta_min) / kc1) + s((theta - theta_max) / kc2)
    and   s(x)     = 1 / (1 + exp(-x))                                              (sigmoid)
"""

from __future__ import annotations

import numpy as np
from casadi import MX, SX, Function, exp, fmax, fmin, vertcat

from bioptim import TorqueBiorbdModel

from cocofest import FesMskModel


THETA_MAX_MARGIN = np.deg2rad(10)
END_STOP_RATE_MAX = 30


def sigmoid(x):
    """Logistic sigmoid, working both on floats/arrays (numpy) and on casadi symbols."""
    return 1 / (1 + exp(-x))


def clip(value, lower, upper):
    """Clip a value between two bounds, working both on floats/arrays (numpy) and on casadi symbols."""
    if isinstance(value, (MX, SX)):
        return fmin(fmax(value, lower), upper)
    return np.clip(value, lower, upper)


class PassiveTorque:
    """
    The exponential passive joint torque model. Its setters follow the cocofest convention
    ``set_xxx(self, model, value)`` so that they can be given directly to a bioptim ParameterList.
    """

    # Default values, overwritten either by an identified set or by the ocp parameters
    DEFAULTS = {
        "k1": 5.0,
        "k2": 1.0,
        "k3": 5.0,
        "k4": 2.0,
        "kc1": 1.0,
        "kc2": 5.0,
        "theta_min": 0.101,
        "theta_max": 2.344,
    }

    def __init__(self, dof_index: int = 0, theta_bounds: tuple[float, float] = None, **parameters):
        """
        Parameters
        ----------
        dof_index: int
            The index of the degree of freedom the passive torque applies to
        theta_bounds: tuple[float, float]
            The angular range the torque is defined over.
        parameters: dict
            Any of k1, k2, k3, k4, kc1, kc2, theta_min, theta_max to override the default value
        """
        unknown = set(parameters) - set(self.DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown passive torque parameter(s): {sorted(unknown)}")

        self.dof_index = dof_index
        self.theta_bounds = theta_bounds
        for key, value in self.DEFAULTS.items():
            setattr(self, key, parameters.get(key, value))

    # ---- Torque ---- #
    def torque(self, theta, theta_dot):
        """
        The passive torque at the joint.

        Parameters
        ----------
        theta: float | np.ndarray | MX
            The joint angle (rad)
        theta_dot: float | np.ndarray | MX
            The joint angular velocity (rad/s)

        Returns
        -------
        The passive joint torque (N.m)
        """
        if self.theta_bounds is not None:
            theta = clip(theta, self.theta_bounds[0], self.theta_bounds[1])

        to_min = theta - self.theta_min
        to_max = theta - self.theta_max

        damping = sigmoid(-to_min / self.kc1) + sigmoid(to_max / self.kc2)

        return (
            self.k1 * exp(-self.k2 * to_min) * sigmoid(-to_min)
            - self.k3 * exp(self.k4 * to_max) * sigmoid(to_max)
            - damping * theta_dot
        )

    # ---- Parameter handling ---- #
    @property
    def identifiable_parameters(self) -> dict:
        """The current value of every parameter of the model."""
        return {key: getattr(self, key) for key in self.DEFAULTS}

    def set_k1(self, model, k1: MX | float):
        """Set the extension-side torque amplitude k1."""
        self.k1 = k1

    def set_k2(self, model, k2: MX | float):
        """Set the extension-side torque exponential rate k2."""
        self.k2 = k2

    def set_k3(self, model, k3: MX | float):
        """Set the flexion-side torque amplitude k3."""
        self.k3 = k3

    def set_k4(self, model, k4: MX | float):
        """Set the flexion-side torque exponential rate k4."""
        self.k4 = k4

    def set_kc1(self, model, kc1: MX | float):
        """Set the width of the extension-side viscous zone kc1."""
        self.kc1 = kc1

    def set_kc2(self, model, kc2: MX | float):
        """Set the width of the flexion-side viscous zone kc2."""
        self.kc2 = kc2

    def set_theta_min(self, model, theta_min: MX | float):
        """Set the joint extension limit theta_min (rad)."""
        self.theta_min = theta_min

    def set_theta_max(self, model, theta_max: MX | float):
        """Set the joint flexion limit theta_max (rad)."""
        self.theta_max = theta_max

    def default_parameter_settings(self, max_elbow_position: float) -> dict:
        """
        The initial guess, bounds, scaling and setter of every identifiable parameter.

        Parameters
        ----------
        max_elbow_position: float
            The measured maximal elbow flexion of the subject (rad), used to bound theta_max

        Returns
        -------
        dict
            The identification settings of every parameter
        """
        return {
            "k1": {"initial_guess": 3, "min_bound": 0.005, "max_bound": 100, "function": self.set_k1, "scaling": 1},
            "k2": {
                "initial_guess": 3,
                "min_bound": 0.005,
                "max_bound": END_STOP_RATE_MAX,
                "function": self.set_k2,
                "scaling": 1,
            },
            "k3": {"initial_guess": 3, "min_bound": 0.005, "max_bound": 100, "function": self.set_k3, "scaling": 1},
            "k4": {
                "initial_guess": 3,
                "min_bound": 0.005,
                "max_bound": END_STOP_RATE_MAX,
                "function": self.set_k4,
                "scaling": 1,
            },
            "kc1": {"initial_guess": 1, "min_bound": 0.1, "max_bound": 10, "function": self.set_kc1, "scaling": 1},
            "kc2": {"initial_guess": 1, "min_bound": 0.01, "max_bound": 6, "function": self.set_kc2, "scaling": 1},
            "theta_max": {
                "initial_guess": max_elbow_position,
                "min_bound": max_elbow_position - THETA_MAX_MARGIN,
                "max_bound": max_elbow_position + THETA_MAX_MARGIN,
                "function": self.set_theta_max,
                "scaling": 1,
            },
            "theta_min": {
                "initial_guess": 0,
                "min_bound": -np.deg2rad(10),
                "max_bound": np.deg2rad(100),
                "function": self.set_theta_min,
                "scaling": 1,
            },
        }

    # ---- Loading an identified set ---- #
    @classmethod
    def from_identification(cls, result: dict, dof_index: int = 0) -> "PassiveTorque":
        """
        Rebuild a passive torque model from a loaded result.

        Parameters
        ----------
        result: dict
            The result of a passive torque identification, as returned by results.load
        dof_index: int
            The index of the degree of freedom the passive torque applies to
        """
        identified_range = result.get("extra", {}).get("identified_range")
        return cls(
            dof_index=dof_index,
            theta_bounds=tuple(identified_range) if identified_range else None,
            **{key: float(value) for key, value in result["parameters"].items() if key in cls.DEFAULTS},
        )


class _WithPassiveTorque:
    """Mixin building the casadi Function of the passive joint torque of a biorbd based model."""

    def joint_passive_torque(self) -> Function:
        """
        The passive joint torque as a casadi Function of (q, qdot, parameters), the same signature as bioptim's own
        BiorbdModel.passive_joint_torque, so that it can be called with nlp.parameters.cx.
        """
        if self.passive_torque.theta_bounds is None:
            # Keep the torque finite when the integrator transiently steps out of the joint range
            q_ranges = self.bounds_from_ranges("q")
            dof = self.passive_torque.dof_index
            self.passive_torque.theta_bounds = (float(q_ranges.min[dof].min()), float(q_ranges.max[dof].max()))

        torque = self.passive_torque.torque(
            theta=self.q[self.passive_torque.dof_index],
            theta_dot=self.qdot[self.passive_torque.dof_index],
        )
        tau = vertcat(*[torque if i == self.passive_torque.dof_index else 0 for i in range(self.nb_tau)])

        return Function(
            "joint_passive_torque",
            [self.q, self.qdot, self.parameters],
            [tau],
            ["q", "qdot", "parameters"],
            ["joint_passive_torque"],
        )


class PassiveTorqueBiorbdModel(_WithPassiveTorque, TorqueBiorbdModel):
    """
    A torque driven biorbd model whose joint torque holds the identifiable passive torque.
    """

    def __init__(self, bio_model, passive_torque: PassiveTorque = None, parameters=None, **kwargs):
        # Set before the super call: bioptim applies the ocp parameters on the model during __init__
        self.passive_torque = passive_torque if passive_torque else PassiveTorque()
        super().__init__(bio_model=bio_model, parameters=parameters, **kwargs)

    def get_basic_variables(self, nlp, states, controls, parameters, algebraic_states, numerical_timeseries):
        q, qdot, tau, external_forces = super().get_basic_variables(
            nlp, states, controls, parameters, algebraic_states, numerical_timeseries
        )
        tau = tau + self.joint_passive_torque()(q, qdot, nlp.parameters.cx)
        return q, qdot, tau, external_forces


class PassiveTorqueFesMskModel(_WithPassiveTorque, FesMskModel):
    """
    A cocofest FES driven musculoskeletal model whose joint torque holds the passive torque identified.
    """

    def __init__(self, *args, passive_torque: PassiveTorque = None, **kwargs):
        self.passive_torque = passive_torque if passive_torque else PassiveTorque()
        super().__init__(*args, **kwargs)

    @staticmethod
    def muscles_joint_torque(
        time, states, controls, parameters, algebraic_states, numerical_data_timeseries, nlp, q=None, qdot=None
    ):
        muscle_joint_torques, dxdt_muscle_list = FesMskModel.muscles_joint_torque(
            time, states, controls, parameters, algebraic_states, numerical_data_timeseries, nlp, q, qdot
        )
        muscle_joint_torques = muscle_joint_torques + nlp.model.joint_passive_torque()(q, qdot, nlp.parameters.cx)
        return muscle_joint_torques, dxdt_muscle_list

