"""
Passive joint torque, Riener and Edrich formulation.

Riener, R., & Edrich, T. (1999). Identification of passive elastic joint moments in the lower extremities.
Journal of Biomechanics, 32(5), 539-544.

    tau_p(theta, theta_dot) = exp(a1 + a2 * theta) - exp(a3 + a4 * theta) + a5 - b * theta_dot
"""

from __future__ import annotations

import numpy as np
from casadi import MX, exp

from .passive_torque_model import clip


class RienerPassiveTorque:
    """The Riener and Edrich passive joint torque, with the same interface as PassiveTorque."""

    # Angles in radians
    DEFAULTS = {
        "a1": 1.0,
        "a2": -3.0,
        "a3": -2.0,
        "a4": 3.0,
        "a5": 0.0,
        "b": 0.1,
    }

    def __init__(self, dof_index: int = 0, theta_bounds: tuple[float, float] = None, **parameters):
        """
        Parameters
        ----------
        dof_index: int
            The index of the degree of freedom the passive torque applies to
        theta_bounds: tuple[float, float]
            The angular range the torque is defined over, outside of which it is held at its boundary value. Same
            role as in PassiveTorque: the exponentials are pure extrapolation away from the identification data.
        parameters: dict
            Any of a1, a2, a3, a4, a5, b to override the default value
        """
        unknown = set(parameters) - set(self.DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown Riener passive torque parameter(s): {sorted(unknown)}")

        self.dof_index = dof_index
        self.theta_bounds = theta_bounds
        for key, value in self.DEFAULTS.items():
            setattr(self, key, parameters.get(key, value))

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

        return exp(self.a1 + self.a2 * theta) - exp(self.a3 + self.a4 * theta) + self.a5 - self.b * theta_dot

    @property
    def identifiable_parameters(self) -> dict:
        """The current value of every parameter of the model."""
        return {key: getattr(self, key) for key in self.DEFAULTS}

    def set_a1(self, model, a1: MX | float):
        """Set the log amplitude of the extension side exponential."""
        self.a1 = a1

    def set_a2(self, model, a2: MX | float):
        """Set the rate of the extension side exponential (negative: it decays as the joint flexes)."""
        self.a2 = a2

    def set_a3(self, model, a3: MX | float):
        """Set the log amplitude of the flexion side exponential."""
        self.a3 = a3

    def set_a4(self, model, a4: MX | float):
        """Set the rate of the flexion side exponential (positive: it grows as the joint flexes)."""
        self.a4 = a4

    def set_a5(self, model, a5: MX | float):
        """Set the constant offset of the passive torque (N.m)."""
        self.a5 = a5

    def set_b(self, model, b: MX | float):
        """Set the viscous coefficient, active over the whole range (N.m.s/rad)."""
        self.b = b

    def default_parameter_settings(self, max_elbow_position: float) -> dict:
        """
        The initial guess, bounds, scaling and setter of every parameter.

        Parameters
        ----------
        max_elbow_position: float
            The measured maximal elbow flexion of the subject (rad). Unused here, the formulation carries no
            explicit limit angle, but kept in the signature so the two formulations are interchangeable.
        """
        return {
            "a1": {"initial_guess": 1.0, "min_bound": -5.0, "max_bound": 4.0, "function": self.set_a1, "scaling": 1},
            "a2": {"initial_guess": -3.0, "min_bound": -15.0, "max_bound": 0.0, "function": self.set_a2, "scaling": 1},
            "a3": {"initial_guess": -2.0, "min_bound": -5.0, "max_bound": 4.0, "function": self.set_a3, "scaling": 1},
            "a4": {"initial_guess": 3.0, "min_bound": 0.0, "max_bound": 15.0, "function": self.set_a4, "scaling": 1},
            "a5": {"initial_guess": 0.0, "min_bound": -10.0, "max_bound": 10.0, "function": self.set_a5, "scaling": 1},
            "b": {"initial_guess": 0.1, "min_bound": 0.0, "max_bound": 5.0, "function": self.set_b, "scaling": 1},
        }

    @classmethod
    def from_identification(cls, result: dict, dof_index: int = 0) -> "RienerPassiveTorque":
        """Rebuild the model from a result loaded with helper/results.py."""
        identified_range = result.get("extra", {}).get("identified_range")
        return cls(
            dof_index=dof_index,
            theta_bounds=tuple(identified_range) if identified_range else None,
            **{key: float(value) for key, value in result["parameters"].items() if key in cls.DEFAULTS},
        )
