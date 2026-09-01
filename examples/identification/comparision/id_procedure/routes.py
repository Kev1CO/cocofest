"""
The identified methods of the study.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import store


COST_UNITS = {"force": ("N", 1.0), "torque": ("Nm", 1.0), "angle": ("deg", 180 / np.pi)}


@dataclass(frozen=True)
class Route:
    """One identification route, as steps.py reads it."""

    METHOD: str
    OBSERVABLE: str
    KINEMATICS: str
    COST: str
    YLABEL: str
    label: str
    colour: str
    family: str = "isometric"
    TAU1: str = "relaxation"
    PASSIVE_TORQUE: str | None = None

    @property
    def UNIT(self) -> str:
        """What this route identifies in, settled by its cost."""
        return COST_UNITS[self.COST][0]

    @property
    def SCALE(self) -> float:
        """What the tracked state is multiplied by to be read in UNIT, ex: 180/pi for an angle."""
        return COST_UNITS[self.COST][1]

    def load(self, subject, debug=False, frequencies=None, **kwargs) -> list:
        """Every usable train (isometric) or flexion (movement) of one subject, as phases."""
        import observables

        if self.family == "isometric":
            return observables.isometric_phases(subject, observable=self.OBSERVABLE)
        return observables.movement_phases(subject, frequencies=frequencies,
                                           observable=self.OBSERVABLE, debug=debug)


ROUTES = (
    Route(METHOD="sensor_id", OBSERVABLE="sensor", KINEMATICS="none", COST="force",
          YLABEL="Sensor force (N)", label="sensor force", colour="#8c564b"),
    Route(METHOD="muscle_force_isometric_id", OBSERVABLE="muscle", KINEMATICS="prescribed", COST="force",
          YLABEL="Muscle force (N)", label="muscle force", colour="tab:blue"),
    Route(METHOD="movement_id", OBSERVABLE="angle", KINEMATICS="free", COST="angle",
          YLABEL="Elbow angle (deg)", label="elbow angle, generic joint", colour="tab:green",
          family="movement", PASSIVE_TORQUE="default"),

    # Tested but not kept for further analysis
    # Route(METHOD="force_ik_movement_id", OBSERVABLE="force", KINEMATICS="prescribed", COST="force",
    #       YLABEL="Muscle force (N)", label="ID force", colour="tab:orange", family="movement"),
    # Route(METHOD="torque_ik_movement_id", OBSERVABLE="torque", KINEMATICS="prescribed", COST="torque",
    #       YLABEL="Elbow torque (Nm)", label="ID torque", colour="tab:red", family="movement"),

    # Route(METHOD="movement_riener_id", OBSERVABLE="angle", KINEMATICS="free", COST="angle",
    #       YLABEL="Elbow angle (deg)", label="elbow angle, own joint", colour="tab:olive",
    #       family="movement", PASSIVE_TORQUE="identified"),
)

BY_NAME = {route.METHOD: route for route in ROUTES}
NAMES = tuple(BY_NAME)

EXPENSIVE = ("movement_riener_id",)
ANALYSED = tuple(name for name in NAMES if name not in EXPENSIVE)


def of(name) -> Route:
    """The route behind a route name."""
    return BY_NAME[name]
