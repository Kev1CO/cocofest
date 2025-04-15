from typing import Callable

from casadi import MX, vertcat, tanh
import numpy as np

from bioptim import (
    ConfigureProblem,
    DynamicsEvaluation,
    NonLinearProgram,
    OptimalControlProgram,
)
from .ding2003 import DingModelFrequency
from .state_configure import StateConfigure


class DingModelPulseIntensityFrequency(DingModelFrequency):
    """
    This is a custom models that inherits from bioptim. CustomModel.
    As CustomModel is an abstract class, some methods are mandatory and must be implemented.
    Such as serialize, name_dof, nb_state.

    This is the Hmed 2018 model using the stimulation frequency and pulse intensity in input.

    Hmed, A. B., Bakir, T., Garnier, Y. M., Sakly, A., Lepers, R., & Binczak, S. (2018).
    An approach to a muscle force model with force-pulse amplitude relationship of human quadriceps muscles.
    Computers in Biology and Medicine, 101, 218-228.
    """

    def __init__(
        self,
        model_name: str = "hmed2018",
        muscle_name: str = None,
        stim_time: list[float] = None,
        previous_stim: dict = None,
        sum_stim_truncation: int = 20,
    ):
        if previous_stim:
            if len(previous_stim["time"]) != len(previous_stim["pulse_intensity"]):
                raise ValueError("The previous_stim time and pulse_intensity must have the same length")
        super(DingModelPulseIntensityFrequency, self).__init__(
            model_name=model_name,
            muscle_name=muscle_name,
            stim_time=stim_time,
            previous_stim=previous_stim,
            sum_stim_truncation=sum_stim_truncation,
        )
        self._with_fatigue = False
        self.stim_pulse_intensity_prev = []
        self.previous_stim = previous_stim if previous_stim else {"time": [], "pulse_intensity": []}

        # --- Default values ---#
        AR_DEFAULT = 0.586  # (-) Translation of axis coordinates.
        BS_DEFAULT = 0.026  # (-) Fiber muscle recruitment constant identification.
        IS_DEFAULT = 63.1  # (mA) Muscle saturation intensity.
        CR_DEFAULT = 0.833  # (-) Translation of axis coordinates.

        # ---- Custom values for the example ---- #
        # ---- Force models ---- #
        self.ar = AR_DEFAULT
        self.bs = BS_DEFAULT
        self.Is = IS_DEFAULT
        self.cr = CR_DEFAULT
        self.impulse_intensity = None

    @property
    def identifiable_parameters(self):
        return {
            "a_rest": self.a_rest,
            "tau1_rest": self.tau1_rest,
            "km_rest": self.km_rest,
            "tau2": self.tau2,
            "ar": self.ar,
            "bs": self.bs,
            "Is": self.Is,
            "cr": self.cr,
        }

    @property
    def pulse_intensity_name(self):
        muscle_name = "_" + self.muscle_name if self.muscle_name else ""
        return "pulse_intensity" + muscle_name

    def set_ar(self, model, ar: MX | float):
        # models is required for bioptim compatibility
        self.ar = ar

    def set_bs(self, model, bs: MX | float):
        self.bs = bs

    def set_Is(self, model, Is: MX | float):
        self.Is = Is

    def set_cr(self, model, cr: MX | float):
        self.cr = cr

    def get_lambda_i(self, nb_stim: int, pulse_intensity: MX | float) -> list[MX | float]:
        return [self.lambda_i_calculation(pulse_intensity[i]) for i in range(nb_stim)]

    # ---- Absolutely needed methods ---- #
    def serialize(self) -> tuple[Callable, dict]:
        # This is where you can serialize your models
        # This is useful if you want to save your models and load it later
        return (
            DingModelPulseIntensityFrequency,
            {
                "tauc": self.tauc,
                "a_rest": self.a_rest,
                "tau1_rest": self.tau1_rest,
                "km_rest": self.km_rest,
                "tau2": self.tau2,
                "ar": self.ar,
                "bs": self.bs,
                "Is": self.Is,
                "cr": self.cr,
            },
        )

    def system_dynamics(
        self,
        cn: MX,
        f: MX,
        t: MX = None,
        t_stim_prev: list[MX] | list[float] = None,
        pulse_intensity: list[MX] | list[float] = None,
        force_length_relationship: MX | float = 1,
        force_velocity_relationship: MX | float = 1,
        passive_force_relationship: MX | float = 0,
    ) -> MX:
        """
        The system dynamics is the function that describes the models.

        Parameters
        ----------
        cn: MX
            The value of the ca_troponin_complex (unitless)
        f: MX
            The value of the force (N)
        t: MX
            The current time at which the dynamics is evaluated (s)
        t_stim_prev: list[MX] | list[float]
            The previous stimulation time (s)
        pulse_intensity: list[MX] | list[float]
            The pulsation intensity of the current stimulation (mA)
        force_length_relationship: MX | float
            The force length relationship value (unitless)
        force_velocity_relationship: MX | float
            The force velocity relationship value (unitless)
        passive_force_relationship: MX | float
            The passive force relationship value (unitless)

        Returns
        -------
        The value of the derivative of each state dx/dt at the current time t
        """
        cn_dot = self.calculate_cn_dot(cn, t, t_stim_prev, pulse_intensity)
        f_dot = self.f_dot_fun(
            cn,
            f,
            self.a_rest,
            self.tau1_rest,
            self.km_rest,
            force_length_relationship=force_length_relationship,
            force_velocity_relationship=force_velocity_relationship,
            passive_force_relationship=passive_force_relationship,
        )  # Equation n°2
        return vertcat(cn_dot, f_dot)

    def lambda_i_calculation(self, pulse_intensity: MX):
        """
        Parameters
        ----------
        pulse_intensity: MX
            The pulsation intensity of the current stimulation (mA)
        Returns
        -------
        The lambda factor, part of the n°1 equation
        """
        lambda_i = self.ar * (tanh(self.bs * (pulse_intensity - self.Is)) + self.cr)  # equation include intensity
        return lambda_i

    @staticmethod
    def lambda_i_calculation_identification(
        pulse_intensity: MX, ar: MX | float, bs: MX | float, Is: MX | float, cr: MX | float
    ):
        """
        Parameters
        ----------
        pulse_intensity: MX
            The pulsation intensity of the current stimulation (mA)
        ar: MX | float
            Translation of axis coordinates (-)
        bs: MX | float
            Fiber muscle recruitment constant identification.
        Is: MX | float
            Muscle saturation intensity (mA)
        cr: MX | float
            Translation of axis coordinates (-)
        Returns
        -------
        The lambda factor, part of the n°1 equation
        """
        lambda_i = ar * (tanh(bs * (pulse_intensity - Is)) + cr)  # equation include intensity
        return lambda_i

    def set_impulse_intensity(self, value: MX):
        """
        Sets the impulse intensity for each pulse (phases) according to the ocp parameter "impulse_intensity"

        Parameters
        ----------
        value: MX
            The pulsation intensity list (s)
        """
        self.impulse_intensity = []
        for i in range(value.shape[0]):
            self.impulse_intensity.append(value[i])

    @staticmethod
    def dynamics(
        time: MX,
        states: MX,
        controls: MX,
        parameters: MX,
        algebraic_states: MX,
        numerical_timeseries: MX,
        nlp: NonLinearProgram,
        fes_model: NonLinearProgram = None,
        force_length_relationship: MX | float = 1,
        force_velocity_relationship: MX | float = 1,
        passive_force_relationship: MX | float = 0,
    ) -> DynamicsEvaluation:
        """
        Functional electrical stimulation dynamic

        Parameters
        ----------
        time: MX
            The system's current node time
        states: MX
            The state of the system CN, F, A, Tau1, Km
        controls: MX
            The controls of the system, none
        parameters: MX
            The parameters acting on the system, final time of each phase
        algebraic_states: MX
            The stochastic variables of the system, none
        numerical_timeseries: MX
            The numerical timeseries of the system
        nlp: NonLinearProgram
            A reference to the phase
        fes_model: DingModelPulseIntensityFrequency
            The current phase fes model
        force_length_relationship: MX | float
            The force length relationship value (unitless)
        force_velocity_relationship: MX | float
            The force velocity relationship value (unitless)
        passive_force_relationship: MX | float
            The passive force relationship value (unitless)
        Returns
        -------
        The derivative of the states in the tuple[MX] format
        """
        model = fes_model if fes_model else nlp.model
        dxdt_fun = model.system_dynamics

        return DynamicsEvaluation(
            dxdt=dxdt_fun(
                cn=states[0],
                f=states[1],
                t=time,
                t_stim_prev=numerical_timeseries,
                pulse_intensity=controls,
                force_length_relationship=force_length_relationship,
                force_velocity_relationship=force_velocity_relationship,
                passive_force_relationship=passive_force_relationship,
            ),
            defects=None,
        )

    def declare_ding_variables(
        self,
        ocp: OptimalControlProgram,
        nlp: NonLinearProgram,
        numerical_data_timeseries: dict[str, np.ndarray] = None,
        contact_type: list = (),
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
        contact_type: list
            A list of contact types. This is used to define the contact forces in the dynamics. Not used in this model.
        """
        StateConfigure().configure_all_fes_model_states(ocp, nlp, fes_model=self)
        StateConfigure().configure_pulse_intensity(ocp, nlp, truncation=self.sum_stim_truncation)
        ConfigureProblem.configure_dynamics_function(ocp, nlp, dyn_func=self.dynamics)

    def min_pulse_intensity(self):
        """
        Returns
        -------
        The minimum pulse intensity threshold of the model
        For lambda_i = ar * (tanh(bs * (pulse_intensity - Is)) + cr) > 0
        """
        return (np.arctanh(-self.cr) / self.bs) + self.Is

    def _get_additional_previous_stim_time(self):
        while len(self.previous_stim["time"]) < self.sum_stim_truncation:
            self.previous_stim["time"].insert(0, -10000000)
            self.previous_stim["pulse_intensity"].insert(0, 50)
        return self.previous_stim
