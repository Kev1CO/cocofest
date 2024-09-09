from dataclasses import dataclass
import numpy as np
from cocofest import FesModel


@dataclass
class ModelConfig:
    """
    Data class to store all the parameter configurations for a model.
    If not specified, the default values will be those defined in the model.

    Attributes
    ----------
    #--- For all models ---#
    model: FesModel
        The model to be used
    a_rest: float
        Force scaling factor
    tau1_rest: float
        Force decline time constant when strongly bound cross-bridges are absent
    km_rest: float
        Ca2+ -troponin complex's strongly bound cross-bridges sensitivity
    tau2: float
        Force decline time constant due to friction between actin and myosin
    alpha_a: float
        Force-model coefficient for a_rest in the fatigue model
    alpha_tau1: float
        Force-model coefficient for tau1_rest in the fatigue model
    alpha_km: float
        Force-model coefficient for km_rest in the fatigue model
    tau_fat: float
        Time constant controlling the recovery of states A, Km, Tau1
    #--- For Ding2007 models ---#
    a_scale: float
        Force scaling factor use in Ding's 2007 models
    pd0: float
        Stimulation offset characterizing how sensitive the muscle is to the pulse duration
    pdt: float
        Time constant controlling the steepness of the A-pd relationship
    #--- For Hmed2018 models ---#
    ar: float
        Translation of axis coordinates
    bs: float
        Fiber muscle recruitment constant identification
    Is: float
        Muscle saturation intensity
    cr: float
        Translation of axis coordinates
    #--- For FesMsk models ---#
    biorbd_model_path: str
        Path to the biorbd model
    fes_muscle_models: any
        Preconfigured Fes models
    activate_force_length_relationship: bool
        Activate force length relationship
    activate_force_velocity_relationship: bool
        Activate force velocity relationship
    """
    model: any = FesModel
    a_rest: float = None
    tau1_rest: float = None
    km_rest: float = None
    tau2: float = None
    alpha_a: float = None
    alpha_tau1: float = None
    alpha_km: float = None
    tau_fat: float = None
    a_scale: float = None
    pd0: float = None
    pdt: float = None
    ar: float = None
    bs: float = None
    Is: float = None
    cr: float = None
    biorbd_model_path: str = None
    fes_muscle_models: any = None
    activate_force_length_relationship: bool = False
    activate_force_velocity_relationship: bool = False


class ModelBuilder:
    """
    Class to build FesModel instances for different stimulation phases.

    Attributes
    ----------
    config : ModelConfig
        Configuration for the model parameters.
    stim_time : np.ndarray
        Array of stimulation times. If a list is passed, it will be converted to a numpy array.
    """

    def __init__(self, config: ModelConfig, stim_time: list = None):
        self.stim_time = np.array(stim_time) if stim_time is not None else np.array([])
        self.model = config.model
        self.config = config

    def _build_model(self, phases) -> list:
        """
        Internal method to build model instances for each phase.

        Parameters
        ----------
        phases : list
            List of phases based on stimulation times.

        Returns
        -------
        list : List of FesModel instances.
        """
        return [self.config.model(time_stim_prev=phases[i][:-1],
                                  time_current_stim=phases[i][-1:],
                                  a_rest=self.config.a_rest,
                                  tau1_rest=self.config.tau1_rest,
                                  km_rest=self.config.km_rest,
                                  tau2=self.config.tau2,
                                  alpha_a=self.config.alpha_a,
                                  alpha_tau1=self.config.alpha_tau1,
                                  alpha_km=self.config.alpha_km,
                                  tau_fat=self.config.tau_fat,
                                  a_scale=self.config.a_scale,
                                  pd0=self.config.pd0,
                                  pdt=self.config.pdt,
                                  ar=self.config.ar,
                                  bs=self.config.bs,
                                  Is=self.config.Is,
                                  cr=self.config.cr,
                                  biorbd_path=self.config.biorbd_model_path,
                                  muscles_model=self.config.fes_muscle_models,
                                  activate_force_length_relationship=self.config.activate_force_length_relationship,
                                  activate_force_velocity_relationship=self.config.activate_force_velocity_relationship)
                for i in range(len(phases))]

    def set_stim_times(self, stim):
        """
        Set new stimulation times and update the model.

        Parameters
        ----------
        stim : list
            List of new stimulation times.
        """
        self.stim_time = np.array(stim)

    def build(self, cycle_final_time: float = None) -> list:
        """
        Build a list of models for each stimulation phase based on the stimulation times.

        Parameters
        ----------
        cycle_final_time : float
            Final time of a unique cycle for nmpc purposes.
        Returns
        -------
        list : List of model instances created for each stimulation phase.
        """
        stimulation_times = self.stim_time
        if cycle_final_time:
            stimulation_times = np.concatenate([np.array(self.stim_time) + cycle_final_time * i for i in range(3)])
            # range 3 refers to the number of simultaneous cycles in nmpc, TODO: change once more than 3 cycles is possible
        phases = [stimulation_times[0:i + 1] for i in range(len(stimulation_times))]
        return self._build_model(phases)
