import numpy as np

from cocofest import FesModel


class ModelBuilder:
    def __init__(self,
                 model=FesModel,
                 stim_time=None,
                 a_rest=None,
                 tau1_rest=None,
                 km_rest=None,
                 tau2=None,
                 alpha_a=None,
                 alpha_tau1=None,
                 alpha_km=None,
                 tau_fat=None,
                 a_scale=None,
                 pd0=None,
                 pdt=None,
                 ar=None,
                 bs=None,
                 Is=None,
                 cr=None,
                 ):

        self.stim_time = stim_time if stim_time is not None else []

        self.model = model
        self.a_rest = a_rest
        self.tau1_rest = tau1_rest
        self.km_rest = km_rest
        self.tau2 = tau2
        self.alpha_a = alpha_a
        self.alpha_tau1 = alpha_tau1
        self.alpha_km = alpha_km
        self.tau_fat = tau_fat
        self.a_scale = a_scale
        self.pd0 = pd0
        self.pdt = pdt
        self.ar = ar
        self.bs = bs
        self.Is = Is
        self.cr = cr

        self.models = self._create_models()

    def _create_models(self):
        # Create DingModel instances for each phase based on stim_time
        length = len(self.stim_time) if isinstance(self.stim_time, list) else self.stim_time.shape[0]
        phases = [self.stim_time[0:i+1] for i in range(length)]
        self.models = [self.model(time_stim_prev=phases[i][:-1] if i + 1 < len(phases) else [],
                                  time_current_stim=phases[i][-1:],
                                  a_rest=self.a_rest,
                                  tau1_rest=self.tau1_rest,
                                  km_rest=self.km_rest,
                                  tau2=self.tau2,
                                  alpha_a=self.alpha_a,
                                  alpha_tau1=self.alpha_tau1,
                                  alpha_km=self.alpha_km,
                                  tau_fat=self.tau_fat,
                                  a_scale=self.a_scale,
                                  pd0=self.pd0,
                                  pdt=self.pdt,
                                  ar=self.ar,
                                  bs=self.bs,
                                  Is=self.Is,
                                  cr=self.cr,
                                  ) for i in range(len(phases))]
        return self.models

    def build(self):
        return self.models

    def build_for_nmpc(self, final_time, n_simultaneous_cycles):
        if not isinstance(self.stim_time, list):
            raise TypeError("stim_time must be a list type, can not build nmpc from symbolic")
        nmpc_stim_time = list(np.array([np.array(self.stim_time) + final_time * i for i in range(3)]).flatten())
        length = len(nmpc_stim_time)
        phases = [nmpc_stim_time[0:i + 1] for i in range(length)]
        self.models = [self.model(time_stim_prev=phases[i][:-1],
                                  time_current_stim=phases[i][-1:],
                                  a_rest=self.a_rest,
                                  tau1_rest=self.tau1_rest,
                                  km_rest=self.km_rest,
                                  tau2=self.tau2,
                                  alpha_a=self.alpha_a,
                                  alpha_tau1=self.alpha_tau1,
                                  alpha_km=self.alpha_km,
                                  tau_fat=self.tau_fat,
                                  a_scale=self.a_scale,
                                  pd0=self.pd0,
                                  pdt=self.pdt,
                                  ar=self.ar,
                                  bs=self.bs,
                                  Is=self.Is,
                                  cr=self.cr,
                                  ) for i in range(len(phases))]
        return self.models

    def set_stim_times(self, stim):
        self.stim_time = stim
        # Update all models with the new stimulation times
        self.models = self._create_models()
