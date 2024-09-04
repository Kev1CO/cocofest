from copy import copy

from cocofest import (FesModel,
                      DingModelFrequency,
                      DingModelPulseDurationFrequency,
                      DingModelIntensityFrequency)


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

        self.model = model()
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

        self._set_model_values()
        self.models = self._create_models()

    def _set_model_values(self):
        if isinstance(self.model, DingModelFrequency):
            if self.a_rest:
                self.model.set_a_rest(self.model, a_rest=self.a_rest)
            if self.tau1_rest:
                self.model.set_tau1_rest(self.model, tau1_rest=self.tau1_rest)
            if self.km_rest:
                self.model.set_km_rest(self.model, km_rest=self.km_rest)
            if self.tau2:
                self.model.set_tau2(self.model, tau2=self.tau2)

        elif isinstance(self.model, DingModelPulseDurationFrequency):
            if self.a_scale:
                self.model.set_a_scale(self.model, a_scale=self.a_scale)
            if self.pd0:
                self.model.set_pd0(self.model, pd0=self.pd0)
            if self.pdt:
                self.model.set_pdt(self.model, pdt=self.pdt)

        elif isinstance(self.model, DingModelIntensityFrequency):
            if self.ar:
                self.model.set_ar(self.model, ar=self.ar)
            if self.bs:
                self.model.set_bs(self.model, bs=self.bs)
            if self.Is:
                self.model.set_Is(self.model, Is=self.Is)
            if self.cr:
                self.model.set_cr(self.model, cr=self.cr)

        if self.model.with_fatigue:
            if self.alpha_a:
                self.model.set_alpha_a(self.model, alpha_a=self.alpha_a)
            if self.alpha_tau1:
                self.model.set_alpha_tau1(self.model, alpha_tau1=self.alpha_tau1)
            if self.alpha_km:
                self.model.set_alpha_km(self.model, alpha_km=self.alpha_km)
            if self.tau_fat:
                self.model.set_tau_fat(self.model, tau_fat=self.tau_fat)

    def _create_models(self):
        # Create DingModel instances for each phase based on stim_time
        length = len(self.stim_time) if isinstance(self.stim_time, list) else self.stim_time.shape[0]
        phases = [self.stim_time[0:i+1] for i in range(length)]
        self.models = [copy(self.model) for i in range(len(phases))]
        for i in range(len(phases)):
            self.models[i].set_stim_prev(phases[i][:-1] if i + 1 < len(phases) else [])
            self.models[i].set_current_stim(phases[i][-1:])
        return self.models

    def build(self):
        return self.models

    def set_stim_times(self, stim):
        self.stim_time = stim
        # Update all models with the new stimulation times
        self.models = self._create_models()
