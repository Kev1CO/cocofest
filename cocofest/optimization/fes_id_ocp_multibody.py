import numpy as np

from bioptim import (
    BoundsList,
    InitialGuessList,
    InterpolationType,
    ParameterList,
    VariableScaling,
)

from ..models.fes_model import FesModel
from ..models.ding2003 import DingModelFrequency
from ..models.ding2007 import DingModelPulseWidthFrequency
from ..models.hmed2018 import DingModelPulseIntensityFrequency
from ..optimization.fes_ocp import OcpFes
from ..optimization.fes_id_ocp import OcpFesId
from ..optimization.fes_ocp_multibody import OcpFesMsk
from ..models.dynamical_model import FesMskModel


class OcpFesIdMultibody(OcpFesId):
    def __init__(self):
        super(OcpFesIdMultibody, self).__init__()
        self.parameters_ding2003 = ["a_rest", "km_rest", "tau1_rest", "tau2"]
        self.parameters_ding2007 = ["a_scale", "km_rest", "tau1_rest", "tau2", "pd0", "pdt"]
        self.parameters_hmed2018 = ["a_rest", "km_rest", "tau1_rest", "tau2", "ar", "bs", "Is", "cr"]

        self.list_key = ["initial_guess", "min_bound", "max_bound", "scaling"]

        self.dict_param_values = {
            "tau1_rest": {
                "initial_guess": 0.5,
                "min_bound": 0.0001,
                "max_bound": 1,
                "scaling": 1,  # 10000
            },
            "tau2": {
                "initial_guess": 0.5,
                "min_bound": 0.0001,
                "max_bound": 1,
                "scaling": 1,  # 10000
            },
            "km_rest": {
                "initial_guess": 0.5,
                "min_bound": 0.001,
                "max_bound": 1,
                "scaling": 1,  # 10000
            },
            "a_scale": {
                "initial_guess": 5000,
                "min_bound": 1,
                "max_bound": 10000,
                "scaling": 1,
            },
            "pd0": {
                "initial_guess": 1e-4,
                "min_bound": 1e-4,
                "max_bound": 6e-4,
                "scaling": 1,  # 1000
            },
            "pdt": {
                "initial_guess": 1e-4,
                "min_bound": 1e-4,
                "max_bound": 6e-4,
                "scaling": 1,  # 1000
            },
            "ar": {
                "initial_guess": 0.5,
                "min_bound": 0.01,
                "max_bound": 1,
                "scaling": 1,
            },  # 100
            "bs": {
                "initial_guess": 0.05,
                "min_bound": 0.001,
                "max_bound": 0.1,
                "scaling": 1,  # 1000
            },
            "Is": {
                "initial_guess": 50,
                "min_bound": 1,
                "max_bound": 150,
                "scaling": 1,
            },
            "cr": {
                "initial_guess": 1,
                "min_bound": 0.01,
                "max_bound": 2,
                "scaling": 1,
            },  # 100
            "a_rest": {
                "initial_guess": 1000,
                "min_bound": 1,
                "max_bound": 10000,
                "scaling": 1,
            }
        }

    @staticmethod
    def set_default_values(msk_model):
        if isinstance(msk_model, FesMskModel):
            model_list = msk_model.muscles_dynamics_model
            muscle_list = msk_model.muscle_names
            if isinstance(model_list[0], DingModelPulseWidthFrequency) and isinstance(model_list[1], DingModelPulseWidthFrequency):
                return {
                    "tau1_rest_BIClong": {
                        "initial_guess": 0.5,
                        "min_bound": 0.0001,
                        "max_bound": 1,
                        "function": model_list[0].set_tau1_rest,
                        "scaling": 1,  # 10000
                    },
                    "tau2_BIClong": {
                        "initial_guess": 0.5,
                        "min_bound": 0.0001,
                        "max_bound": 1,
                        "function": model_list[0].set_tau2,
                        "scaling": 1,  # 10000
                    },
                    "km_rest_BIClong": {
                        "initial_guess": 0.5,
                        "min_bound": 0.001,
                        "max_bound": 1,
                        "function": model_list[0].set_km_rest,
                        "scaling": 1,  # 10000
                    },
                    "a_scale_BIClong": {
                        "initial_guess": 5000,
                        "min_bound": 1,
                        "max_bound": 10000,
                        "function": model_list[0].set_a_scale,
                        "scaling": 1,
                    },
                    "pd0_BIClong": {
                        "initial_guess": 1e-4,
                        "min_bound": 1e-4,
                        "max_bound": 6e-4,
                        "function": model_list[0].set_pd0,
                        "scaling": 1,  # 1000
                    },
                    "pdt_BIClong": {
                        "initial_guess": 1e-4,
                        "min_bound": 1e-4,
                        "max_bound": 6e-4,
                        "function": model_list[0].set_pdt,
                        "scaling": 1,  # 1000
                    },
                    "tau1_rest_BICshort": {
                        "initial_guess": 0.5,
                        "min_bound": 0.0001,
                        "max_bound": 1,
                        "function": model_list[1].set_tau1_rest,
                        "scaling": 1,  # 10000
                    },
                    "tau2_BICshort": {
                        "initial_guess": 0.5,
                        "min_bound": 0.0001,
                        "max_bound": 1,
                        "function": model_list[1].set_tau2,
                        "scaling": 1,  # 10000
                    },
                    "km_rest_BICshort": {
                        "initial_guess": 0.5,
                        "min_bound": 0.001,
                        "max_bound": 1,
                        "function": model_list[1].set_km_rest,
                        "scaling": 1,  # 10000
                    },
                    "a_scale_BICshort": {
                        "initial_guess": 5000,
                        "min_bound": 1,
                        "max_bound": 10000,
                        "function": model_list[1].set_a_scale,
                        "scaling": 1,
                    },
                    "pd0_BICshort": {
                        "initial_guess": 1e-4,
                        "min_bound": 1e-4,
                        "max_bound": 6e-4,
                        "function": model_list[1].set_pd0,
                        "scaling": 1,  # 1000
                    },
                    "pdt_BICshort": {
                        "initial_guess": 1e-4,
                        "min_bound": 1e-4,
                        "max_bound": 6e-4,
                        "function": model_list[1].set_pdt,
                        "scaling": 1,  # 1000
                    },
                }
            else:
                raise ValueError("le modèle n'est pas Ding2007")

    def set_default_values_2(self, msk_model):
        if isinstance(msk_model, FesMskModel):
            dict = {}
            for model, muscle in zip(msk_model.muscles_dynamics_model, msk_model.muscle_names):
                if isinstance(model, DingModelPulseWidthFrequency):
                    for key in self.parameters_ding2007:
                        entry = key + "_" + muscle
                        dict[entry] = self.dict_param_values[key].copy()
                        dict[entry]["function"] = getattr(model, "set_" + key)
                elif isinstance(model, DingModelPulseIntensityFrequency):
                    for key in self.parameters_hmed2018:
                        entry = key + "_" + muscle
                        dict[entry] = self.dict_param_values[key].copy()
                        dict[entry]["function"] = getattr(model, "set_" + key)
                elif isinstance(model, DingModelFrequency):
                    for key in self.parameters_ding2003:
                        entry = key + "_" + muscle
                        dict[entry] = self.dict_param_values[key].copy()
                        dict[entry]["function"] = getattr(model, "set_" + key)
            return dict

    def set_u_bounds(self):
        pass
