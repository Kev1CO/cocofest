import re
import pickle
import os

import numpy as np
import pytest

from bioptim import Solver, Shooting, SolutionIntegrator, Solution
from cocofest import (
    OcpFesId,
    IvpFes,
    DingModelFrequency,
    DingModelPulseDurationFrequency,
    DingModelIntensityFrequency,
    DingModelFrequencyWithFatigue,
    DingModelFrequencyForceParameterIdentification,
)

model = DingModelFrequency
stim = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
discontinuity = []
n_shooting = [10, 10, 10, 10, 10, 10, 10, 10, 10]
final_time_phase = (0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)
force_at_node = [
    0.0,
    15.854417817548697,
    36.352691513848825,
    54.97388526038394,
    70.82053020534033,
    83.4194867436611,
    92.41942711195031,
    97.59278680323499,
    98.91780522472382,
    96.66310621259122,
    91.40987115240357,
    99.06445243586181,
    111.71243288393804,
    123.15693012605658,
    132.377564374587,
    138.80824696656194,
    142.02210466592646,
    141.73329382731202,
    137.8846391765497,
    130.73845520335917,
    120.89955520685814,
    125.38085346839195,
    135.5102503776543,
    144.68352158676947,
    151.82103024326696,
    156.32258873146037,
    157.7352209679825,
    155.7522062101244,
    150.30163417046595,
    141.63941942299235,
    130.37366930327116,
    133.8337968502193,
    143.14339960917405,
    151.5764058207702,
    158.03337418092343,
    161.90283357182437,
    162.7235878298402,
    160.18296049789234,
    154.20561770076475,
    145.04700073978324,
    133.31761023027764,
    136.45613333201518,
    145.51077095073057,
    153.71376988470408,
    159.9593525193574,
    163.63249745182017,
    164.26944493409266,
    161.5556743081453,
    155.41480228777354,
    146.102138099759,
    134.2289338756813,
    137.26784246878907,
    146.24355077243675,
    154.37534966572915,
    160.55549761772778,
    164.16787361651103,
    164.74792419884477,
    161.980557871029,
    155.7890666284376,
    146.42871894433975,
    134.51099963420057,
    137.5190757238638,
    146.47035441998682,
    154.58011604419679,
    160.74001118470105,
    164.33357848128765,
    164.89601880480728,
    162.11206398140135,
    155.90490550332385,
    146.52979923122618,
    134.5983019993197,
    137.59683509441382,
    146.54055256585733,
    154.64349341999517,
    160.7971200991939,
    164.3848659017574,
    164.94185566229874,
    162.15276652256654,
    155.9407588680354,
    146.56108465562548,
    134.62532301012072,
    137.6209024468235,
    146.56227963878422,
    154.66310939202043,
    160.8147959157575,
    164.4007399039886,
    164.95604265730248,
    162.16536439204623,
    155.9518558658154,
    146.5707678272839,
]

additional_key_settings = {
    "a_rest": {
        "initial_guess": 1000,
        "min_bound": 1,
        "max_bound": 10000,
        "function": model.set_a_rest,
        "scaling": 1,
    },
    "km_rest": {
        "initial_guess": 0.5,
        "min_bound": 0.001,
        "max_bound": 1,
        "function": model.set_km_rest,
        "scaling": 1000,
    },
    "tau1_rest": {
        "initial_guess": 0.5,
        "min_bound": 0.0001,
        "max_bound": 1,
        "function": model.set_tau1_rest,
        "scaling": 1000,
    },
    "tau2": {
        "initial_guess": 0.5,
        "min_bound": 0.0001,
        "max_bound": 1,
        "function": model.set_tau2,
        "scaling": 1000,
    },
}


def test_ocp_id():
    # --- Setting simulation parameters --- #
    n_stim = 10
    n_shooting = 10
    final_time = 1
    extra_phase_time = 1

    # --- Creating the simulated data to identify on --- #
    # Building the Initial Value Problem
    ivp = IvpFes(
        model=DingModelFrequency(),
        n_stim=n_stim,
        n_shooting=n_shooting,
        final_time=final_time,
        use_sx=True,
        extend_last_phase=extra_phase_time,
    )

    # Creating the solution from the initial guess
    sol_from_initial_guess = Solution.from_initial_guess(ivp, [ivp.x_init, ivp.u_init, ivp.p_init, ivp.s_init])

    # Integrating the solution
    result = sol_from_initial_guess.integrate(
        shooting_type=Shooting.SINGLE, integrator=SolutionIntegrator.OCP, merge_phases=True
    )

    force = result.states["F"]
    force = force.tolist()
    time = [result.time.tolist()]
    stim_temp = [0 if i == 0 else result.ocp.nlp[i].tf for i in range(len(result.ocp.nlp))]
    stim = [sum(stim_temp[: i + 1]) for i in range(len(stim_temp))]

    # Saving the data in a pickle file
    dictionary = {
        "time": time,
        "force": force,
        "stim_time": stim,
    }

    pickle_file_name = "temp_identification_simulation.pkl"
    with open(pickle_file_name, "wb") as file:
        pickle.dump(dictionary, file)

    # --- Identifying the model parameters --- #
    ocp = DingModelFrequencyForceParameterIdentification(
        model=DingModelFrequency(),
        data_path=[pickle_file_name],
        identification_method="full",
        identification_with_average_method_initial_guess=False,
        key_parameter_to_identify=["a_rest", "km_rest", "tau1_rest", "tau2"],
        additional_key_settings={},
        n_shooting=n_shooting,
        use_sx=True,
    )

    identification_result = ocp.force_model_identification()

    # --- Delete the temp file ---#
    os.remove(f"temp_identification_simulation.pkl")

    np.testing.assert_almost_equal(identification_result["a_rest"], model().a_rest, decimal=0)
    np.testing.assert_almost_equal(identification_result["km_rest"], model().km_rest, decimal=3)
    np.testing.assert_almost_equal(identification_result["tau1_rest"], model().tau1_rest, decimal=3)
    np.testing.assert_almost_equal(identification_result["tau2"], model().tau2, decimal=3)


def test_all_ocp_id_errors():
    # with pytest.raises(
    #     ValueError, match="a_rest, km_rest, tau1_rest and tau2 must be set for fatigue model identification"
    # ):
    #     OcpFesId.prepare_ocp(model=DingModelFrequencyWithFatigue(), a_rest=3009, km_rest=0.103, tau1_rest=0.050957)

    # a_rest = "3009"
    # with pytest.raises(
    #     TypeError, match=re.escape(f"a_rest must be int or float type," f" currently a_rest is {type(a_rest)}) type.")
    # ):
    #     OcpFesId.prepare_ocp(
    #         model=DingModelFrequencyWithFatigue(), a_rest=a_rest, km_rest=0.103, tau1_rest=0.050957, tau2=0.06
    #     )
    #
    # km_rest = "0.103"
    # with pytest.raises(
    #     TypeError,
    #     match=re.escape(f"km_rest must be int or float type," f" currently km_rest is {type(km_rest)}) type."),
    # ):
    #     OcpFesId.prepare_ocp(
    #         model=DingModelFrequencyWithFatigue(), a_rest=3009, km_rest=km_rest, tau1_rest=0.050957, tau2=0.06
    #     )
    #
    # tau1_rest = "0.050957"
    # with pytest.raises(
    #     TypeError,
    #     match=re.escape(f"tau1_rest must be int or float type," f" currently tau1_rest is {type(tau1_rest)}) type."),
    # ):
    #     OcpFesId.prepare_ocp(
    #         model=DingModelFrequencyWithFatigue(), a_rest=3009, km_rest=0.103, tau1_rest=tau1_rest, tau2=0.06
    #     )
    #
    # tau2 = "0.06"
    # with pytest.raises(
    #     TypeError, match=re.escape(f"tau2 must be int or float type," f" currently tau2 is {type(tau2)}) type.")
    # ):
    #     OcpFesId.prepare_ocp(
    #         model=DingModelFrequencyWithFatigue(), a_rest=3009, km_rest=0.103, tau1_rest=0.050957, tau2=tau2
    #     )

    n_shooting = "10"
    with pytest.raises(
        TypeError,
        match=re.escape(f"n_shooting must be list type," f" currently n_shooting is {type(n_shooting)}) type."),
    ):
        OcpFesId.prepare_ocp(model=DingModelFrequency(), n_shooting=n_shooting)

    n_shooting = [10, 10, 10, 10, 10, 10, 10, 10, "10"]
    with pytest.raises(TypeError, match=re.escape(f"n_shooting must be list of int type.")):
        OcpFesId.prepare_ocp(model=DingModelFrequency(), n_shooting=n_shooting)

    final_time_phase = "0.1"
    with pytest.raises(
        TypeError,
        match=re.escape(
            f"final_time_phase must be tuple type," f" currently final_time_phase is {type(final_time_phase)}) type."
        ),
    ):
        OcpFesId.prepare_ocp(
            model=DingModelFrequency(),
            n_shooting=[10, 10, 10, 10, 10, 10, 10, 10, 10],
            final_time_phase=final_time_phase,
        )

    force_tracking = "10"
    with pytest.raises(
        TypeError,
        match=re.escape(
            f"force_tracking must be list type," f" currently force_tracking is {type(force_tracking)}) type."
        ),
    ):
        OcpFesId.prepare_ocp(
            model=DingModelFrequency(),
            n_shooting=[10, 10, 10, 10, 10, 10, 10, 10, 10],
            final_time_phase=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
            force_tracking=force_tracking,
        )

    force_tracking = [10, 10, 10, 10, 10, 10, 10, 10, "10"]
    with pytest.raises(TypeError, match=re.escape(f"force_tracking must be list of int or float type.")):
        OcpFesId.prepare_ocp(
            model=DingModelFrequency(),
            n_shooting=[10, 10, 10, 10, 10, 10, 10, 10, 10],
            final_time_phase=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
            force_tracking=force_tracking,
        )

    pulse_duration = 0.001
    with pytest.raises(
        TypeError,
        match=re.escape(f"pulse_duration must be list type, currently pulse_duration is {type(pulse_duration)}) type."),
    ):
        OcpFesId.prepare_ocp(
            model=DingModelPulseDurationFrequency(),
            n_shooting=[10, 10, 10, 10, 10, 10, 10, 10, 10],
            final_time_phase=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
            force_tracking=[10, 10, 10, 10, 10, 10, 10, 10, 10],
            pulse_duration=pulse_duration,
        )

    pulse_intensity = 20
    with pytest.raises(
        TypeError,
        match=re.escape(
            f"pulse_intensity must be list type, currently pulse_intensity is {type(pulse_intensity)}) type."
        ),
    ):
        OcpFesId.prepare_ocp(
            model=DingModelIntensityFrequency(),
            n_shooting=[10, 10, 10, 10, 10, 10, 10, 10, 10],
            final_time_phase=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
            force_tracking=[10, 10, 10, 10, 10, 10, 10, 10, 10],
            pulse_intensity=pulse_intensity,
        )
