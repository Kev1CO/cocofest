"""
THIS IS NOT CONVERGING TO THE EXPECTED RESULT - SEE ISSUE #11
--
This example demonstrates the way of identifying the Hmed2018 model parameter using noisy simulated data.
First we integrate the model with a given parameter set. Then we add noise to the previously calculated force output.
Finally, we use the noisy data to identify the model parameters.
"""

import pickle
import os
import matplotlib.pyplot as plt
import numpy as np

from bioptim import SolutionMerge, OdeSolver

from cocofest import (
    ModelMaker,
    DingModelPulseIntensityFrequency,
    DingModelPulseIntensityFrequencyForceParameterIdentification,
    IvpFes,
)
from cocofest.identification.identification_method import full_data_extraction


def simulate_data(final_time=2, stim_time=None, pulse_intensity=None):
    ivp_model = ModelMaker.create_model("hmed2018", stim_time=stim_time, sum_stim_truncation=10)
    fes_parameters = {"model": ivp_model, "stim_time": stim_time, "pulse_intensity": pulse_intensity}
    ivp_parameters = {"final_time": final_time, "use_sx": True}

    # Building the Initial Value Problem
    ivp = IvpFes(fes_parameters, ivp_parameters)

    # Integrating the solution
    result, time = ivp.integrate()

    # Adding noise to the force
    noise = np.random.normal(0, 0.5, len(result["F"][0]))
    force = result["F"][0]  # + noise

    # Saving the data in a pickle file
    dictionary = {"time": time, "force": force, "stim_time": stim_time, "pulse_intensity": pulse_intensity}

    pickle_file_name = "../data/temp_identification_simulation.pkl"
    with open(pickle_file_name, "wb") as file:
        pickle.dump(dictionary, file)

    return pickle_file_name


def prepare_ocp(pickle_file_name, final_time=2, stim_time=None, pulse_intensity=None):
    ocp_model = DingModelPulseIntensityFrequency(stim_time=stim_time, sum_stim_truncation=10)
    return DingModelPulseIntensityFrequencyForceParameterIdentification(
        model=ocp_model,
        data_path=[pickle_file_name],
        identification_method="full",
        double_step_identification=False,
        key_parameter_to_identify=[
            "a_rest",
            "km_rest",
            "tau1_rest",
            "tau2",
            "ar",
            "bs",
            "Is",
            "cr",
        ],
        additional_key_settings={},
        final_time=final_time,
        use_sx=True,
        n_threads=6,
        ode_solver=OdeSolver.RK4(n_integration_steps=10),
    )


def main():
    stim_time = np.linspace(0, 1, 11)[:-1].tolist()
    pulse_intensity = np.random.randint(30, 130, 10).tolist()
    final_time = 2

    pickle_file_name = simulate_data(final_time=final_time, stim_time=stim_time, pulse_intensity=pulse_intensity)
    ocp = prepare_ocp(pickle_file_name, final_time=final_time, stim_time=stim_time, pulse_intensity=pulse_intensity)

    identified_parameters = ocp.force_model_identification()
    force_ocp = ocp.force_identification_result.stepwise_states(to_merge=SolutionMerge.NODES)["F"][0]
    time_ocp = ocp.force_identification_result.stepwise_time(to_merge=SolutionMerge.NODES).T[0]
    print(identified_parameters)

    (
        pickle_time_data,
        pickle_stim_apparition_time,
        pickle_muscle_data,
        pickle_discontinuity_phase_list,
    ) = full_data_extraction([pickle_file_name])

    result_dict = {
        "a_rest": [identified_parameters["a_rest"], DingModelPulseIntensityFrequency().a_rest],
        "km_rest": [identified_parameters["km_rest"], DingModelPulseIntensityFrequency().km_rest],
        "tau1_rest": [identified_parameters["tau1_rest"], DingModelPulseIntensityFrequency().tau1_rest],
        "tau2": [identified_parameters["tau2"], DingModelPulseIntensityFrequency().tau2],
        "ar": [identified_parameters["ar"], DingModelPulseIntensityFrequency().ar],
        "bs": [identified_parameters["bs"], DingModelPulseIntensityFrequency().bs],
        "Is": [identified_parameters["Is"], DingModelPulseIntensityFrequency().Is],
        "cr": [identified_parameters["cr"], DingModelPulseIntensityFrequency().cr],
    }

    # Plotting the identification result
    plt.title("Force state result")
    plt.plot(pickle_time_data, pickle_muscle_data, color="blue", label="simulated")
    plt.plot(time_ocp, force_ocp, color="red", label="identified")
    plt.xlabel("time (s)")
    plt.ylabel("force (N)")

    y_pos = 0.85
    for key, value in result_dict.items():
        plt.annotate(f"{key} : ", xy=(0.7, y_pos), xycoords="axes fraction", color="black")
        plt.annotate(str(round(value[0], 5)), xy=(0.78, y_pos), xycoords="axes fraction", color="red")
        plt.annotate(str(round(value[1], 5)), xy=(0.85, y_pos), xycoords="axes fraction", color="blue")
        y_pos -= 0.05

    # Delete the temp file
    os.remove(pickle_file_name)

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
