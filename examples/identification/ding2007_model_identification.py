"""
NOT CONVERGING TO THE EXPECTED RESULT - SEE ISSUE
--
This example demonstrates the way of identifying the Ding 2007 model parameter using noisy simulated data.
First we integrate the model with a given parameter set. Then we add noise to the previously calculated force output.
Finally, we use the noisy data to identify the model parameters.
"""

import pickle
import os
import matplotlib.pyplot as plt
import numpy as np

from bioptim import SolutionMerge, OdeSolver

from cocofest import (
    DingModelPulseWidthFrequency,
    DingModelPulseWidthFrequencyForceParameterIdentification,
    IvpFes,
    ModelMaker,
)
from cocofest.identification.identification_method import full_data_extraction


def simulate_data(final_time=2, stim_time=None, pulse_width=None):
    ivp_model = ModelMaker.create_model("ding2007", stim_time=stim_time, sum_stim_truncation=10)
    fes_parameters = {"model": ivp_model, "stim_time": stim_time, "pulse_width": pulse_width}
    ivp_parameters = {"final_time": final_time, "use_sx": True}

    # --- Creating the simulated data to identify on --- #
    # Building the Initial Value Problem
    ivp = IvpFes(fes_parameters, ivp_parameters)

    # Integrating the solution
    result, time = ivp.integrate()

    # Adding noise to the force
    noise = np.random.normal(0, 0.5, len(result["F"][0]))
    force = result["F"][0] + noise

    # Saving the data in a pickle file
    dictionary = {"time": time, "force": force, "stim_time": stim_time, "pulse_width": pulse_width}

    pickle_file_name = "../data/temp_identification_simulation.pkl"
    with open(pickle_file_name, "wb") as file:
        pickle.dump(dictionary, file)

    return pickle_file_name


def prepare_ocp():
    # --- Setting simulation parameters --- #
    stim_time = np.linspace(0, 1, 11)[:-1].tolist()
    pulse_width = np.random.uniform(0.0002, 0.0006, 10).tolist()
    final_time = 2

    pickle_file_name = simulate_data(final_time=final_time, stim_time=stim_time, pulse_width=pulse_width)

    # --- Identifying the model parameters --- #
    ocp_model = DingModelPulseWidthFrequency(stim_time=stim_time, sum_stim_truncation=10)
    return (
        DingModelPulseWidthFrequencyForceParameterIdentification(
            model=ocp_model,
            data_path=[pickle_file_name],
            identification_method="full",
            double_step_identification=False,
            key_parameter_to_identify=["tau1_rest", "tau2", "km_rest", "a_scale", "pd0", "pdt"],
            additional_key_settings={},
            final_time=final_time,
            use_sx=True,
            n_threads=6,
            ode_solver=OdeSolver.RK4(n_integration_steps=10),
        ),
        pickle_file_name,
    )


def main():
    ocp, pickle_file_name = prepare_ocp()
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
        "tau1_rest": [identified_parameters["tau1_rest"], DingModelPulseWidthFrequency().tau1_rest],
        "tau2": [identified_parameters["tau2"], DingModelPulseWidthFrequency().tau2],
        "km_rest": [identified_parameters["km_rest"], DingModelPulseWidthFrequency().km_rest],
        "a_scale": [identified_parameters["a_scale"], DingModelPulseWidthFrequency().a_scale],
        "pd0": [identified_parameters["pd0"], DingModelPulseWidthFrequency().pd0],
        "pdt": [identified_parameters["pdt"], DingModelPulseWidthFrequency().pdt],
    }

    # Plotting the identification result
    plt.title("Force state result")
    plt.plot(pickle_time_data, pickle_muscle_data, "-.", color="blue", label="simulated")
    plt.plot(time_ocp, force_ocp, color="red", label="identified")

    plt.xlabel("time (s)")
    plt.ylabel("force (N)")

    y_pos = 0.85
    for key, value in result_dict.items():
        plt.annotate(f"{key} : ", xy=(0.7, y_pos), xycoords="axes fraction", color="black")
        plt.annotate(str(round(value[0], 5)), xy=(0.78, y_pos), xycoords="axes fraction", color="red")
        plt.annotate(str(round(value[1], 5)), xy=(0.85, y_pos), xycoords="axes fraction", color="blue")
        y_pos -= 0.05

    # --- Delete the temp file ---#
    os.remove(f"../data/temp_identification_simulation.pkl")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
