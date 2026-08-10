"""
This class will transform the experiment data into a format that can be used for optimal control problems (OCP)."
It will save the data in a pickle file, which can be loaded later for OCP formulation and solving.
The class will handle the transformation of the data, including any necessary preprocessing steps, to ensure that it
is in the correct format for OCP.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from scipy.interpolate import interp1d

class DataToOCP:
    def __init__(self):
        self.data_dictionary = []
        self.ocp_data = []

    def open_files(self, file_path_list):
        for path in file_path_list:
            self.data_dictionary.append(pkl.load(open(path, "rb")))

    @staticmethod
    def force_to_ocp(force, time, n_shooting, final_time):
            """
            Interpolate force at shooting points using cubic spline.
            """
            shooting_times = np.linspace(0, final_time, n_shooting + 1)
            interp_func = interp1d(time, force, kind='cubic', bounds_error=False, fill_value='extrapolate')
            force_at_shooting = interp_func(shooting_times)
            return force_at_shooting[np.newaxis, :]

    def get_data_for_ocp(self, plot=False):
        if len(self.data_dictionary) == 0:
            raise ValueError("No data loaded. Please load data using the open_files method before calling get_data_for_ocp.")

        final_time = []
        frequency = []
        n_shooting = []
        force = []
        weight_cost = []
        pulse_width = []
        stim_time = []
        force_decayed = []

        for data in self.data_dictionary:
            for i in range(len(data["time"])):
                # False when the force never came back near zero, i.e. a train dominated by drift. 
                force_decayed.append(bool(data.get("force_decayed", [True] * len(data["time"]))[i]))
                phase_time = data["time"][i][-1] - data["time"][i][0]
                denominator = 1/data["frequency"]
                shooting_nodes = int(phase_time / denominator)
                time_to_freq = shooting_nodes * denominator
                final_time.append(time_to_freq)
                frequency.append(data["frequency"])
                n_shooting.append(shooting_nodes)
                pulse_width.append(data["pulse_width"][i]/1e6)
                last_stim_time = data["stim_time"][i][-1] - data["stim_time"][i][0]
                nb_stim = int(last_stim_time / denominator)
                round_last_stim_time = nb_stim * denominator
                stim_time.append(list(np.linspace(0, round_last_stim_time, nb_stim + 1)))
                time_in_phase = np.array(data["time"][i]) - data["time"][i][0]
                force.append(self.force_to_ocp(data["force"][i], time_in_phase, shooting_nodes, time_to_freq))

                if plot:
                    plt.plot(time_in_phase, data["force"][i], label="Force exp")
                    plt.plot(np.linspace(0, time_to_freq, shooting_nodes + 1), force[-1][0], label="Force ocp")
                    plt.legend()
                    plt.show()

        max_n_shooting = max(n_shooting)
        for j in range(len(n_shooting)):
            weight_cost.append(max_n_shooting / n_shooting[j])

        ocp_dict = {
        "final_time": final_time,
        "frequency": frequency,
        "n_shooting": n_shooting,
        "force": force,
        "weight_cost": weight_cost,
        "pulse_width": pulse_width,
        "stim_time": stim_time,
        "force_decayed": force_decayed,
        }

        return ocp_dict


if __name__ == "__main__":
    force_folder = Path(__file__).resolve().parent.parent.parent / "data" / "force" / "P01"
    ocp_data = DataToOCP()
    ocp_data.open_files(file_path_list=sorted(str(path) for path in force_folder.glob("*force*.pkl")))
    ocp_values = ocp_data.get_data_for_ocp()
