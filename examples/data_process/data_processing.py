import pickle

import matplotlib.pyplot as plt
from examples.data_process.c3d_to_q import C3dToQ
import pandas as pd
import ast
from pathlib import Path


def auto_process_motion(p_n_list, save=True, plot=False):
    """
    This function processes the motion c3d files for the specified participants.
    You can choose to save the processed data in pickle files and/or plot the results.
    Parameters
    ----------
    p_n_list: list
        List of participant numbers to process.
    save: bool
        If True, saves the processed data in pickle files.
    plot: bool
        If True, plots the processed data.
    """
    p_data = pd.read_csv("/home/mickaelbegon/Documents/Stage_Florine/Data/data_participants.csv", sep=";")
    for p_n in p_n_list:
        freq_str = p_data.iloc[p_n - 1]["freq_motion"]
        freq_list = ast.literal_eval(freq_str)
        seeds_str = p_data.iloc[p_n - 1]["seed_motion"]
        seeds_list = ast.literal_eval(seeds_str)

        for i in range(len(freq_list)):
            p_nb = str(p_n) if len(str(p_n)) == 2 else "0" + str(p_n)
            data_path = "P" + p_nb + "/p" + p_nb + "_motion_" + str(freq_list[i]) + "Hz_" + str(seeds_list[i]) + ".c3d"
            print(data_path)
            whole_data_path = f"/home/mickaelbegon/Documents/Stage_Florine/Data/{data_path}"
            converter = C3dToQ(c3d_path=whole_data_path)
            converter.frequency_stimulation = freq_list[i]
            dict = converter.get_sliced_time_Q_rad()
            if save:
                current_file_dir = Path(__file__).parent
                pickle_path = f"{current_file_dir}/pkl_files/p{p_nb}_motion_{freq_list[i]}Hz_{seeds_list[i]}.pkl"
                converter.save_in_pkl(dict, pickle_path)
            if plot:
                time = dict["time"]
                stim_time = dict["stim_time"]
                q = dict["q"]

                for j in range(len(time)):
                    if j==0:
                        plt.plot(time[j], q[j], label="Q (rad)", color="blue")
                        plt.scatter(stim_time[j], [0] * len(stim_time[j]), color="green", label="Stimulations")
                    else:
                        plt.plot(time[j], q[j], color="blue")
                        plt.scatter(stim_time[j], [0] * len(stim_time[j]), color="green")
                plt.title(f"Participant {p_n} - Freq {freq_list[i]}Hz - Seed {seeds_list[i]}")
                plt.xlabel("Time (s)")
                plt.ylabel("Elbow angle (rad)")
                plt.legend()
                plt.show()

def check_data_motion(p_n_list):
    """
    This function plots the processed motion data after being automatically ran with the auto_process_motion function.
    Parameters
    ----------
    p_n_list: list
        List of participant numbers to check.
    """
    p_data = pd.read_csv("/home/mickaelbegon/Documents/Stage_Florine/Data/data_participants.csv", sep=";")
    for p_n in p_n_list:
        freq_str = p_data.iloc[p_n - 1]["freq_motion"]
        freq_list = ast.literal_eval(freq_str)
        seeds_str = p_data.iloc[p_n - 1]["seed_motion"]
        seeds_list = ast.literal_eval(seeds_str)

        for i in range(len(freq_list)):
            p_nb = str(p_n) if len(str(p_n)) == 2 else "0" + str(p_n)
            current_file_dir = Path(__file__).parent
            pickle_path = f"{current_file_dir}/pkl_files/p{p_nb}_motion_{freq_list[i]}Hz_{seeds_list[i]}.pkl"
            with open(pickle_path, "rb") as f:
                data = pickle.load(f)
            for j in range(len(data["time"])):
                if j==0:
                    plt.plot(data["time"][j], data["q"][j], label="Force BIC Long", color="blue")
                    plt.scatter(data["stim_time"][j], [0] * len(data["stim_time"][j]), color="green", label="Stimulations")
                else:
                    plt.plot(data["time"][j], data["q"][j], color="blue")
                    plt.scatter(data["stim_time"][j], [0] * len(data["stim_time"][j]), color="green")
            plt.title(f"Participant {p_n} - Freq {freq_list[i]}Hz - Seed {seeds_list[i]}")
            plt.xlabel("Time (s)")
            plt.ylabel("Q (rad)")
            plt.legend()
            plt.show()


if __name__ == "__main__":
    # auto_process_motion([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], save=True, plot=False)
    check_data_motion([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])