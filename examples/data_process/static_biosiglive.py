import pandas as pd

from biosiglive import MskFunctions
from pathlib import Path
import matplotlib.pyplot as plt
import pickle
import ast

import numpy as np
import biorbd

from examples.data_process.c3d_to_force import C3dToForce

def load_model(model_path):
    return biorbd.Model(model_path)

def load_data(data_path, freq_stim):
    converter = C3dToForce(
        c3d_path=data_path,
        calibration_matrix_path="/home/mickaelbegon/Documents/Stage_Florine/matrix.txt",
        frequency_stimulation=freq_stim,
    )
    converter.get_data_at_handle()
    data_dict = converter.handle_dictionary

    return data_dict

def save_in_pkl(data, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(data, f)

def get_muscles_forces(p_n, data_path, biomodel_path, freq_stim, save=True):
    """
    This function uses the MskFunctions class from biosiglive to compute the muscle forces.
    Parameters
    ----------
    p_n: int
        Participant number
    data_path: str
    biomodel_path: str
    freq_stim: int
        Frequency of the stimulation in Hz
    save: bool
        True if you want to save the results in a pickle file, False otherwise

    Returns
    -------
    A dictionary containing the time, stim_time, force_biclong, force_bicshort, tau_list, muscles activation (act),
    and residual torque (res).
    """
    current_file_dir = Path(__file__).parent
    whole_biomodel_path = "/home/mickaelbegon/Documents/Stage_Florine/modelScaled/" + biomodel_path
    model = load_model(whole_biomodel_path)
    whole_data_path = "/home/mickaelbegon/Documents/Stage_Florine/Data/" + data_path
    data_dict = load_data(whole_data_path, freq_stim)
    time = data_dict["time"]
    stim_time = data_dict["stim_time"]

    msk_functions = MskFunctions(model=whole_biomodel_path)

    # Load participants data base
    p_data = pd.read_csv("/home/mickaelbegon/Documents/Stage_Florine/Data/data_participants.csv", sep=";")
    d_handle_rotation = p_data.iloc[p_n - 1]["handle_elbow_dist"] / 100 # Convert cm to m

    tau_list = []

    # Get the maximum isometric forces for the muscles from the biomodel
    max_forces = [model.muscle(0).characteristics().forceIsoMax(), model.muscle(1).characteristics().forceIsoMax()]
    force_biclong = []
    force_bicshort = []

    for j in range(len(data_dict["time"])):
        for i in range(len(data_dict["time"][j])):
            Q = np.array([[90.]])
            Qdot = np.array([[0.]])
            norm = np.sqrt(data_dict["x"][j][i]**2 + data_dict["y"][j][i]**2 + data_dict["z"][j][i]**2)
            Tau = np.array([[d_handle_rotation * norm]])
            tau_list.append(Tau[0][0])
            act, res = msk_functions.compute_static_optimization(Q, Qdot, Tau, use_residual_torque=True)

    for i in range(len(res[0])):
        force_bicshort.append(act[1][i] * max_forces[1])
        force_biclong.append(act[0][i] * max_forces[0])

    force_biclong = slicing(force_biclong, time)
    force_bicshort = slicing(force_bicshort, time)

    dict = {
        "time": time,
        "stim_time": stim_time,
        "force_biclong": force_biclong,
        "force_bicshort": force_bicshort,
        "tau_list": tau_list,
        "act": act,
        "res": res
    }
    if save:
        save_path = data_path[4:-3] + "pkl"
        saving_path = f"{current_file_dir}/pkl_files/{save_path}"
        save_in_pkl(dict, saving_path)

    return dict

def slicing(data, time):
    """
    This function slices the data according to the time intervals provided. You will need this function to slice again
    data after using MskFunctions static optimisation
    Parameters
    ----------
    data: list
    time: list[np.array]
        The time data must already be sliced

    Returns
    -------
    The sliced data
    """
    first = 0
    sliced_data = []
    for i in range(len(time)):
        last = first + len(time[i])
        sliced_data.append(data[first:last])
        first = last
    return sliced_data

def auto_process(p_n_list, save=True, plot=False):
    """
    This function processes the data for each participant in the list. You can run it automatically choosing plot=False
    and save=True, then check the results using the check_data function.
    Parameters
    ----------
    p_n_list: list[int]
        List of all participants numbers
    save: bool
        True if you want to save the results in a pickle file, False otherwise
    plot: bool
        True if you want to plot the results, False otherwise
    """
    p_data = pd.read_csv("/home/mickaelbegon/Documents/Stage_Florine/Data/data_participants.csv", sep=";")
    for p_n in p_n_list:
        freq_str = p_data.iloc[p_n - 1]["freq_force"]
        freq_list = ast.literal_eval(freq_str)
        seeds_str = p_data.iloc[p_n - 1]["seed_force"]
        seeds_list = ast.literal_eval(seeds_str)

        for i in range(len(freq_list)):
            p_nb = str(p_n) if len(str(p_n)) == 2 else "0" + str(p_n)
            data_path = "P" + p_nb + "/p" + p_nb + "_force_" + str(freq_list[i]) + "Hz_" + str(seeds_list[i]) + ".c3d"
            print(data_path)
            model_path = "p" + p_nb + "_scaling_scaled.bioMod"
            dict = get_muscles_forces(p_n=p_n, data_path=data_path, biomodel_path=model_path, freq_stim=freq_list[i], save=save)
            if plot:
                time = dict["time"]
                stim_time = dict["stim_time"]
                force_biclong = dict["force_biclong"]
                force_bicshort = dict["force_bicshort"]

                for j in range(len(time)):
                    if j==0:
                        plt.plot(time[j], force_biclong[j], label="Force BIC Long", color="blue")
                        plt.plot(time[j], force_bicshort[j], label="Force BIC Short", color="orange")
                        plt.scatter(stim_time[j], [0] * len(stim_time[j]), label="Stimulations", color="green")
                    else:
                        plt.plot(time[j], force_biclong[j], color="blue")
                        plt.plot(time[j], force_bicshort[j], color="orange")
                        plt.scatter(stim_time[j], [0] * len(stim_time[j]), color="green")
                plt.title(f"Participant {p_n} - Freq {freq_list[i]}Hz - Seed {seeds_list[i]}")
                plt.xlabel("Time (s)")
                plt.ylabel("Force (N)")
                plt.legend()
                plt.show()

def check_data(p_n_list):
    """
    This function plots the data for each participant in the list. You can use it to check the results from the
    auto_process function.
    Parameters
    ----------
    p_n_list: list[int
        List of all participants numbers
    """
    p_data = pd.read_csv("/home/mickaelbegon/Documents/Stage_Florine/Data/data_participants.csv", sep=";")
    for p_n in p_n_list:
        freq_str = p_data.iloc[p_n - 1]["freq_force"]
        freq_list = ast.literal_eval(freq_str)
        seeds_str = p_data.iloc[p_n - 1]["seed_force"]
        seeds_list = ast.literal_eval(seeds_str)

        for i in range(len(freq_list)):
            p_nb = str(p_n) if len(str(p_n)) == 2 else "0" + str(p_n)
            current_file_dir = Path(__file__).parent
            pickle_path = f"{current_file_dir}/pkl_files/p{p_nb}_force_{freq_list[i]}Hz_{seeds_list[i]}.pkl"
            with open(pickle_path, "rb") as f:
                data = pickle.load(f)
            for j in range(len(data["time"])):
                if j==0:
                    plt.plot(data["time"][j], data["force_biclong"][j], label="Force BIC Long", color="blue")
                    plt.plot(data["time"][j], data["force_bicshort"][j], label="Force BIC Short", color="orange")
                    plt.scatter(data["stim_time"][j], [0] * len(data["stim_time"][j]), label="Stimulations", color="green")
                else:
                    plt.plot(data["time"][j], data["force_biclong"][j], color="blue")
                    plt.plot(data["time"][j], data["force_bicshort"][j], color="orange")
                    plt.scatter(data["stim_time"][j], [0] * len(data["stim_time"][j]), color="green")
            plt.title(f"Participant {p_n} - Freq {freq_list[i]}Hz - Seed {seeds_list[i]}")
            plt.xlabel("Time (s)")
            plt.ylabel("Force (N)")
            plt.legend()
            plt.show()


if __name__ == "__main__":
    #auto_process([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], save=True, plot=False)
    check_data([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])