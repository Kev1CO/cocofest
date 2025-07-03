from biosiglive import MskFunctions
from pathlib import Path
import matplotlib.pyplot as plt
import pickle

import numpy as np
import biorbd

from examples.data_process.c3d_to_force import C3dToForce


def one_iter(position):
    current_file_dir = Path(__file__).parent

    msk_functions = MskFunctions(model=f"{current_file_dir}/../model_msk/arm26_allbiceps_1dof.bioMod")

    model = biorbd.Model(f"{current_file_dir}/../model_msk/arm26_allbiceps_1dof.bioMod")

    converter = C3dToForce(
        c3d_path="/home/mickaelbegon/Documents/Stage_Florine/essai1_florine_50Hz_400us_15mA_TR1s.c3d",
        calibration_matrix_path="/home/mickaelbegon/Documents/Stage_Florine/matrix.txt",
        saving_pickle_path=f"{current_file_dir}/essai1_florine_force_biceps.pkl",
        frequency_acquisition=10000,
        frequency_stimulation=50,
        rest_time=1,
        model_path=f"{current_file_dir}/../model_msk/simplified_UL_Seth.bioMod",
        elbow_angle=90,
        muscle_name="BIC_long",
        dof_name="r_ulna_radius_hand_r_elbow_flex_RotX",
        )

    converter.get_data_at_handle()
    data_dict = converter.dictionary
    nqdot = model.nbQdot()

    activation = []
    residual = []


    Q = np.array([[90.]])
    Qdot = np.array([[0.]])
    Tau = np.array([[0.23*data_dict["x"][5][position]]])

    act, res = msk_functions.compute_static_optimization(Q, Qdot, Tau, use_residual_torque=True)

    return act, res, Tau[0]

def n_iter(n):
    current_file_dir = Path(__file__).parent

    msk_functions = MskFunctions(model=f"{current_file_dir}/../model_msk/arm26_allbiceps_1dof.bioMod")

    model = biorbd.Model(f"{current_file_dir}/../model_msk/arm26_allbiceps_1dof.bioMod")

    converter = C3dToForce(
        c3d_path="/home/mickaelbegon/Documents/Stage_Florine/essai1_florine_50Hz_400us_15mA_TR1s.c3d",
        calibration_matrix_path="/home/mickaelbegon/Documents/Stage_Florine/matrix.txt",
        saving_pickle_path=f"{current_file_dir}/essai1_florine_force_biceps.pkl",
        frequency_acquisition=10000,
        frequency_stimulation=50,
        rest_time=1,
        model_path=f"{current_file_dir}/../model_msk/simplified_UL_Seth.bioMod",
        elbow_angle=90,
        muscle_name="BIC_long",
        dof_name="r_ulna_radius_hand_r_elbow_flex_RotX",
    )

    converter.get_data_at_handle()
    data_dict = converter.dictionary
    nqdot = model.nbQdot()
    tau_list = []

    for i in range(n):  # len(data_dict["time"])):
        Q = np.array([[90.]])
        Qdot = np.array([[0.]])
        Tau = np.array([[0.23 * data_dict["x"][5][i + 3000]]])
        tau_list.append(Tau[0][0])
        act, res = msk_functions.compute_static_optimization(Q, Qdot, Tau, use_residual_torque=True)

    return act, res, tau_list

def load_model(model_path):
    current_file_dir = Path(__file__).parent

    model = biorbd.Model(f"{current_file_dir}/../model_msk/{model_path}")

    return model

def load_data(data_path):
    current_file_dir = Path(__file__).parent
    converter = C3dToForce(
        c3d_path=data_path,
        calibration_matrix_path="/home/mickaelbegon/Documents/Stage_Florine/matrix.txt",
        saving_pickle_path=f"{current_file_dir}/essai1_florine_force_biceps.pkl",
        frequency_acquisition=10000,
        frequency_stimulation=50,
        rest_time=1,
        model_path=f"{current_file_dir}/../model_msk/simplified_UL_Seth.bioMod",
        elbow_angle=90,
        muscle_name="BIC_long",
        dof_name="r_ulna_radius_hand_r_elbow_flex_RotX",
    )
    converter.get_data_at_handle()
    data_dict = converter.dictionary

    return data_dict

def get_muscles_forces_from_inv_dynamics(data_path, model_path):
    with open(data_path, "rb") as f: #"/home/mickaelbegon/Documents/Stage_Florine/tau_example.pkl"
        data = pickle.load(f)
    current_file_dir = Path(__file__).parent
    msk_functions = MskFunctions(model=f"{current_file_dir}/../model_msk/{model_path}")

    data = np.array(data) - data[-1]
    tau_list = []

    for i in range(len(data)):
        Q = np.array([[90.]])
        Qdot = np.array([[0.]])
        Tau = np.array([[data[i]]])
        tau_list.append(Tau[0][0])
        act, res = msk_functions.compute_static_optimization(Q, Qdot, Tau, use_residual_torque=True)

    max_forces = [624.29999999999995, 435.56]
    force_biclong = []
    force_bicshort = []

    for i in range(len(res[0])):
        force_bicshort.append(act[1][i] * max_forces[1])
        force_biclong.append(act[0][i] * max_forces[0])

    return force_biclong, force_bicshort, tau_list, act

def get_muscles_forces(data_path, model_path):
    current_file_dir = Path(__file__).parent
    model = load_model(model_path)
    data_dict = load_data(data_path)

    msk_functions = MskFunctions(model=f"{current_file_dir}/../model_msk/{model_path}")

    tau_list = []

    for i in range(len(data_dict["time"][-1])):
        Q = np.array([[90.]])
        Qdot = np.array([[0.]])
        Tau = np.array([[0.23 * data_dict["x"][-1][i]]])
        tau_list.append(Tau[0][0])
        act, res = msk_functions.compute_static_optimization(Q, Qdot, Tau, use_residual_torque=True)

    max_forces = [624.29999999999995, 435.56]
    force_biclong = []
    force_bicshort = []

    for i in range(len(res[0])):
        force_bicshort.append(act[1][i] * max_forces[1])
        force_biclong.append(act[0][i] * max_forces[0])

    return force_biclong, force_bicshort, tau_list, act

def result(n):
    act_n, res_n, tau_n = n_iter(n)
    act1 = []
    act2 = []
    res = []
    tau = []
    #for i in range(n):
    #    a, r, t = one_iter(i*300 + 3000)
    #    act1.append(a[0][0])
    #    act2.append(a[1][0])
    #    res.append(r[0][0])
    #    tau.append(t[0])
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    #axs[1, 0].plot(act1, label="Activation 1 - 1 iter")
    #axs[1, 1].plot(act2, label="Activation 2 - 1 iter")
    #axs[0, 1].plot(res, label="Residual - 1 iter")
    #axs[0, 0].plot(tau, label="Tau 1 iter")
    axs[1, 0].plot(act_n[0], label="Activation 1 - n iter")
    axs[1, 1].plot(act_n[1], label="Activation 2 - n iter")
    axs[0, 1].plot(res_n[0], label="Residual n iter")
    axs[0, 0].plot(tau_n, label="Tau n iter")
    axs[0, 0].set_title("Tau")
    axs[0, 1].set_title("Residual")
    axs[1, 0].set_title("Activation 1")
    axs[1, 1].set_title("Activation 2")
    axs[0, 0].legend()
    axs[0, 1].legend()
    axs[1, 0].legend()
    axs[1, 1].legend()
    plt.show()

if __name__ == "__main__":
    force_biclong_inv_dy, force_bishort_inv_dy, tau_inv_dy, act_inv_dy = get_muscles_forces_from_inv_dynamics(
        data_path="/home/mickaelbegon/Documents/Stage_Florine/tau_p07.pkl",
        model_path="arm26_allbiceps_1dof.bioMod")


    force_biclong, force_bishort, tau, act = get_muscles_forces(data_path="/home/mickaelbegon/Documents/Stage_Florine/p07_force_50Hz_19.c3d",
                       model_path="arm26_allbiceps_1dof.bioMod")

    fig, axs = plt.subplots(2, 3, figsize=(10, 8))
    axs[0, 0].plot(tau, label="tau from static")
    axs[0, 0].plot(tau_inv_dy, label="tau from inverse dynamics")
    axs[0, 0].set_title("Tau")
    axs[0, 0].legend()
    axs[0, 1].plot(force_biclong, label="Force BIC long from static")
    axs[0, 1].plot(force_biclong_inv_dy, label="Force BIC long from inverse dynamics")
    axs[0, 1].set_title("Force BIC long")
    axs[0, 1].legend()
    axs[1, 1].plot(force_bishort, label="Force BIC short from static")
    axs[1, 1].plot(force_bishort_inv_dy, label="Force BIC short from inverse dynamics")
    axs[1, 1].set_title("Force BIC short")
    axs[1, 1].legend()
    axs[0, 2].plot(act[0], label="Activation BIC long from static")
    axs[0, 2].plot(act_inv_dy[0], label="Activation BIC long from inverse dynamics")
    axs[0, 2].set_title("Activation BIC long")
    axs[0, 2].legend()
    axs[1, 2].plot(act[1], label="Activation BIC short from static")
    axs[1, 2].plot(act_inv_dy[1], label="Activation BIC short from inverse dynamics")
    axs[1, 2].set_title("Activation BIC short")
    axs[1, 2].legend()

    plt.show()

    #print(force_biclong)