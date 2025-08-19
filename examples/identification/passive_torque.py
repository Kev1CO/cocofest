import numpy as np
import matplotlib.pyplot as plt
from math import exp

from examples import C3dToQ


def passive_torque(q, qdot):
    k1 = 0.24395358  # profil bon
    k2 = 5.103129
    k3 = 1.194639
    k4 = 5.41585281
    kc1 = 1.05713656
    kc2 = 0.19654403
    theta_max = 2.27074652
    theta_min = 0.49997778
    k1 = 0.10000623
    k2 = 5.00041991
    k3 = 1.00007728
    k4 = 6.99990535
    kc1 = 0.17628383
    kc2 = 1.42997773
    theta_max = 2.40089727
    theta_min = 0.43825389

    def sigmoide(x):
        return 1 / (1 + exp(-x))

    c = (sigmoide((q - theta_max) / kc2) + sigmoide(-(q - theta_min) / kc1))
    basse = k1 * exp(-k2 * (q - theta_min)) * sigmoide(-(q - theta_min))
    haute = k3 * exp(k4 * (q - theta_max)) * sigmoide(q - theta_max)
    passive_torque = basse - haute - c * qdot

    return passive_torque

q_list = np.linspace(0, 2.5, 50)
qdot_list = np.zeros_like(q_list)

def compute_passive_torque_map(q_list, qdot_list):
    tau_matrix = np.zeros((len(q_list), len(qdot_list)))
    for i, q in enumerate(q_list):
        for j, qdot in enumerate(qdot_list):
            tau_matrix[i, j] = passive_torque(q, qdot)
    return tau_matrix

def graphe(q_list, qdot_list, tau_matrix):
    q, qdot = np.meshgrid(q_list, qdot_list)

    fig = plt.figure()
    ax = fig.add_subplot()#110, projection='3d')

    # Surface plot
    # surface = ax.plot_surface(q, qdot, tau_matrix, cmap='viridis')
    # ax.scatter(0.4, 0, passive_torque(0.4, 0), color='red', s=100, label='Initial point')
    # ax.scatter(2.5, 0, passive_torque(2.5, 0), color='blue', s=100, label='Maximal point')
    plt.plot(q_list, tau_matrix[:, 0], label='Torque Passif', color='blue')

    # Labels
    ax.set_xlabel('q')
    #ax.set_ylabel('qdot')
    ax.set_ylabel('Torque Passif')
    plt.title("Torque Passif en fonction de q avec qdot=0")

    # Barre de couleur
    # fig.colorbar(surface, shrink=0.5, aspect=5)

    plt.show()

def simulate_from_data_exp():
    converter = C3dToQ("/home/mickaelbegon/Documents/Stage_Florine/Data/P05/p05_motion_50Hz_62.c3d")
    converter.frequency_stimulation = 50
    data_dict = converter.get_sliced_time_Q_rad()
    q = data_dict["q"][:1][0]
    qdot = np.diff(q)
    tau_list = []
    for i in range(len(qdot)):
        tau_list.append(passive_torque(q[i], qdot[i]))

    q_list = np.linspace(0, 2.5, len(tau_list))
    plt.plot(q_list, tau_list, label="Torque Passif")
    plt.xlabel("q (rad)")
    plt.ylabel("Torque Passif (Nm)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    tau_matrix = compute_passive_torque_map(q_list, qdot_list)
    graphe(q_list, qdot_list, tau_matrix)
    # simulate_from_data_exp()