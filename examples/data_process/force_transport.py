import numpy as np
import pickle


def alpha_ins_fun(l_biceps, l_ins, alpha_elbow):
    delta = l_ins**2 - (l_biceps**2 + l_ins**2 - l_biceps**2 / np.sin(alpha_elbow) ** 2) / np.sin(alpha_elbow) ** 2
    X1 = 0
    X2 = 0
    alpha_ins_1 = alpha_elbow
    alpha_ins_2 = alpha_elbow
    alpha_ins_3 = alpha_elbow
    alpha_ins_4 = alpha_elbow

    if delta >= 0:
        X1 = np.sin(alpha_elbow) ** 2 * (2 * l_biceps * l_ins + np.sqrt(delta)) / (2 * l_biceps**2)
        X2 = np.sin(alpha_elbow) ** 2 * (2 * l_biceps * l_ins - np.sqrt(delta)) / (2 * l_biceps**2)

        if -1 <= X1 <= 1:
            alpha_ins_1 = np.pi - np.arccos(X1)
            alpha_ins_2 = np.pi + np.arccos(X1)

        elif -1 <= X2 <= 1:
            alpha_ins_3 = np.pi - np.arccos(X2)
            alpha_ins_4 = np.pi + np.arccos(X2)

        else:
            raise ValueError("No solution found")

    else:
        raise ValueError("Equation cannot be solved")

    return [alpha_ins_1, alpha_ins_2, alpha_ins_3, alpha_ins_4]


def force_biceps_fun(alpha_ins: float, force: list):
    rotation_matrix = np.array(
        [[-np.cos(alpha_ins), 0, np.sin(alpha_ins)], [0, 1, 0], [-np.sin(alpha_ins), 0, -np.cos(alpha_ins)]]
    )
    biceps_force = np.dot(rotation_matrix, force)

    return biceps_force


def norm_force_fun(force):
    force_x = force[0]
    force_y = force[1]
    force_z = force[2]

    norm_force_list = []

    for i in range(len(force[0])):
        norm_force = np.sqrt(force_x**2 + force_y**2 + force_z**2)
        norm_force_list.append(norm_force)

    return norm_force_list


def get_data_from_bio():
    pass


def read_pkl_to_force_vector(pickle_path):
    if isinstance(pickle_path, str):
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)
            force = [data["x"], data["y"], data["z"]]
    else:
        raise ValueError("pickle_path must be a string")

    return force
