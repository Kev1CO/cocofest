import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import heapq
import pickle

from pyomeca import Analogs, Markers


class C3DToQ:
    def __init__(self, c3d_path: str | list[str]):
        self.markers_index = {"should_r": 0, "delt_r":1, "elbow_r":2, "rwra": 3, "rwrb": 4}

        if isinstance(c3d_path, str):
            self.c3d_path = [c3d_path]
        elif isinstance(c3d_path, list):
            self.c3d_path = c3d_path
        else:
            raise ValueError("c3d_path must be a string or a list of strings.")

        self.markers_name: list[str] = []
        self.data: np.ndarray
        self.data_dict: dict[str, np.ndarray] = {}
        self.wirst_position: np.ndarray
        self.forearm_position: np.ndarray
        self.humerus_position: np.ndarray
        self.forearm_position_proj: np.ndarray
        self.humerus_position_proj: np.ndarray
        self.elbow_angle_rad: np.ndarray
        self.elbow_angle_deg: np.ndarray
        self.Q_rad: np.ndarray
        self.Q_deg: np.ndarray

    def load_c3d(self):
        """
        Load C3D file(s) and extract marker data.
        """

        for path in self.c3d_path:
            c3d = Markers.from_c3d(path)
            self.markers_name = list(c3d.channel.data)
            self.data = c3d.data
            lst_index = self.set_index(self.markers_name)
            # self.data = self.reindex_2d_list(self.data, lst_index)
            self.data_dict = {}
            for i, marker in enumerate(self.markers_index.keys()):
                self.data_dict[marker] = self.data[:3, i, :]

    @staticmethod
    def get_index(name, lst):
        """
        This function returns the index of the name in the list.
        Parameters
        ----------
        name: str | int | float
        lst: list

        Returns
        -------
        The index of the name in the list
        """
        indice = 0
        for i in range(len(lst)):
            if name == lst[i]:
                indice = i
        return indice

    def set_index(self, lst_markers):
        """
        This function create a list of new index based on the default index dict
        Parameters
        ----------
        raw_data: Analogs
            The raw data from the c3d file

        Returns
        -------
        A list of new index
        """
        lst_index = []
        for i in range(len(self.markers_index.keys())):
            name = list(self.markers_index.keys())[i]
            indice = self.get_index(name, lst_markers)
            lst_index.append(indice)
        return lst_index

    @staticmethod
    def reindex_2d_list(data, new_indices):  # TODO : reindex_3D_list
        """
        This function reindex a 2D list based on the new index list
        Parameters
        ----------
        data: array
            The data to reindex
        new_indices: list
            Contains new index

        Returns
        -------
        Reindex data array
        """
        # Ensure the new_indices list is not out of bounds
        if max(new_indices) >= len(data) or min(new_indices) < 0:
            raise ValueError("Invalid new_indices list. Out of bounds.")

        # Create a new 2D list with re-ordered elements
        new_data = [[data[i][j] for j in range(len(data[i]))] for i in new_indices]

        return new_data

    def get_wrist_position(self):
        self.wirst_position = np.zeros_like(self.data_dict["rwra"])
        for i in range(len(self.data_dict["rwra"][0])):
            self.wirst_position[0, i] = (self.data_dict["rwrb"][0, i] + self.data_dict["rwra"][0, i]) / 2
            self.wirst_position[1, i] = (self.data_dict["rwrb"][1, i] + self.data_dict["rwra"][1, i]) / 2
            self.wirst_position[2, i] = (self.data_dict["rwrb"][2, i] + self.data_dict["rwra"][2, i]) / 2

        return self.wirst_position

    @staticmethod
    def get_segment_vector(start, end):

        return np.array(end) - np.array(start)

    @staticmethod
    def projection_vectors(u, v):
        u_proj = np.zeros_like(u)
        v_proj = np.zeros_like(v)
        e1 = np.zeros_like(u)
        e2 = np.zeros_like(v)
        v_proj_sur_e1 = np.zeros_like(v)

        for i in range(len(u[0])):
            e1[:, i] = u[:, i] / np.linalg.norm(u[:, i])

            v_proj_sur_e1[:, i] = np.dot(v[:, i], e1[:, i]) * e1[:, i]

            e2[:, i] = v[:, i] - v_proj_sur_e1[:, i]
            e2[:, i] /= np.linalg.norm(e2[:, i])

            u_proj[0, i] = np.dot(u[:, i], e1[:, i])
            u_proj[1, i] = np.dot(u[:, i], e2[:, i])
            v_proj[0, i] = np.dot(v[:, i], e1[:, i])
            v_proj[1, i] = np.dot(v[:, i], e2[:, i])

        return u_proj, v_proj

    @staticmethod
    def get_angle(u, v):
        """
        Calculate the angle between two vectors in radians.
        Parameters
        ----------
        u: array
            The first vector
        v: array
            The second vector

        Returns
        -------
        Angle in radians
        """
        angle = np.zeros_like(u[0])
        for i in range(len(u[0])):
            norm_u = np.linalg.norm(u[:, i])
            norm_v = np.linalg.norm(v[:, i])
            dot_product = np.dot(u[:, i], v[:, i])
            cos_theta = dot_product / (norm_u * norm_v)
            angle[i] = np.arccos(cos_theta)
        return angle

    @staticmethod
    def save_in_pkl(data, pkl_path):
        if isinstance(pkl_path, str):
            pkl_path = [pkl_path]
        elif isinstance(pkl_path, list):
            pkl_path = pkl_path
        else:
            raise ValueError("pkl_path must be a string or a list of strings.")

        for path in pkl_path:
            with open(path, "wb") as file:
                pickle.dump(data, file)

    def _get_q(self):
        self.load_c3d()
        self.get_wrist_position()
        self.forearm_position = self.get_segment_vector(start=self.data_dict["elbow_r"], end=self.wirst_position)
        self.humerus_position = self.get_segment_vector(start=self.data_dict["elbow_r"], end=self.data_dict["should_r"])
        self.forearm_position_proj, self.humerus_position_proj = self.projection_vectors(self.forearm_position,
                                                                                         self.humerus_position)
        self.elbow_angle_rad = self.get_angle(self.forearm_position_proj[:2], self.humerus_position_proj[:2])
        self.elbow_angle_deg = np.rad2deg(self.elbow_angle_rad)
        self.Q_rad = np.pi - self.elbow_angle_rad
        self.Q_deg = np.rad2deg(self.Q_rad)

        return self.Q_rad

    def get_q_rad(self):
        Q_rad = self._get_q()
        return Q_rad

    def get_q_deg(self):
        Q_rad = self._get_q()
        return np.rad2deg(Q_rad)


if __name__ == "__main__":
    c3d_path = "C:\\Users\\flori_4ro0b8\\Documents\\Stage_S2M\\c3d_file\\essais_mvt_16.04.25\\Gap_interpol\\Kevin_mouv_50hz_250-300-350-400-450us_20mA_1s_1sr.c3d"
    c3d_to_q = C3DToQ(c3d_path)
    angle = c3d_to_q.Q_deg
    plt.plot(angle)
    plt.show()
    # "C:\\Users\\flori_4ro0b8\\Documents\\Stage_S2M\\c3d_file\\essais_mvt_16.04.25\\Florine_mouv_50hz_250-300-350-400-450us_15mA_1s_1sr.c3d"