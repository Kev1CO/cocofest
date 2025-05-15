import numpy as np
import matplotlib.pyplot as plt
import heapq
import pickle

from pyomeca import Analogs, Markers


class C3dToQ:
    def __init__(self, c3d_path: str | list[str]):
        self.markers_index = {"should_r": 0, "delt_r": 1, "elbow_r": 2, "rwra": 3, "rwrb": 4}

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
        self.frequency_acquisition: int = 100  # Hz
        self.frequency_acquisition_stim: int = 10000  # Hz
        self.average_time_difference: float = 0.0
        self.frequency_stimulation: int = 50  # Hz
        self.data_stim: np.ndarray
        self.time_stim: np.ndarray
        self.time: np.ndarray
        self.stimulation_time: list[float] = []

    def load_c3d(self):
        """
        Load C3D file(s) and extract marker data.
        """

        for path in self.c3d_path:
            c3d = Markers.from_c3d(path)
            self.markers_name = list(c3d.channel.data)
            self.data = c3d.data
            self.time = c3d.time.data
            lst_index = self._set_index(self.markers_name)
            #self.data = self._reindex_3d_list(self.data, lst_index)
            self.data_dict = {}
            for i, marker in enumerate(self.markers_index.keys()):
                self.data_dict[marker] = self.data[:3, i, :]

    def load_analog(self):
        for path in self.c3d_path:
            analog = Analogs.from_c3d(path)
            index = self._get_index("Electric Current.Channel_5", analog.channel.data)
            self.data_stim = analog.data[index]
            self.time_stim = analog.time.data

    @staticmethod
    def _get_index(name, lst):
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

    def _set_index(self, lst_markers):
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
            indice = self._get_index(name, lst_markers)
            lst_index.append(indice)
        return lst_index

    @staticmethod
    def _reindex_3d_list(data, new_indices):
        """
        This function reindex a 3D list based on the new index list
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
        if max(new_indices) >= len(data[0]) or min(new_indices) < 0:
            raise ValueError("Invalid new_indices list. Out of bounds.")

        new_data = [[data[i][j] for j in new_indices] for i in range(len(data))]

        return new_data

    def _get_wrist_position(self):
        self.wirst_position = np.zeros_like(self.data_dict["rwra"])
        for i in range(len(self.data_dict["rwra"][0])):
            self.wirst_position[0, i] = (self.data_dict["rwrb"][0, i] + self.data_dict["rwra"][0, i]) / 2
            self.wirst_position[1, i] = (self.data_dict["rwrb"][1, i] + self.data_dict["rwra"][1, i]) / 2
            self.wirst_position[2, i] = (self.data_dict["rwrb"][2, i] + self.data_dict["rwra"][2, i]) / 2

        return self.wirst_position

    @staticmethod
    def _get_segment_vector(start, end):

        return np.array(end) - np.array(start)

    @staticmethod
    def _projection_vectors(u, v):
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
    def _get_angle(u, v):
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

    @staticmethod
    def _get_stimulation(time, stimulation_signal, average_time_difference=None):
        """
        This function detects the stimulation time and returns the time and index of the stimulation
        Parameters
        ----------
        time: array
            The time data
        stimulation_signal: array
            The stimulation signal
        average_time_difference: float
            The average time difference to add to the stimulation time

        Returns
        -------
        time_peaks: list
            The time of the stimulation
        peaks: list
            The index of the stimulation
        """
        derivative = np.diff(stimulation_signal)

        threshold_positive = np.mean(heapq.nlargest(200, stimulation_signal)) / 2
        threshold_negative = np.mean(heapq.nsmallest(200, stimulation_signal)) / 2

        positive = np.where(stimulation_signal > threshold_positive)[0]
        negative = np.where(stimulation_signal < threshold_negative)[0]

        if negative[0] < positive[0]:
            derivative = -derivative

        derivative_threshold = np.mean(heapq.nlargest(200, derivative)) / 2

        above_threshold = np.where(derivative > derivative_threshold)[0]

        peaks = [above_threshold[0]]
        for index in above_threshold[1:]:
            if index - peaks[-1] > 10:
                peaks.append(index)
        time_peaks = [time[peak] for peak in peaks]

        if average_time_difference:
            time_peaks = np.array(time_peaks) + average_time_difference
            peaks = np.array(peaks) + int(average_time_difference * self.frequency_acquisition_stim)

        if isinstance(time_peaks, np.ndarray):
            time_peaks = time_peaks.tolist()
        if isinstance(peaks, np.ndarray):
            peaks = peaks.tolist()

        return time_peaks, peaks

    def slice_data(self, data):
        self.load_analog()
        stimulation_time, peaks_index = self._get_stimulation(self.time_stim, self.data_stim, self.average_time_difference)

        sliced_time = []
        sliced_data = []
        sliced_stim_time = []

        temp_peaks_index = peaks_index
        i = 0

        delta = self.frequency_acquisition_stim / self.frequency_stimulation * 1.3

        while len(temp_peaks_index) != 0 and i < len(peaks_index) - 1:
            first = peaks_index[i]
            first_stim = i
            while i + 1 < len(peaks_index) and peaks_index[i + 1] - peaks_index[i] < delta:
                i += 1

            j = int(peaks_index[i] * self.frequency_acquisition / self.frequency_acquisition_stim) + 15

            while data[j + 1] - data[j] < 0:
                j += 1

            last = j
            #if i + 1 >= len(peaks_index):
                #last = first + self.frequency_acquisition_stim * 2
            #else:
                #last = peaks_index[i + 1] - 1

            first = int(first * self.frequency_acquisition / self.frequency_acquisition_stim)
            #last = int(last * self.frequency_acquisition / self.frequency_acquisition_stim) + 1

            sliced_time.append(self.time[first:last])
            sliced_data.append(data[first:last])
            sliced_stim_time.append(stimulation_time[first_stim:i])

            i += 1

            temp_peaks_index = [peaks for peaks in temp_peaks_index if peaks > last]

        return sliced_time, sliced_data, sliced_stim_time

    @staticmethod
    def _set_time_continuity(sliced_stim_time, sliced_time):
        sliced_stim_time[0] = np.array(sliced_stim_time[0]) - sliced_time[0][0]
        sliced_time[0] = np.array(sliced_time[0]) - sliced_time[0][0]

        for i in range(len(sliced_time) - 1):
            sliced_stim_time[i + 1] = np.array(sliced_stim_time[i + 1]) - (sliced_time[i + 1][0] - sliced_time[i][-1])
            sliced_time[i + 1] = np.array(sliced_time[i + 1]) - (sliced_time[i+1][0] - sliced_time[i][-1])

        return sliced_time, sliced_stim_time

    def _get_q(self):
        self.load_c3d()
        self._get_wrist_position()
        self.forearm_position = self._get_segment_vector(start=self.data_dict["elbow_r"], end=self.wirst_position)
        self.humerus_position = self._get_segment_vector(start=self.data_dict["elbow_r"], end=self.data_dict["should_r"])
        self.forearm_position_proj, self.humerus_position_proj = self._projection_vectors(self.forearm_position,
                                                                                         self.humerus_position)
        self.elbow_angle_rad = self._get_angle(self.forearm_position_proj[:2], self.humerus_position_proj[:2])
        self.Q_rad = np.pi - self.elbow_angle_rad

        return self.Q_rad

    def get_q_rad(self):
        Q_rad = self._get_q()
        return Q_rad

    def get_q_deg(self):
        Q_rad = self._get_q()
        return np.rad2deg(Q_rad)

    def get_time(self):
        return self.time

    def get_sliced_time_Q_rad(self):
        Q_rad = self._get_q()
        sliced_time, sliced_data, sliced_stim_time = self.slice_data(Q_rad)
        sliced_time, sliced_stim_time = self._set_time_continuity(sliced_stim_time, sliced_time)
        dictionary = {"q": sliced_data, "time": sliced_time, "stim_time": sliced_stim_time}
        return dictionary

    def get_sliced_time_Q_deg(self):
        Q_deg = self.get_q_deg()
        sliced_time, sliced_data, sliced_stim_time = self.slice_data(Q_deg)
        sliced_time, sliced_stim_time = self._set_time_continuity(sliced_stim_time, sliced_time)
        dictionary = {"q": sliced_data, "time": sliced_time, "stim_time": sliced_stim_time}
        return dictionary


if __name__ == "__main__":
    c3d_path = "C:\\Users\\flori_4ro0b8\\Documents\\Stage_S2M\\cocofest\\examples\\data_process\\lucie_50Hz_250-300-350-400-450usx2_22mA_1dof_1sr.c3d"
    c3d_to_q = C3dToQ(c3d_path)
    Q_rad = c3d_to_q.get_q_rad()
    time = c3d_to_q.get_time()
    dict = c3d_to_q.get_sliced_time_Q_rad()

    for i in range(len(dict["q"])):
        plt.plot(dict["time"][i], dict["q"][i])
        plt.scatter(dict["stim_time"][i], [0] * len(dict["time"][i]), color="red")
    plt.show()
