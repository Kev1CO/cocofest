import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from copy import deepcopy
import heapq
import pickle

from biorbd import Model

from pyomeca import Analogs


class C3dToMuscleForce:
    def __init__(self):
        self.biceps_moment_arm = None
        self.Qddot = None
        self.Qdot = None
        self.Q = None
        self.model = None
        self.default_index = {"sensor.V1":0, "sensor.V2":1, "sensor.V3":2, "sensor.V4":3, "sensor.V5":4, "sensor.V6":5, "Electric Current.Channel_1_m":6, "Electric Current.Channel_5":7}

    def extract_force_from_c3d(
        self, c3d_path=None, calibration_matrix_path=None, for_id=True, saving_pickle_path=None, **kwargs
    ):
        time_list = []
        stim_index_list = []
        # Conversion into list
        c3d_path_list = [c3d_path] if isinstance(c3d_path, str) else c3d_path

        saving_pickle_path_list = [saving_pickle_path] if isinstance(saving_pickle_path, str) else saving_pickle_path
        if saving_pickle_path_list:
            if len(saving_pickle_path_list) != 1 and len(saving_pickle_path_list) != len(c3d_path_list):
                raise ValueError(
                    "The number of saving_pickle_path must be the same as the number of c3d_path."
                    "If you entered only one path, the file name will be iterated."
                )

        for i in range(len(c3d_path_list)):
            c3d_path = c3d_path_list[i]
            if not isinstance(c3d_path, str):
                raise TypeError("c3d_path must be a str or a list of str.")
            raw_data = Analogs.from_c3d(c3d_path)

            # Low pass filter parameters
            order = kwargs["order"] if "order" in kwargs else 1
            cutoff = kwargs["cutoff"] if "cutoff" in kwargs else 2
            if not isinstance(order, int | None) or not isinstance(cutoff, int | None):
                raise TypeError("window_length and order must be either None or int type")
            if type(order) != type(cutoff):
                raise TypeError("window_length and order must be both None or int type")

            time = raw_data.time.values.tolist()

            # Filter data from c3d file
            filtered_data = (
                np.array(raw_data.meca.low_pass(order=order, cutoff=cutoff, freq=raw_data.rate))
                if order and cutoff
                else raw_data
            )

            # lst_index = self.set_index(raw_data)
            # filtered_data = self.reindex_2d_list(filtered_data, lst_index)

            # Get force from voltage data
            if calibration_matrix_path:
                self.calibration_matrix = self.read_text_file_to_matrix(calibration_matrix_path)
                filtered_6d_force = self.calibration_matrix @ filtered_data[2:]
            else:
                if "already_calibrated" in kwargs:
                    if kwargs["already_calibrated"] is True:
                        filtered_6d_force = filtered_data[2:]
                    else:
                        raise ValueError("already_calibrated must be either True or False")
                else:
                    raise ValueError(
                        "Please specify if the data is already calibrated or not with already_calibrated input."
                        "If not, please provide a calibration matrix path"
                    )

            # plt.plot(filtered_6d_force[0])
            # plt.title("Avant set_zero_level")
            # plt.show()

            filtered_6d_force = self.set_zero_level(filtered_6d_force, average_on=[20000, 50000])
            stim_data = self.set_zero_level(raw_data[1].data, average_on=[20000, 50000])

            # plt.plot(filtered_6d_force[0])
            # plt.title("Après set_zero_level")
            # plt.show()

            if for_id:
                check_stimulation = kwargs["check_stimulation"] if "check_stimulation" in kwargs else None

                # Detect stimulation time
                if "average_time_difference" in kwargs and "frequency_acquisition" in kwargs:
                    stimulation_time, peaks = self.get_stim(
                        time,
                        stim_data,
                        average_time_difference=kwargs["average_time_difference"],
                        frequency_acquisition=kwargs["frequency_acquisition"],
                        check_stimulation=check_stimulation,
                    )
                else:
                    stimulation_time, peaks = self.get_stim(time, stim_data, check_stimulation=check_stimulation)
                # Get the data from 6D file
                sliced_time, sliced_data = self.slice_data(time, filtered_6d_force, peaks)
                temp_time = deepcopy(sliced_time)
                temp_data = deepcopy(sliced_data)

                time_list.append(temp_time)
                stim_index_list.append(peaks)

                # stim = []
                # for i in stimulation_time:
                #     stim.append(stim_data[int(i * 10000)])
                # plt.plot(time, stim_data, color='blue')
                # plt.scatter(stimulation_time, stim, color='red')
                # plt.show()

                # force_stim = []
                # for i in stimulation_time:
                #     force_stim.append(filtered_6d_force[0, int(i*10000)])
                # plt.plot(time, filtered_6d_force[0], color='blue')
                # plt.scatter(stimulation_time, force_stim, color='red')
                # plt.show()

                # plt.plot(sliced_time, sliced_data[0])
                # plt.show()

                if "plot" in kwargs:
                    if kwargs["plot"]:
                        for k in range(len(sliced_time)):
                            plt.plot(sliced_time[k], sliced_data[0][k])
                        for k in range(len(peaks)):
                            plt.plot(time[peaks[k]], filtered_6d_force[0][peaks[k]], "x")
                        plt.show()

                # Save data as dictionary in pickle file
                if saving_pickle_path_list:
                    if len(saving_pickle_path_list) == 1:
                        if saving_pickle_path_list[:-4] == ".pkl":
                            save_pickle_path = saving_pickle_path_list[:-4] + "_" + str(i) + ".pkl"
                        else:
                            save_pickle_path = saving_pickle_path_list[0] + "_" + str(i) + ".pkl"
                    else:
                        save_pickle_path = saving_pickle_path_list[i]

                    dictionary = {
                        "time": sliced_time,
                        "x": sliced_data[0],
                        "y": sliced_data[1],
                        "z": sliced_data[2],
                        "mx": sliced_data[3],
                        "my": sliced_data[4],
                        "mz": sliced_data[5],
                        "stim_time": stimulation_time,
                    }
                    with open(save_pickle_path, "wb") as file:
                        pickle.dump(dictionary, file)
            else:
                if saving_pickle_path_list[0].endswith(".pkl"):
                    save_pickle_path = saving_pickle_path_list[:-4] + "_" + str(i) + ".pkl"
                else:
                    save_pickle_path = saving_pickle_path_list[0] + "_" + str(i) + ".pkl"

                dictionary = {
                    "time": time,
                    "x": filtered_6d_force[0],
                    "y": filtered_6d_force[1],
                    "z": filtered_6d_force[2],
                    "mx": filtered_6d_force[3],
                    "my": filtered_6d_force[4],
                    "mz": filtered_6d_force[5],
                    "stim_time": raw_data[6],
                }
                with open(save_pickle_path, "wb") as file:
                    pickle.dump(dictionary, file)
        return time_list, stim_index_list

    @staticmethod
    def read_text_file_to_matrix(file_path):
        """

        Parameters
        ----------
        file_path: str
            Path to calibration matrix file

        Returns
        -------
        Calibration matrix as an array (6x6)
        """
        try:
            # Read the text file and split lines
            with open(file_path, "r") as file:
                lines = file.readlines()
            # Initialize an empty list to store the rows
            data = []
            # Iterate through the lines, split by tabs, and convert to float
            for line in lines:
                row = [float(value) for value in line.strip().split()]
                data.append(row)
            # Convert the list of lists to a NumPy matrix
            matrix = np.array(data)
            return matrix
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return None

    def set_index(self, raw_data):
        lst_index = []
        for i in range(len(raw_data.channel)):
            index_name = raw_data.channel[i].item()
            lst_index.append(self.default_index[index_name])

        return lst_index

    @staticmethod
    def reindex_2d_list(data, new_indices):
        """
        Parameters
        ----------
        data: array
            The data to reindex
        new_indices: list
            Contains new indices

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

    @staticmethod
    def set_zero_level(data: np.array, average_length: int = 1000, average_on: list[int] = None):
        """
        Parameters
        ----------
        data: array
            The data to set the zero level
        average_length: int
            The number of points to average
        average_on: list[int, int]
            The window to average

        Returns
        -------
        The data with the zero level set
        """
        if len(data.shape) == 1:
            return (
                data - np.mean(data[average_on[0] : average_on[1]])
                if average_on
                else data - np.mean(data[:average_length])
            )
        else:
            for i in range(data.shape[0]):
                data[i] = (
                    data[i] - np.mean(data[i][average_on[0] : average_on[1]])
                    if average_on
                    else data[i] - np.mean(data[i][:average_length])
                )
            return data

    @staticmethod
    def slice_data(time, data, stimulation_peaks, main_axis=0):
        """
        Parameters
        ----------
        time: list
        data : array
            Contains force data from c3d file
        stimulation_peaks: list
            Contains time stimulation indexes
        main_axis: int
            Choose the main axis to consider (default : x)

        Returns
        -------
        sliced_time: list
            Contains
        sliced_data: list
            Contains
        """
        sliced_time = []
        temp_stimulation_peaks = stimulation_peaks
        x = []
        y = []
        z = []
        mx = []
        my = []
        mz = []

        while len(temp_stimulation_peaks) != 0:
            substact_to_zero = data[:, temp_stimulation_peaks[0]]
            for i in range(len(substact_to_zero)):
                data[:, temp_stimulation_peaks[0] :][i] = data[:, temp_stimulation_peaks[0] :][i] - substact_to_zero[i]

            first = temp_stimulation_peaks[0]
            # print(temp_stimulation_peaks)
            # print(len(temp_stimulation_peaks))
            # print(first)
            last = (
                next((x for x, val in enumerate(-data[main_axis, first:]) if val < 0), len(data[main_axis, first:]))
                + first
            )

            x.extend(data[0, first:last].tolist())
            y.extend(data[1, first:last].tolist())
            z.extend(data[2, first:last].tolist())
            mx.extend(data[3, first:last].tolist())
            my.extend(data[4, first:last].tolist())
            mz.extend(data[5, first:last].tolist())

            sliced_time.extend(time[first:last])

            temp_stimulation_peaks = [peaks for peaks in temp_stimulation_peaks if peaks > last]
        sliced_data = [x, y, z, mx, my, mz]
        return sliced_time, sliced_data

    @staticmethod
    def get_stim(
        time,
        stimulation_signal,
        average_time_difference: float = None,
        frequency_acquisition: int = None,
        check_stimulation: bool = False,
    ):
        """
        Parameters
        ----------
        time
        stimulation_signal
        average_time_difference: float
            Time gap between the sending of the stim and the actual stim (s)
        frequency_acquisition: float
            Parameter of the measuring device (Hz)
        check_stimulation: bool
            If you want to plot the stimulation signal

        Returns
        -------
        time_peaks: list
            Contains peaks' time
        peaks: list
            Contains peaks' indexes
        """
        if average_time_difference:
            if not isinstance(average_time_difference, float):
                raise TypeError("average_time_difference must be a float.")
            if not frequency_acquisition:
                raise ValueError("Please specify the acquisition frequency when average_time_difference is entered.")
            if not isinstance(frequency_acquisition, int):
                raise TypeError("frequency_acquisition must be an integer.")
            if abs(average_time_difference) < 1 / frequency_acquisition:
                raise ValueError(
                    "average_time_difference must be bigger than the inverse of the acquisition frequency."
                )
        # Definition of thresholds : the largest and smallest values
        threshold_positive = np.mean(heapq.nlargest(200, stimulation_signal)) / 2
        threshold_negative = np.mean(heapq.nsmallest(200, stimulation_signal)) / 2

        positive = np.where(stimulation_signal > threshold_positive)
        negative = np.where(stimulation_signal < threshold_negative)

        if negative[0][0] < positive[0][0]:
            stimulation_signal = -stimulation_signal  # invert the signal if the first peak is negative
            threshold = -threshold_negative
        else:
            threshold = threshold_positive

        # peaks contains the index of peaks
        peaks, _ = find_peaks(stimulation_signal, distance=10, height=threshold)

        # time_peaks contains the time of the peaks
        time_peaks = []
        for i in range(len(peaks)):
            time_peaks.append(time[peaks[i]])

        # If you want to plot the stimulation signal
        if check_stimulation:
            for k in range(len(time_peaks)):
                plt.plot(time_peaks[k], stimulation_signal[peaks[k]], "x")
            plt.plot(time, stimulation_signal)
            plt.show()

        if average_time_difference:
            time_peaks = np.array(time_peaks) + average_time_difference
            peaks = np.array(peaks) + int(average_time_difference * frequency_acquisition)

        if isinstance(time_peaks, np.ndarray):
            time_peaks = time_peaks.tolist()
        if isinstance(peaks, np.ndarray):
            peaks = peaks.tolist()

        return time_peaks, peaks

    @staticmethod
    def stimulation_detection_for_time_diff(time, stimulation_signal):
        """

        Parameters
        ----------
        time
        stimulation_signal

        Returns
        -------

        """
        # Definition of thresholds : the largest and smallest values
        threshold_positive = np.mean(heapq.nlargest(200, stimulation_signal)) / 2
        threshold_negative = np.mean(heapq.nsmallest(200, stimulation_signal)) / 2

        positive = np.where(stimulation_signal > threshold_positive)
        negative = np.where(stimulation_signal < threshold_negative)

        if negative[0][0] < positive[0][0]:
            stimulation_signal = -stimulation_signal  # invert the signal if the first peak is negative
            threshold = -threshold_negative
        else:
            threshold = threshold_positive

        # peaks contains the index of peaks
        peaks, _ = find_peaks(stimulation_signal, distance=10, height=threshold)

        # time_peaks contains the time of the peaks
        time_peaks = []
        for i in range(len(peaks)):
            time_peaks.append(time[peaks[i]])

        if isinstance(time_peaks, np.ndarray):
            time_peaks = time_peaks.tolist()
        if isinstance(peaks, np.ndarray):
            peaks = peaks.tolist()

        return time_peaks, peaks

    def get_avg_time_diff(self, c3d_path: str | list[str] = None):
        # Conversion into list
        c3d_path_list = [c3d_path] if isinstance(c3d_path, str) else c3d_path

        for i in range(len(c3d_path_list)):
            c3d_path = c3d_path_list[i]
            if not isinstance(c3d_path, str):
                raise TypeError("c3d_path must be a str or a list of str.")
            raw_data = Analogs.from_c3d(c3d_path)
            time = raw_data.time.values.tolist()

            time_peaks_muscle, peaks_muscle = self.stimulation_detection_for_time_diff(
                time=time, stimulation_signal=raw_data.values[0]
            )

            time_peaks_measured, peaks_measured = self.stimulation_detection_for_time_diff(
                time=time, stimulation_signal=raw_data.values[1]
            )

            if len(time_peaks_measured) == len(time_peaks_muscle):
                time_diff = np.array(time_peaks_muscle) - np.array(time_peaks_measured)

                avg_time_diff = np.mean(time_diff)
            else:
                raise ValueError("Measured data and muscle data must have same frequency")

            return avg_time_diff

    @staticmethod
    def read_pkl_to_force_vector(pickle_path):
        """
        Parameters
        ----------
        pickle_path: str

        Returns
        -------
        An array of the 3 force components
        """
        if isinstance(pickle_path, str):
            pickle_path = pickle_path[:-4] + ".pkl_0" + pickle_path[-4:]
            with open(pickle_path, "rb") as f:
                data = pickle.load(f)
                force = [data["x"], data["y"], data["z"]]
                stim_time = data["stim_time"]
        else:
            raise ValueError("pickle_path must be a string")

        return force, stim_time

    @staticmethod
    def norm_force_fun(force):
        """
        Parameters
        ----------
        force: array

        Returns
        -------
        A list of the normalized force
        """
        force_x = force[0]
        force_y = force[1]
        force_z = force[2]

        norm_force = np.sqrt(np.square(force_x) + np.square(force_y) + np.square(force_z))

        return norm_force

    @staticmethod
    def local_sensor_to_local_hand(sensor_data: np.array) -> np.array:
        """
        This function is used to convert the sensor data from the local axis to the local hand axis.
        This a rotation along x axis whatever the elbow angle
        Parameters
        ----------
        sensor_data

        Returns
        -------

        """
        rotation_angle = np.pi / 2
        rotation_matrix = np.array([[1, 0, 0], [0, np.cos(rotation_angle), - np.sin(rotation_angle)], [0, np.sin(rotation_angle, np.cos[rotation_angle])]])
        new_sensor_data = rotation_matrix @ sensor_data

        return new_sensor_data

    @staticmethod
    def local_hand_to_local(force_data: np.array, elbow_angle: float) -> np.array:
        """
        This function is used to convert the sensor data from the local hand axis to the local muscle axis.
        This is a rotation of elbow angle along z axis
        Parameters
        ----------
        force_data
        elbow_angle: float
            Must be in radians

        Returns
        -------

        """
        rotation_angle = np.pi - np.radians(elbow_angle)
        rotation_matrix = np.array([[np.cos(rotation_angle), - np.sin(rotation_angle), 0], [np.sin(rotation_angle), np.cos(rotation_angle), 0], [0, 0, 1]])
        new_force_data = rotation_matrix @ force_data

        return new_force_data

    def load_model(self, forearm_angle: int | float):
        # Load a predefined model
        self.model = Model("model_msk/simplified_UL_Seth.bioMod")
        # Get number of q, qdot, qddot
        nq = self.model.nbQ()
        nqdot = self.model.nbQdot()
        nqddot = self.model.nbQddot()

        # Choose a position/velocity/acceleration to compute dynamics from
        if nq != 2:
            raise ValueError("The number of degrees of freedom has changed.")  # 0
        self.Q = np.array([0.0, np.radians(forearm_angle)])  # "0" arm along body and "1.57" 90° forearm position  |__.
        self.Qdot = np.zeros((nqdot,))  # speed null
        self.Qddot = np.zeros((nqddot,))  # acceleration null

        # Biceps moment arm
        self.model.musclesLengthJacobian(self.Q).to_array()
        if self.model.muscleNames()[1].to_string() != "BIClong":
            raise ValueError("Biceps muscle index as changed.")  # biceps is index 1 in the model
        self.biceps_moment_arm = self.model.musclesLengthJacobian(self.Q).to_array()[1][1]

        # Expressing the external force array [Mx, My, Mz, Fx, Fy, Fz]
        # experimentally applied at the hand into the last joint
        if self.model.segments()[15].name().to_string() != "r_ulna_radius_hand_r_elbow_flex":
            raise ValueError("r_ulna_radius_hand_r_elbow_flex index as changed.")

        if self.model.markerNames()[3].to_string() != "r_ulna_radius_hand":
            raise ValueError("r_ulna_radius_hand marker index as changed.")

        if self.model.markerNames()[4].to_string() != "hand":
            raise ValueError("hand marker index as changed.")

    def get_muscle_positions(self):
        pass

    def project_force_on_muscle(self):
        pass

    def force_transfer(self, force):
        pass

    def get_force(self, c3d_path, calibration_matrix_path, saving_pickle_path, for_id=True):
        time_list, stim_index_list = self.extract_force_from_c3d(
            c3d_path=c3d_path,
            calibration_matrix_path=calibration_matrix_path,
            for_id=for_id,
            saving_pickle_path=saving_pickle_path,
        )
        measured_force, stim_time = self.read_pkl_to_force_vector(pickle_path=saving_pickle_path)
        # muscle_force = self.force_transfer(measured_force)
        norm_muscle_force = self.norm_force_fun(measured_force) * 10

        return norm_muscle_force, stim_time, time_list, stim_index_list


if __name__ == "__main__":
    c3d_converter = C3dToMuscleForce()
    norm_muscle_force, stim_time, time_list, stim_index_list = c3d_converter.get_force(
        c3d_path="C:\\Users\\flori_4ro0b8\\Documents\\Stage_S2M\\c3d_file\\exp_id\\id_exp_florine_50Hz_400us_15mA_test1.c3d",
        calibration_matrix_path="C:\\Users\\flori_4ro0b8\\Documents\\Stage_S2M\\matrix.txt",
        saving_pickle_path="C:\\Users\\flori_4ro0b8\\Documents\\Stage_S2M\\id_exp_florine_50Hz_400us_15mA_test1.pkl",
        for_id=True,
    )
    time = np.linspace(0, len(norm_muscle_force), len(norm_muscle_force))
    plt.plot(time, norm_muscle_force)
    plt.show()
