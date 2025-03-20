import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from copy import deepcopy
import heapq
import pickle

from pyomeca import Analogs


class C3dToMuscleForce:
    def __init__(self):
        pass

    def extract_force_from_c3d(
        self, c3d_path=None, calibration_matrix_path=None, for_id=True, saving_pickle_path=None, **kwargs
    ):

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

            if "input_channel" in kwargs:
                filtered_data = self.reindex_2d_list(filtered_data, kwargs["input_channel"])

            # Get force from voltage data
            if calibration_matrix_path:
                self.calibration_matrix = self.read_text_file_to_matrix(calibration_matrix_path)
                filtered_6d_force = self.calibration_matrix @ filtered_data[:6]
            else:
                if "already_calibrated" in kwargs:
                    if kwargs["already_calibrated"] is True:
                        filtered_6d_force = filtered_data[:6]
                    else:
                        raise ValueError("already_calibrated must be either True or False")
                else:
                    raise ValueError(
                        "Please specify if the data is already calibrated or not with already_calibrated input."
                        "If not, please provide a calibration matrix path"
                    )

            filtered_6d_force = self.set_zero_level(filtered_6d_force, average_on=[1000, 3000])

            if for_id:
                check_stimulation = kwargs["check_stimulation"] if "check_stimulation" in kwargs else None

                # Detect stimulation time
                if "average_time_difference" in kwargs and "frequency_acquisition" in kwargs:
                    stimulation_time, peaks = self.get_stim(
                        time,
                        raw_data[6].data,
                        average_time_difference=kwargs["average_time_difference"],
                        frequency_acquisition=kwargs["frequency_acquisition"],
                        check_stimulation=check_stimulation,
                    )
                else:
                    stimulation_time, peaks = self.get_stim(time, raw_data[6].data, check_stimulation=check_stimulation)
                # Get the data from 6D file
                sliced_time, sliced_data = self.slice_data(time, filtered_6d_force, peaks)
                temp_time = deepcopy(sliced_time)
                temp_data = deepcopy(sliced_data)

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
            last = (
                next((x for x, val in enumerate(-data[main_axis, first:]) if val < 0), len(data[main_axis, first:]))
                + first
            )

            x.append(data[0, first:last].tolist())
            y.append(data[1, first:last].tolist())
            z.append(data[2, first:last].tolist())
            mx.append(data[3, first:last].tolist())
            my.append(data[4, first:last].tolist())
            mz.append(data[5, first:last].tolist())

            sliced_time.append(time[first:last])

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

    def force_transfer(self, force):
        pass

    def get_force(self, c3d_path, calibration_matrix_path, saving_pickle_path, for_id=True):
        self.extract_force_from_c3d(
            c3d_path=c3d_path,
            calibration_matrix_path=calibration_matrix_path,
            for_id=for_id,
            saving_pickle_path=saving_pickle_path,
        )
        measured_force, stim_time = self.read_pkl_to_force_vector(pickle_path=saving_pickle_path)
        # muscle_force = self.force_transfer(measured_force)
        norm_muscle_force = self.norm_force_fun(measured_force)

        return norm_muscle_force, stim_time


if __name__ == "__main__":
    c3d_converter = C3dToMuscleForce()
    norm_muscle_force, stim_time = c3d_converter.get_force(
        c3d_path="C:\\Users\\flori_4ro0b8\\Documents\\Stage_S2M\\c3d_file\\exp_id\\id_exp_florine_50Hz_400us_15mA_test1.c3d",
        calibration_matrix_path="C:\\Users\\flori_4ro0b8\\Documents\\Stage_S2M\\matrix.txt",
        saving_pickle_path="C:\\Users\\flori_4ro0b8\\Documents\\Stage_S2M\\id_exp_florine_50Hz_400us_15mA_test1.pkl",
        for_id=True,
    )

    plt.plot(stim_time, norm_muscle_force)
    plt.show()
