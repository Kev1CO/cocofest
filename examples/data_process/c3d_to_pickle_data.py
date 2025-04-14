import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import heapq
import pickle

from pyomeca import Analogs


class C3dToPickleData:
    def __init__(
        self,
        c3d_path=None,
        calibration_matrix_path=None,
        for_id=True,
        saving_pickle_path=None,
        **kwargs,
    ):

        self.default_index = {
            "sensor.V1": 0,
            "sensor.V2": 1,
            "sensor.V3": 2,
            "sensor.V4": 3,
            "sensor.V5": 4,
            "sensor.V6": 5,
            "Torque.ergometer": 6,
            "Electric Current.Channel_5": 7,
        }

        self.time_list = []
        self.stim_index_list = []
        self.filtered_data = None
        self.filtered_6d_force = None
        self.torque_ergometer = None
        self.stim_data = None
        self.calibration_matrix = None
        self.sliced_time = None
        self.sliced_data = None
        self.avg_stim_time = None

        # Conversion into list
        c3d_path_list = [c3d_path] if isinstance(c3d_path, str) else c3d_path

        saving_pickle_path_list = (
            [saving_pickle_path]
            if isinstance(saving_pickle_path, str)
            else saving_pickle_path
        )
        if saving_pickle_path_list:
            if len(saving_pickle_path_list) != 1 and len(
                saving_pickle_path_list
            ) != len(c3d_path_list):
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
            cutoff = kwargs["cutoff"] if "cutoff" in kwargs else 10
            if not isinstance(order, int | None) or not isinstance(cutoff, int | None):
                raise TypeError(
                    "window_length and order must be either None or int type"
                )
            if type(order) != type(cutoff):
                raise TypeError("window_length and order must be both None or int type")

            time = raw_data.time.values.tolist()

            # Filter data from c3d file
            self.filtered_data = (
                -np.array(
                    raw_data.meca.low_pass(
                        order=order, cutoff=cutoff, freq=raw_data.rate
                    )
                )
                if order and cutoff
                else raw_data
            )

            # Reindex data
            lst_index = self.set_index(raw_data)
            raw_data_reindex = self.reindex_2d_list(raw_data.data, lst_index)
            self.filtered_data = self.reindex_2d_list(self.filtered_data, lst_index)

            self.torque_ergometer = self.filtered_data[6]

            # Get force from voltage data
            if calibration_matrix_path:
                self.calibration_matrix = self.read_text_file_to_matrix(calibration_matrix_path)
                self.filtered_6d_force = self.calibration_matrix @ self.filtered_data[:6]
                non_filtered_force = self.calibration_matrix @ raw_data_reindex[:6]

            else:
                if "already_calibrated" in kwargs:
                    if kwargs["already_calibrated"] is True:
                        self.filtered_6d_force = self.filtered_data[:6]
                    else:
                        raise ValueError("already_calibrated must be either True or False")

                else:
                    raise ValueError(
                        "Please specify if the data is already calibrated or not with already_calibrated input."
                        "If not, please provide a calibration matrix path"
                    )

            self.filtered_6d_force = self.set_zero_level(self.filtered_6d_force, average_on=[0, 20000])
            non_filtered_force = self.set_zero_level(non_filtered_force, average_on=[0, 20000])
            self.stim_data = self.set_zero_level(raw_data_reindex[-1], average_on=[0, 20000])
            self.torque_ergometer = self.set_zero_level(self.torque_ergometer, average_on=[0, 20000])
            self.filtered_data = self.set_zero_level(self.filtered_data, average_on=[0, 20000])

            # plt.plot(time, -self.filtered_data[4], label='my', color='orange')
            # plt.plot(time, self.torque_ergometer, label='ergometer', color='red')
            # plt.legend()
            # plt.show()

            self.torque_ergometer = np.array(self.torque_ergometer) * np.mean(self.filtered_6d_force[4]) / np.mean(self.filtered_data[4])

            plt.plot(time, -self.filtered_6d_force[4], label='my', color='orange')
            plt.plot(time, self.torque_ergometer, label='ergometer', color='red')
            plt.legend()
            #plt.show()

            # plt.plot(time, -self.filtered_6d_force[0], color='blue', alpha=0.5, label='low pass 10Hz')
            # plt.plot(time, -non_filtered_force[0], color='green', alpha=0.5, label='no filter')
            # plt.legend()
            # plt.show()

            if for_id:
                self.avg_stim_time = self.get_avg_time_diff()

                self.rest_time = kwargs["rest_time"] if "rest_time" in kwargs else 1

                self.frequency_acquisition = (
                    kwargs["frequency_acquisition"]
                    if "frequency_acquisition" in kwargs
                    else 10000
                )
                self.frequency_stimulation = (
                    kwargs["frequency_stimulation"]
                    if "frequency_stimulation" in kwargs
                    else 50
                )

                # Detect stimulation time
                stimulation_time, peaks = self.get_stimulation(
                    time=time,
                    stimulation_signal=self.stim_data,
                    average_time_difference=self.avg_stim_time,
                )

                # stim = []
                # for i in stimulation_time:
                #     stim.append(self.stim_data[int(i * 10000)])
                # plt.plot(time, self.stim_data, color='blue', alpha=0.5)
                # plt.scatter(stimulation_time, [0] * len(stimulation_time), color='red', label='derivative')
                # plt.legend()
                # plt.show()

                #Add ergometer torque to data
                if isinstance(self.filtered_6d_force, list):
                    self.filtered_6d_force += self.torque_ergometer
                else:
                    self.filtered_6d_force = np.concatenate((self.filtered_6d_force, np.array(self.torque_ergometer).reshape(1, -1)))

                # Slice the data from 6D file
                self.sliced_time, self.sliced_data = self.slice_data(
                    time=time, data=self.filtered_6d_force, stimulation_index=peaks
                )

                # force_stim = []
                # for j in stimulation_time:
                #     force_stim.append(self.filtered_6d_force[0, int(j * 10000)])
                # plt.plot(time, self.filtered_6d_force[0], color='blue', alpha=0.5, label='low pass 10Hz')
                # plt.plot(time, self.stim_data*1000, color='black', alpha=0.5)
                # plt.plot(time, non_filtered_force[0], color='green', alpha=0.5, label='no filter')
                # plt.scatter(stimulation_time, force_stim, color='red', label='derivative')
                # plt.legend()
                # plt.show()

                color = [
                    "blue",
                    "orange",
                    "green",
                    "red",
                    "purple",
                    "pink",
                    "black",
                    "brown",
                    "gray",
                    "lightblue",
                ]
                for j in range(len(self.sliced_time)):
                    plt.plot(
                        self.sliced_time[j], self.sliced_data[0][j], color=color[j]
                    )
                plt.show()

                self.set_to_zero_slice()

                for j in range(len(self.sliced_time)):
                    plt.plot(
                        self.sliced_time[j], self.sliced_data[0][j], color=color[j]
                    )
                plt.show()

                # Save data as dictionary in pickle file
                if saving_pickle_path_list:
                    if len(saving_pickle_path_list) == 1:
                        if saving_pickle_path_list[:-4] == ".pkl":
                            save_pickle_path = (
                                saving_pickle_path_list[:-4] + "_" + str(i) + ".pkl"
                            )
                        else:
                            save_pickle_path = (
                                saving_pickle_path_list[0] + "_" + str(i) + ".pkl"
                            )
                    else:
                        save_pickle_path = saving_pickle_path_list[i]

                    dictionary = {
                        "time": self.sliced_time,
                        "x": self.sliced_data[0],
                        "y": self.sliced_data[1],
                        "z": self.sliced_data[2],
                        "mx": self.sliced_data[3],
                        "my": self.sliced_data[4],
                        "mz": self.sliced_data[5],
                        "torque_ergometer": self.sliced_data[6],
                        "stim_time": stimulation_time,
                    }
                    with open(save_pickle_path, "wb") as file:
                        pickle.dump(dictionary, file)
            else:
                if saving_pickle_path_list[0].endswith(".pkl"):
                    save_pickle_path = (
                        saving_pickle_path_list[:-4] + "_" + str(i) + ".pkl"
                    )
                else:
                    save_pickle_path = (
                        saving_pickle_path_list[0] + "_" + str(i) + ".pkl"
                    )

                dictionary = {
                    "time": time,
                    "x": self.filtered_6d_force[0],
                    "y": self.filtered_6d_force[1],
                    "z": self.filtered_6d_force[2],
                    "mx": self.filtered_6d_force[3],
                    "my": self.filtered_6d_force[4],
                    "mz": self.filtered_6d_force[5],
                    "torque_ergometer": self.filtered_6d_force[6],
                    "stim_time": raw_data[6],
                }
                with open(save_pickle_path, "wb") as file:
                    pickle.dump(dictionary, file)

    @staticmethod
    def read_text_file_to_matrix(file_path):
        """
        This function reads a txt file containing a calibration matrix and returns it as a NumPy array.
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

    def set_index(self, raw_data):
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
        for i in range(len(self.default_index.keys())):
            name = list(self.default_index.keys())[i]
            indice = self.get_index(name, raw_data.channel)
            lst_index.append(indice)
        return lst_index

    @staticmethod
    def reindex_2d_list(data, new_indices):
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

    @staticmethod
    def set_zero_level(
        data: np.array, average_length: int = 1000, average_on: list[int] = None
    ):
        """
        This function sets the zero level of the data by subtracting the mean of the first n points
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
        if isinstance(data, list):
            data = np.array(data)

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

    def slice_data(self, time, data, stimulation_index):
        """
        This function slices the data into trains based on the stimulation index
        Parameters
        ----------
        time: array
            The time data
        data: array
            The data to slice
        stimulation_index: list
            The index of the stimulation

        Returns
        -------
        sliced_time: list
            The sliced time data
        sliced_data: list
            The sliced data
        """
        x = []
        y = []
        z = []
        mx = []
        my = []
        mz = []
        torque_ergometer = []

        sliced_time = []

        temp_stimulation_index = stimulation_index

        i = 0
        delta = self.frequency_acquisition / self.frequency_stimulation * 1.3

        while len(temp_stimulation_index) != 0 and i < len(stimulation_index) - 1:
            first = stimulation_index[i]
            while (
                i + 1 < len(stimulation_index)
                and stimulation_index[i + 1] - stimulation_index[i] < delta
            ):
                i += 1

            if i + 1 >= len(stimulation_index):
                last = first + self.frequency_acquisition * (self.rest_time + 1)
            else:
                last = stimulation_index[i + 1] - 1

            x.append(data[0][first:last].tolist())
            y.append(data[1][first:last].tolist())
            z.append(data[2][first:last].tolist())
            mx.append(data[3][first:last].tolist())
            my.append(data[4][first:last].tolist())
            mz.append(data[5][first:last].tolist())
            torque_ergometer.append(data[6][first:last].tolist())
            sliced_time.append(time[first:last])

            i += 1

            temp_stimulation_index = [
                peaks for peaks in temp_stimulation_index if peaks > last
            ]

        sliced_data = [x, y, z, mx, my, mz, torque_ergometer]

        return sliced_time, sliced_data

    def set_to_zero_slice(self):
        """
        This function sets the zero level of the sliced data by subtracting the first value of each slice
        """
        for i in range(len(self.sliced_data)):
            for j in range(len(self.sliced_data[i])):
                self.sliced_data[i][j] = (
                    np.array(self.sliced_data[i][j]) - self.sliced_data[i][j][0]
                )
                for k, val in enumerate(self.sliced_data[i][j][1:]):
                    if val <= 0:
                        self.sliced_data[i][j][k + 1 :] = 0
                        break
                if np.all(self.sliced_data[i][j][1:] > 0):
                    derivative = np.diff(self.sliced_data[i][j])
                    drop_detected = False
                    for k in range(len(derivative) - 1):
                        if not drop_detected and derivative[k] <= -0.01:
                            drop_detected = True
                        elif drop_detected and derivative[k + 1] >= 0:
                            self.sliced_data[i][j][k + 1 :] = 0
                            break

    def get_stimulation(self, time, stimulation_signal, average_time_difference=None):
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
            peaks = np.array(peaks) + int(
                average_time_difference * self.frequency_acquisition
            )

        if isinstance(time_peaks, np.ndarray):
            time_peaks = time_peaks.tolist()
        if isinstance(peaks, np.ndarray):
            peaks = peaks.tolist()

        return time_peaks, peaks

    @staticmethod
    def stimulation_detection_for_time_diff(time, stimulation_signal):
        """
        This function detects the stimulation time and returns the time and index of the stimulation
        Parameters
        ----------
        time: array
            The time data
        stimulation_signal: array
            The stimulation signal

        Returns
        -------
        time_peaks: list
            The time of the stimulation
        peaks: list
            The index of the stimulation
        """
        # Definition of thresholds : the largest and smallest values
        threshold_positive = np.mean(heapq.nlargest(200, stimulation_signal)) / 2
        threshold_negative = np.mean(heapq.nsmallest(200, stimulation_signal)) / 2

        positive = np.where(stimulation_signal > threshold_positive)
        negative = np.where(stimulation_signal < threshold_negative)

        if negative[0][0] < positive[0][0]:
            stimulation_signal = (
                -stimulation_signal
            )  # invert the signal if the first peak is negative
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

    def get_avg_time_diff(
        self, c3d_path_stim_diff: str | list[str] = "stim_diff_50Hz.c3d"
    ):
        """
        This function calculates the average time difference between the stimulation time and the measured data
        Parameters
        ----------
        c3d_path_stim_diff: str | list[str]
            The path to the c3d file containing the stimulation signal and the measured data

        Returns
        -------
        avg_time_diff: float
            The average time difference between the stimulation time and the measured data
        """
        # Conversion into list
        c3d_path_list = (
            [c3d_path_stim_diff]
            if isinstance(c3d_path_stim_diff, str)
            else c3d_path_stim_diff
        )

        for i in range(len(c3d_path_list)):
            c3d_path = c3d_path_list[i]
            if not isinstance(c3d_path, str):
                raise TypeError("c3d_path must be a str or a list of str.")
            raw_data = Analogs.from_c3d(c3d_path)
            time = raw_data.time.values.tolist()

            time_peaks_muscle, peaks_muscle = self.stimulation_detection_for_time_diff(
                time=time, stimulation_signal=raw_data.values[0]
            )

            time_peaks_measured, peaks_measured = (
                self.stimulation_detection_for_time_diff(
                    time=time, stimulation_signal=raw_data.values[1]
                )
            )

            if len(time_peaks_measured) == len(time_peaks_muscle):
                time_diff = np.array(time_peaks_muscle) - np.array(time_peaks_measured)

                avg_time_diff = np.mean(time_diff)
            else:
                raise ValueError(
                    "Measured data and muscle data must have same frequency"
                )

            return avg_time_diff


if __name__ == "__main__":
    C3dToPickleData(
        c3d_path="essai5_florine_50Hz_400us_15mA_TR1s_vertical.c3d",
        calibration_matrix_path="matrix.txt",
        saving_pickle_path="essai5_florine_50Hz_400us_15mA_TR1s_vertical.pkl",
        for_id=True,
        frequency_acquisition=10000,
        frequency_stimulation=50,
        rest_time=1,
    )
