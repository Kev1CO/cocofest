import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import heapq
import pickle

from pyomeca import Analogs

from biorbd import Model

"""This class provides a method to process c3d files and extract 6D force data. 
- Please be aware that names of Vicon outputs can be different from mine, you can change them in the dictionary default_index.
- The average stimulation time difference (avg_stim_time) is based on the average time difference between the stimulation 
signal sent to Vicon and the one send to the electrodes. It depends on several factors and needs to be calculated again.
You can use the method _get_avg_time_diff to calculate it."""
class C3dToForce:
    def __init__(
        self,
        c3d_path=None,
        calibration_matrix_path=None,
        saving_pickle_path=None,
        model_path: str = None,
        muscle_name: str | list[str] = None,
        dof_name: str | list[str] = None,
        elbow_angle: int | float = 90,
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
        self.handle_dictionary = {}

        if c3d_path is None:
            raise ValueError("Please provide a c3d paths.")

        self.c3d_path = c3d_path

        self.saving_pickle_path = saving_pickle_path if saving_pickle_path else f"{c3d_path[:-3]}pkl"

        # Needs to be changed if you're using a different experimental set-up
        self.avg_stim_time = {20: 0.0008799999999999671, 33: 0.0008608695652174252, 50: 0.0008739549839228076}

        self.rest_time = kwargs["rest_time"] if "rest_time" in kwargs else 1

        self.frequency_acquisition = kwargs["frequency_acquisition"] if "frequency_acquisition" in kwargs else 10000

        self.frequency_stimulation = kwargs["frequency_stimulation"] if "frequency_stimulation" in kwargs else 50
        if "frequency_stimulation" not in kwargs:
            raise Warning("Please provide the frequency of stimulation, the default value is 50Hz.")

        self.calibration_matrix_path = calibration_matrix_path

        self.already_calibrated = kwargs["already_calibrated"] if "already_calibrated" in kwargs else False

        self.order = kwargs["order"] if "order" in kwargs else 1
        self.cutoff = kwargs["cutoff"] if "cutoff" in kwargs else 10
        if not isinstance(self.order, int | None) or not isinstance(self.cutoff, int | None):
            raise TypeError("window_length and order must be either None or int type")

        if type(self.order) != type(self.cutoff):
            raise TypeError("window_length and order must be both None or int type")

        if "transfer_force" in kwargs and kwargs["transfer_force"]:
            if model_path is None:
                raise ValueError("Please provide a path to the model.")
            if not isinstance(model_path, str):
                raise TypeError("Please provide a str type model path.")

            self.model_path = model_path
            self.local_data = None
            self.model = None
            self.Q = None
            self.Qdot = None
            self.Qddot = None
            self.dof_name = dof_name
            self.muscle_name = muscle_name
            self.muscle_moment_arm = None
            self.muscle_force_vector = None
            self.muscle_force_vector_list = []
            self.saved_dictionary = {}

            # Load the model
            self.load_model(elbow_angle)

            # Saving muscle/dof names and indexes as dict
            self.muscle_name_index = {}
            for i in range(len(self.model.muscleNames())):
                self.muscle_name_index[self.model.muscleNames()[i].to_string()] = i

            self.dof_name_index = {}
            for i in range(len(self.model.nameDof())):
                self.dof_name_index[self.model.nameDof()[i].to_string()] = i

            if self.muscle_name not in self.muscle_name_index.keys():
                raise ValueError(
                    f"Please provide a muscle name in the muscle_index dictionary : {list(self.muscle_name_index.keys())}."
                )

            if self.dof_name not in self.dof_name_index.keys():
                raise ValueError(
                    f"Please provide a dof name in the dof_index dictionary : {list(self.dof_name_index.keys())}."
                )

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
    def get_index(object, lst):
        """
        This function returns the index of the given object in the list.
        Parameters
        ----------
        object: str | int | float
            The object whose index we want to find
        lst: list
            The list in which to search for the object's index
        Returns
        -------
        The index of the object in the list
        """
        indice = 0
        for i in range(len(lst)):
            if object == lst[i]:
                indice = i
        return indice

    def set_index(self, raw_data):
        """
        This function create a list of new indexes based on the default index dict. This list has the right format to
        use the reindex_2d_list function.
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
        This function reindex a 2D list based on the new index list.
        Parameters
        ----------
        data: array
            The data to reindex
        new_indices: list
            Contains new indexes

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
        This function sets the zero level of the data by subtracting the mean of the first n points (default n=1000)
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
        This function slices the data into trains based on the stimulation indexes. It detects the stimulation trains by
        comparing the gap between two stimulation indexes. In the same stimulation train, the gap between two indexes is
        pretty constant, whereas between two indexes of two different trains, the gap is larger (above delta).
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

        self.sliced_time = []
        self.sliced_stim_time = []

        temp_stimulation_index = stimulation_index

        i = 0
        delta = self.frequency_acquisition / self.frequency_stimulation * 1.3 #choosing 30% is arbitrary, but it works well in practice

        while len(temp_stimulation_index) != 0 and i < len(stimulation_index) - 1:
            first = stimulation_index[i]
            first_stim = i
            while i + 1 < len(stimulation_index) and stimulation_index[i + 1] - stimulation_index[i] < delta:
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
            self.sliced_time.append(time[first:last])
            self.sliced_stim_time.append(self.stimulation_time[first_stim:i])

            i += 1

            temp_stimulation_index = [peaks for peaks in temp_stimulation_index if peaks > last]

        self.sliced_data = [x, y, z, mx, my, mz, torque_ergometer]

    def set_to_zero_slice(self):
        """
        This function sets the zero level of the sliced data by subtracting the first value of each slice. Each slice
        will start at 0. The rest part is also set to zero.
        """
        for i in range(len(self.sliced_data)):
            for j in range(len(self.sliced_data[i])):
                self.sliced_data[i][j] = np.array(self.sliced_data[i][j]) - self.sliced_data[i][j][0]
                for k, val in enumerate(self.sliced_data[i][j][1000:]): #in case first values are negative
                    if val <= 0:
                        self.sliced_data[i][j][k + 1 :] = 0
                        break
                if np.all(self.sliced_data[i][j][1:] > 0):
                    derivative = np.diff(self.sliced_data[i][j])
                    drop_detected = False
                    for l in range(len(derivative) - 1):
                        if not drop_detected and derivative[l] <= -0.01:
                            drop_detected = True
                        elif drop_detected and derivative[l + 1] >= 0:
                            self.sliced_data[i][j][l + 1 :] = 0
                            break

    def get_stimulation(self, time, stimulation_signal):
        """
        This function detects the stimulation and returns the time and index of the stimulation. It is based on the
        derivative's values of the stimulation signal.
        Parameters
        ----------
        time: array
            The time data
        stimulation_signal: array
            The stimulation signal

        Returns
        -------
        time_peaks: list
            The stimulation's time
        peaks: list
            The stimulation's indexes
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

        time_peaks = np.array(time_peaks) + self.avg_stim_time[self.frequency_stimulation]
        peaks = np.array(peaks) + int(self.avg_stim_time[self.frequency_stimulation] * self.frequency_acquisition)

        if isinstance(time_peaks, np.ndarray):
            time_peaks = time_peaks.tolist()
        if isinstance(peaks, np.ndarray):
            peaks = peaks.tolist()

        return time_peaks, peaks

    @staticmethod
    def _stimulation_detection_for_time_diff(time, stimulation_signal):
        """
        This function detects the stimulation time and returns the time and index of the stimulation. It is used to
        compute average time difference between the stimulation sent to acquisition and the one sent to the electrodes.
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
            stimulation_signal = -stimulation_signal # invert the signal if the first peak is negative
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

    def _get_avg_time_diff(self, c3d_path_stim_diff: str | list[str]):
        """
        This function calculates the average time difference between the stimulation sent to acquisition and the one
        sent to the electrodes.
        Parameters
        ----------
        c3d_path_stim_diff: str | list[str]
            The path to the c3d file(s) containing the stimulation signals sent to the acquisition and the electrodes.

        Returns
        -------
        avg_time_diff: float
            The average time difference between the stimulation time and the measured data
        """
        # Conversion into list
        c3d_path_list = [c3d_path_stim_diff] if isinstance(c3d_path_stim_diff, str) else c3d_path_stim_diff

        for i in range(len(c3d_path_list)):
            c3d_path = c3d_path_list[i]
            if not isinstance(c3d_path, str):
                raise TypeError("c3d_path must be a str or a list of str.")
            raw_data = Analogs.from_c3d(c3d_path)
            time = raw_data.time.values.tolist()

            time_peaks_muscle, peaks_muscle = self._stimulation_detection_for_time_diff(
                time=time, stimulation_signal=raw_data.values[0]
            )

            time_peaks_measured, peaks_measured = (
                self._stimulation_detection_for_time_diff(
                    time=time, stimulation_signal=raw_data.values[1]
                )
            )

            if len(time_peaks_measured) == len(time_peaks_muscle):
                time_diff = np.array(time_peaks_muscle) - np.array(time_peaks_measured)

                avg_time_diff = np.mean(time_diff)
            else:
                raise ValueError("Measured data and muscle data must have same frequency")

            return avg_time_diff

    @staticmethod
    def save_in_pkl(data, saving_pickle_path):
        """
        This function saves the given data in a pickle file.
        Parameters
        ----------
        data
            the data to save
        saving_pickle_path : str
            The path where the data will be saved as a pickle file.

        """
        with open(saving_pickle_path, "wb") as file:
            pickle.dump(data, file)

    def _calibration(self):
        """
        This function calibrates the data using the calibration matrix. If the calibration matrix is not provided, it
        checks if the data is already calibrated. If not, it raises an error.
        """
        if self.calibration_matrix_path is None and self.already_calibrated is False:
            raise ValueError("Please provide a calibration matrix path.")
        elif self.calibration_matrix_path is None and self.already_calibrated is True:
            self.filtered_6d_force = self.filtered_data[:6]
        else:
            self.calibration_matrix = self.read_text_file_to_matrix(self.calibration_matrix_path)
            self.filtered_6d_force = self.calibration_matrix @ self.filtered_data[:6]

    def _load_c3d(self, c3d_path):
        """
        This function loads the c3d file and extracts the analog data.
        Parameters
        ----------
        c3d_path: str
            file path to the c3d file
        """
        if not isinstance(c3d_path, str):
            raise TypeError("c3d_path must be a str or a list of str.")
        self.raw_data = Analogs.from_c3d(c3d_path)

    def get_data_at_handle(self):
        """
        This function provides the force data at the handle. It uses all the functions defined above.
        """
        # Getting data from c3d file
        self._load_c3d(self.c3d_path)

        self.time = self.raw_data.time.values.tolist()
        # Filtering data
        self.filtered_data = -np.array(self.raw_data.meca.low_pass(order=self.order, cutoff=self.cutoff,
                                                          freq=self.raw_data.rate)) if self.order and self.cutoff else self.raw_data
        # Reindexing raw_data
        lst_index = self.set_index(self.raw_data)
        raw_data_reindex = self.reindex_2d_list(self.raw_data.data, lst_index)
        self.filtered_data = self.reindex_2d_list(self.filtered_data, lst_index)

        self.torque_ergometer = self.filtered_data[6]

        # Calibrating data
        self._calibration()

        # Setting zero level
        self.filtered_6d_force = self.set_zero_level(self.filtered_6d_force, average_on=[0, 20000])
        self.stim_data = self.set_zero_level(raw_data_reindex[-1], average_on=[0, 20000])
        self.torque_ergometer = self.set_zero_level(self.torque_ergometer, average_on=[0, 20000])

        # Detect stimulation time
        self.stimulation_time, peaks = self.get_stimulation(
            time=self.time,
            stimulation_signal=self.stim_data,
        )
        # Add ergometer torque to data
        if isinstance(self.filtered_6d_force, list):
            self.filtered_6d_force += self.torque_ergometer
        else:
            self.filtered_6d_force = np.concatenate(
                (self.filtered_6d_force, np.array(self.torque_ergometer).reshape(1, -1)))

        # Slice the data from 6D file
        self.slice_data(time=self.time, data=self.filtered_6d_force, stimulation_index=peaks)

        # for j in range(len(self.sliced_time)):
        #    plt.plot(self.sliced_time[j], self.sliced_data[0][j])
        # plt.show()

        # Setting to zero each slice
        self.set_to_zero_slice()

        #for j in range(len(self.sliced_time)):
        #    plt.plot(self.sliced_time[j], self.sliced_data[0][j])
        #plt.show()

        self.handle_dictionary = {
            "time": self.sliced_time,
            "x": self.sliced_data[0],
            "y": self.sliced_data[1],
            "z": self.sliced_data[2],
            "mx": self.sliced_data[3],
            "my": self.sliced_data[4],
            "mz": self.sliced_data[5],
            "torque_ergometer": self.sliced_data[6],
            "stim_time": self.sliced_stim_time,
        }

    def load_model(self, elbow_angle: int | float):
        """
        This function is used to load the model and set the initial position, velocity and acceleration.
        Parameters
        ----------
        elbow_angle: int | float
            The elbow angle in degrees. It must be between 0 and 180 degrees.
            If elbow_angle is a list, the function will use the first element of the list.
        """
        # Load a predefined model
        self.model = Model(self.model_path)
        # Get number of q, qdot, qddot
        nq = self.model.nbQ()
        nqdot = self.model.nbQdot()
        nqddot = self.model.nbQddot()

        # Choose a position/velocity/acceleration to compute dynamics from
        if nq != 2:
            raise ValueError("The number of degrees of freedom has changed.")
        self.Q = np.array([0.0, np.radians(elbow_angle)])
        self.Qdot = np.zeros((nqdot,))  # speed null
        self.Qddot = np.zeros((nqddot,))  # acceleration null

    @staticmethod
    def local_sensor_to_local_hand(sensor_data: np.array) -> np.array:
        """
        This function is used to convert the sensor data from the local axis to the local hand axis.
        This a rotation along x-axis whatever the elbow angle
        Parameters
        ----------
        sensor_data: np.array
            The sensor data in the local axis

        Returns
        -------
        new_sensor_data: np.array
            The sensor data in the local hand axis
        """
        rotation_angle = np.pi / 2
        rotation_matrix = np.array(
            [
                [1, 0, 0],
                [0, np.cos(rotation_angle), -np.sin(rotation_angle)],
                [0, np.sin(rotation_angle), np.cos(rotation_angle)],
            ]
        )
        new_sensor_data = rotation_matrix @ sensor_data

        return new_sensor_data

    def select_muscle_and_dof(self):
        """
        This function is used to select the muscle and dof from the model.
        """
        dof_index = self.dof_name_index[self.dof_name]
        muscle_index = self.muscle_name_index[self.muscle_name]
        self.muscle_moment_arm = -self.model.musclesLengthJacobian(self.Q).to_array()[muscle_index][dof_index]

    def get_muscle_force(self, local_data):
        """
        This function is used to compute the muscle force from the local data.
        Parameters
        ----------
        local_data: array
            Contains the force and torque data in the local muscle axis

        """
        self.muscle_force_vector = []
        for i in range(len(local_data[0])):
            spatial_vector = np.array(
                [
                    local_data[0][i],
                    local_data[1][i],
                    local_data[2][i],
                    local_data[3][i],
                    local_data[4][i],
                    local_data[5][i],
                ]
            )

            external_force = self.model.externalForceSet()
            external_force.addInSegmentReferenceFrame(
                segmentName="r_ulna_radius_hand",
                vector=spatial_vector,
                pointOfApplication=np.array([0, 0, 0]),
            )

            tau = self.model.InverseDynamics(
                self.Q, self.Qdot, self.Qddot, external_force
            ).to_array()
            dof_index = self.dof_name_index[self.dof_name]
            tau = tau[dof_index]
            muscle_force = tau / self.muscle_moment_arm
            self.muscle_force_vector.append(muscle_force)

    def get_force(self, save: bool = False, plot: bool = True):
        """
        This function processes the data at the handle and computes the muscle force vector for each stimulation train.
        It uses all the functions defined above.
        Parameters
        ----------
        save: bool
            If True, the data will be saved in a pickle file.
        plot: bool
            If True, the data will be plotted.
        """
        self.get_data_at_handle()
        for i in range(len(self.handle_dictionary["x"])):
            force_data = np.array([self.handle_dictionary["x"][i], self.handle_dictionary["y"][i], self.handle_dictionary["z"][i]])
            torque_data = np.array([self.handle_dictionary["mx"][i], self.handle_dictionary["my"][i], self.handle_dictionary["mz"][i]])
            local_force_data = self.local_sensor_to_local_hand(force_data)
            local_torque_data = self.local_sensor_to_local_hand(torque_data)
            self.local_data = np.concatenate((local_force_data, local_torque_data))
            self.select_muscle_and_dof()
            self.get_muscle_force(local_data=self.local_data)

            self.muscle_force_vector = np.array(self.muscle_force_vector) - self.muscle_force_vector[0]
            self.muscle_force_vector_list.append(self.muscle_force_vector)

        self.saved_dictionary = {"force": self.muscle_force_vector_list, "time": self.handle_dictionary["time"], "stim_time": self.handle_dictionary["stim_time"], "muscle_name": self.muscle_name, "dof_name": self.dof_name}

        if save:
            self.save_in_pkl(self.saved_dictionary, self.saving_pickle_path)

        if plot:
            for i in range(len(self.muscle_force_vector_list)):
                plt.plot(self.handle_dictionary["time"][i], self.muscle_force_vector_list[i], color="blue")
                plt.scatter(self.handle_dictionary["stim_time"][i], [0] * len(self.handle_dictionary["stim_time"][i]), color="red", label="Stimulation")
            plt.title('Muscle Force and Stimulation')
            plt.show()


if __name__ == "__main__":
    force_converter = C3dToForce(
        c3d_path="/home/mickaelbegon/Documents/Stage_Florine/Data/P11/p11_force_20Hz_29.c3d",
        calibration_matrix_path="matrix.txt",
        saving_pickle_path="test.pkl",
        frequency_stimulation=20,
        rest_time=1,
        #model_path="C:\\Users\\flori_4ro0b8\\Documents\\Stage_S2M\\cocofest\\examples\\model_msk\\simplified_UL_Seth.bioMod",
        #elbow_angle=90,
        #muscle_name="BIC_long",
        #dof_name="r_ulna_radius_hand_r_elbow_flex_RotX",
        #transfer_force=True
    )
    force_converter.get_data_at_handle()
    data_stim = force_converter.filtered_6d_force

    plt.plot(data_stim[0], label="x", color="blue")
    plt.plot(data_stim[1], label="y", color="orange")
    plt.plot(data_stim[2], label="z", color="red")
    plt.legend()
    plt.show()

    handle_data = force_converter.handle_dictionary
    time = handle_data["time"]
    stim_time = handle_data["stim_time"]
    x = handle_data["x"]
    y = handle_data["y"]
    z = handle_data["z"]
    for i in range(len(x)):
        plt.plot(time[i], x[i], label="x", color="blue")
        plt.plot(time[i], y[i], label="y", color="orange")
        plt.plot(time[i], z[i], label="z", color="red")
        plt.scatter(stim_time[i], [0] * len(stim_time[i]), color="green", label="Stimulation")
    plt.legend()
    plt.show()



