"""
This class process 6D force data from a hand sensor (in c3d file format) to determine the muscle force production.
- Caution: names of Vicon outputs can be different from file, modification available in default_index dictionary.
"""
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, savgol_filter
import heapq
import pickle

import ezc3d

from biorbd import Model

SEEDS_PULSE_WIDTH_PATH = Path(__file__).resolve().parent / "seeds_pulse_width.pkl"

# Rotation from the sensor frame to the forearm frame of r_ulna_radius_hand,
# estimated from the stimulation trains of the experiment.
SENSOR_TO_FOREARM = np.array(
    [
        [+0.007387, -0.241584, -0.970352],
        [-0.480144, +0.850342, -0.215361],
        [+0.877159, +0.467499, -0.109714],
    ]
)

class C3dToForce:
    def __init__(
        self,
        c3d_path=None,
        calibration_matrix_path=None,
        saving_pickle_path=None,
        model_path: str = None,
        muscle_name: str | list[str] = None,
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
            "Electric Current.Channel_5": 6,
        }

        self.raw_data = None
        self.channel_names = None
        self.rate = None
        self.filtered_data = None
        self.filtered_6d_force = None
        self.stim_data = None
        self.calibration_matrix = None
        self.sliced_time = None
        self.sliced_data = None
        self.time = None
        self.handle_dictionary = {}

        if c3d_path is None:
            raise ValueError("Please provide a c3d paths.")

        self.c3d_path = c3d_path

        self.saving_pickle_path = saving_pickle_path if saving_pickle_path else f"{c3d_path[:-3]}pkl"

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

            if "handle_distance" not in kwargs:
                raise ValueError(
                    "Please provide handle_distance (m), the distance from the elbow to the handle where the "
                    "sensor load is applied. It is the handle_elbow_dist column of data_participants.xlsx."
                )
            self.handle_distance = kwargs["handle_distance"]

            self.model_path = model_path
            self.local_data = None
            self.model = None
            self.Q = None
            self.Qdot = None
            self.Qddot = None
            self.muscle_name = muscle_name
            self.muscle_moment_arm = None
            self.muscle_force_vector = None
            self.muscle_force_vector_list = []
            self.saved_dictionary = {}

            # Load the model
            self.load_model(elbow_angle)

            # Saving muscle names and indexes as dict
            self.muscle_name_index = {}
            for i in range(len(self.model.muscleNames())):
                self.muscle_name_index[self.model.muscleNames()[i].to_string()] = i

            if self.muscle_name not in self.muscle_name_index.keys():
                raise ValueError(
                    f"Please provide a muscle name in the muscle_index dictionary : {list(self.muscle_name_index.keys())}."
                )

        with open(SEEDS_PULSE_WIDTH_PATH, 'rb') as f:
            seed_pw_data = pickle.load(f)
        file_seed = int(re.search('z_(.*).c3d', c3d_path).group(1))
        self.pulse_width = seed_pw_data[file_seed]

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
            with open(file_path, "r") as file:
                lines = file.readlines()
            data = []
            for line in lines:
                row = [float(value) for value in line.strip().split()]
                data.append(row)
            matrix = np.array(data)
            return matrix
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return None

    def reorder_data_by_channels(self, data_to_reorder):
        """
        Reorganize C3d file channels for data processing

        Parameters
        ----------
        data_to_reorder : array
            data to reorder.

        Returns
        -------
        array
            Reordered data.
        """
        channel_list = list(self.channel_names)
        indices = [channel_list.index(name) for name in self.default_index.keys() if name in channel_list]
        if len(indices) != len(self.default_index):
            missing = [name for name in self.default_index.keys() if name not in channel_list]
            raise ValueError(f"Missing channels in data : {missing}")

        return [data_to_reorder[i] for i in indices]

    @staticmethod
    def low_pass(data, order, cutoff, freq):
        """
        Zero-phase Butterworth low-pass filter, applied channel by channel.

        Parameters
        ----------
        data: np.ndarray
            The (n_channels, n_frames) data to filter
        order: int
            The order of the Butterworth filter
        cutoff: int | float
            The cut-off frequency (Hz)
        freq: int | float
            The sampling frequency of the data (Hz)

        Returns
        -------
        The filtered data
        """
        b, a = butter(N=order, Wn=cutoff, btype="low", output="ba", fs=freq)
        return filtfilt(b, a, np.asarray(data), axis=-1)

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

        self.sliced_time = []
        self.sliced_stim_time = []

        temp_stimulation_index = stimulation_index

        i = 0
        delta = self.frequency_acquisition / self.frequency_stimulation * 1.3

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
            self.sliced_time.append(time[first:last])
            self.sliced_stim_time.append(self.stimulation_time[first_stim:i])

            i += 1

            temp_stimulation_index = [peaks for peaks in temp_stimulation_index if peaks > last]

        self.sliced_data = [x, y, z, mx, my, mz]

    def set_to_zero_slice(self):
        """
        The end of the force production is now cut once, further down the chain,
        on the muscle force itself, where the decision is physically meaningful.
        """
        for i in range(len(self.sliced_data)):
            for j in range(len(self.sliced_data[i])):
                data = np.array(self.sliced_data[i][j])
                self.sliced_data[i][j] = data - data[0]

    def get_stimulation(self, time, stimulation_signal):
        """
        This function detects the stimulation and returns the time and index of the stimulation.
        It automatically handles negative or positive peaks based on the first significant peak.
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
        # Define thresholds for positive and negative peaks
        threshold_positive = np.mean(heapq.nlargest(200, stimulation_signal)) / 2
        threshold_negative = np.mean(heapq.nsmallest(200, stimulation_signal)) / 2

        # Find indices where signal exceeds thresholds
        positive_indices = np.where(stimulation_signal > threshold_positive)[0]
        negative_indices = np.where(stimulation_signal < threshold_negative)[0]

        # Determine polarity: if negative comes first, invert the signal
        if negative_indices.size > 0 and (positive_indices.size == 0 or negative_indices[0] < positive_indices[0]):
            stimulation_signal = -stimulation_signal
            threshold = -threshold_negative
        else:
            threshold = threshold_positive

        peaks = []
        i = 0
        min_rate = 0.5
        while i < len(stimulation_signal):
            if stimulation_signal[i] > threshold:
                start_i = i
                while i < len(stimulation_signal) and stimulation_signal[i] > threshold:
                    i += 1
                end_i = i - 1
                group_peaks = [start_i]
                group_time = self.time[end_i] - self.time[start_i]
                if group_time > 0:
                    rate = len(group_peaks) / group_time
                    if rate >= min_rate:
                        peaks.append(start_i - 1)
            else:
                i += 1

        # Convert peaks to time
        time_peaks = [time[peak] for peak in peaks]

        if isinstance(time_peaks, np.ndarray):
            time_peaks = time_peaks.tolist()
        if isinstance(peaks, np.ndarray):
            peaks = peaks.tolist()

        return time_peaks, peaks

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
        c3d = ezc3d.c3d(c3d_path)
        self.channel_names = c3d["parameters"]["ANALOG"]["LABELS"]["value"]
        self.raw_data = c3d["data"]["analogs"][0]
        self.rate = c3d["header"]["analogs"]["frame_rate"]
        first_frame = c3d["header"]["analogs"]["first_frame"]
        self.time = ((first_frame + np.arange(self.raw_data.shape[1])) / self.rate).tolist()

    def get_data_at_handle(self):
        """
        This function provides the force data at the handle. It uses all the functions defined above.
        """
        # Get data from c3d file
        self._load_c3d(self.c3d_path)

        # Filter data.
        self.filtered_data = (
            -self.low_pass(self.raw_data, order=self.order, cutoff=self.cutoff, freq=self.rate)
            if self.order and self.cutoff
            else np.array(self.raw_data)
        )

        # Smoothing
        for i in range(self.filtered_data.shape[0]):
            self.filtered_data[i] = savgol_filter(self.filtered_data[i], 5000, 1, deriv=0, delta=1.0, axis=-1,
                                                      mode='interp', cval=0.0)

        # Reindex raw_data
        raw_data_reindex = self.reorder_data_by_channels(self.raw_data)
        self.filtered_data = self.reorder_data_by_channels(self.filtered_data)

        # Calibrating data
        self._calibration()

        # Detect stimulation time
        self.stim_data = raw_data_reindex[-1]
        self.stimulation_time, self.peaks = self.get_stimulation(
            time=self.time,
            stimulation_signal=self.stim_data,
        )

        # Slice the data from 6D file
        self.slice_data(time=self.time, data=self.filtered_6d_force, stimulation_index=self.peaks)

        # Setting to zero each slice
        self.set_to_zero_slice()

        self.handle_dictionary = {
            "time": self.sliced_time,
            "x": self.sliced_data[0],
            "y": self.sliced_data[1],
            "z": self.sliced_data[2],
            "mx": self.sliced_data[3],
            "my": self.sliced_data[4],
            "mz": self.sliced_data[5],
            "stim_time": self.sliced_stim_time,
        }

    def load_model(self, elbow_angle: int | float):
        """
        This function is used to load the model and set the initial position, velocity and acceleration.
        Parameters
        ----------
        elbow_angle: int | float
            The elbow angle in degrees. It must be between 0 and 180 degrees.
        """
        self.model = Model(self.model_path)
        nq = self.model.nbQ()
        nqdot = self.model.nbQdot()
        nqddot = self.model.nbQddot()

        self.Q = np.array([np.radians(elbow_angle)])
        self.Qdot = np.zeros((nqdot,))
        self.Qddot = np.zeros((nqddot,))

    @staticmethod
    def local_sensor_to_local_hand(sensor_data: np.array) -> np.array:
        """
        Express sensor data (force or moment) in the forearm frame of r_ulna_radius_hand.
        """
        return SENSOR_TO_FOREARM @ sensor_data

    def select_muscle_and_dof(self):
        """
        This function is used to select the muscle and dof from the model.
        """
        muscle_index = self.muscle_name_index[self.muscle_name]
        self.muscle_moment_arm = self.model.musclesLengthJacobian(self.Q).to_array()[muscle_index][0]

    def get_muscle_force(self, local_data):
        """
        This function is used to compute the muscle force from the local data.

        Parameters
        ----------
        local_data: array
            Contains the force and torque data in the local muscle axis, as (fx, fy, fz, mx, my, mz)
        """
        point_of_application = np.array([0.0, -self.handle_distance, 0.0])

        self.muscle_force_vector = []
        for i in range(len(local_data[0])):
            force = np.array([local_data[0][i], local_data[1][i], local_data[2][i]])
            moment = np.array([local_data[3][i], local_data[4][i], local_data[5][i]])
            spatial_vector = np.concatenate((moment, force))

            external_force = self.model.externalForceSet()
            external_force.addInSegmentReferenceFrame(
                segmentName="r_ulna_radius_hand",
                vector=spatial_vector,
                pointOfApplication=point_of_application,
            )

            tau = self.model.InverseDynamics(
                self.Q, self.Qdot, self.Qddot, external_force
            ).to_array()[0]
            self.muscle_force_vector.append(-tau / self.muscle_moment_arm)

    @staticmethod
    def force_production_end(force, rest_fraction=0.02):
        """
        Index at which a train stops producing force: the first sample, after the force peak, where the force has
        decayed back to `rest_fraction` of that peak.

        Parameters
        ----------
        force: np.ndarray
            The muscle force of one train, already brought back to zero at its first sample
        rest_fraction: float
            The fraction of the peak under which the muscle is considered back at rest

        Returns
        -------
        int
            The index to cut at, or the full length when the force never decays back (a train to distrust)
        """
        if force.size == 0:
            return 0

        peak_index = int(np.argmax(force))
        peak = force[peak_index]
        if peak <= 0:
            return force.size

        decayed = np.where(force[peak_index:] <= rest_fraction * peak)[0]
        return force.size if decayed.size == 0 else peak_index + int(decayed[0]) + 1

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
        sliced_time, decayed_flags = [], []

        for i in range(len(self.handle_dictionary["x"])):
            force_data = np.array([self.handle_dictionary["x"][i], self.handle_dictionary["y"][i], self.handle_dictionary["z"][i]])
            torque_data = np.array([self.handle_dictionary["mx"][i], self.handle_dictionary["my"][i], self.handle_dictionary["mz"][i]])
            local_force_data = self.local_sensor_to_local_hand(force_data)
            local_torque_data = self.local_sensor_to_local_hand(torque_data)
            self.local_data = np.concatenate((local_force_data, local_torque_data))
            self.select_muscle_and_dof()
            self.muscle_force_vector = []
            self.get_muscle_force(local_data=self.local_data)
            self.muscle_force_vector = np.array(self.muscle_force_vector) - self.muscle_force_vector[0]

            # The muscle cannot pull negative, the small excursions below zero are drift and measurement noise
            self.muscle_force_vector = np.maximum(self.muscle_force_vector, 0)

            # Cut the train once the force production is over, and cut its time vector with it
            end = self.force_production_end(self.muscle_force_vector)
            decayed_flags.append(bool(end < self.muscle_force_vector.size))
            train_force = self.muscle_force_vector[:end].copy()

            if train_force.size:
                train_force[-1] = 0.0
            self.muscle_force_vector_list.append(train_force)
            sliced_time.append(np.asarray(self.handle_dictionary["time"][i])[:end])

        self.saved_dictionary = {"force": self.muscle_force_vector_list,
                                 "time": sliced_time,
                                 "frequency": self.frequency_stimulation,
                                 "pulse_width": self.pulse_width,
                                 "stim_time": self.handle_dictionary["stim_time"],
                                 "muscle_name": self.muscle_name,
                                 # False when the force never came back near zero, i.e. a train dominated by drift
                                 "force_decayed": decayed_flags}

        if save:
            self.save_in_pkl(self.saved_dictionary, self.saving_pickle_path)

        if plot:
            for i in range(len(self.muscle_force_vector_list)):
                plt.plot(self.handle_dictionary["time"][i], self.muscle_force_vector_list[i], color="blue")
                plt.scatter(self.handle_dictionary["stim_time"][i], [0] * len(self.handle_dictionary["stim_time"][i]), color="red", label="Stimulation")
            plt.title('Muscle Force and Stimulation')
            plt.show()
