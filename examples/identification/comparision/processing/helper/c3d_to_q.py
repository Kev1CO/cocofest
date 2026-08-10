"""
This class loads the motion c3d files of the experiment, computes the elbow angle from the shoulder/elbow/wrist
markers and slices it into one stimulation train per condition.

- Marker names can differ from one Vicon setup to another, they can be changed through the marker_names argument.
- The c3d files are read with ezc3d.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pickle

import ezc3d

ANALOG_SATURATION = 1e6


class C3dToQ:
    def __init__(
        self,
        c3d_path: str | list[str],
        marker_names: tuple[str, str, str] = ("should_r", "elbow_r", "wrist_r"),
        stim_channel: str = "Electric Current.Channel_5",
    ):
        """
        Parameters
        ----------
        c3d_path: str | list[str]
            The path(s) to the motion c3d file(s)
        marker_names: tuple[str, str, str]
            The shoulder, elbow and wrist marker names, in this order
        stim_channel: str
            The name of the analog channel holding the stimulator current
        """
        if isinstance(c3d_path, str):
            self.c3d_path = [c3d_path]
        elif isinstance(c3d_path, list):
            self.c3d_path = c3d_path
        else:
            raise ValueError("c3d_path must be a string or a list of strings.")

        self.marker_names = marker_names
        self.stim_channel = stim_channel

        self.data_dict: dict[str, np.ndarray] = {}
        self.time: np.ndarray | None = None
        self.data_stim: np.ndarray | None = None
        self.time_stim: np.ndarray | None = None
        self.Q_rad: np.ndarray | None = None

        self.frequency_acquisition: int = 100  # Hz, markers
        self.frequency_acquisition_stim: int = 10000  # Hz, analogs
        self.frequency_stimulation: int = 50  # Hz
        # Delay between the stimulator trigger seen on the analog channel and the actual pulse, per frequency
        self.average_time_difference: dict = {
            20: 0.0008799999999999671,
            33: 0.0008608695652174252,
            50: 0.0008739549839228076,
        }

    # ---- Reading the c3d ---- #
    @staticmethod
    def _find_label(labels: list, wanted: str, kind: str) -> int:
        """
        Index of `wanted` in `labels`.
        """
        for i, label in enumerate(labels):
            if label == wanted or label.split(":")[-1] == wanted:
                return i
        raise ValueError(f"{kind} '{wanted}' not found, the file only holds {labels}.")

    def load_c3d(self):
        """
        Load the c3d file(s), extract the marker and stimulation data and drop the corrupted tail, if any.
        """
        if self.time is not None:  # Already loaded
            return

        markers, analogs, time, time_stim = [], [], [], []
        for path in self.c3d_path:
            c3d = ezc3d.c3d(str(path))

            point_rate = c3d["header"]["points"]["frame_rate"]
            analog_rate = c3d["header"]["analogs"]["frame_rate"]
            self.frequency_acquisition = int(point_rate)
            self.frequency_acquisition_stim = int(analog_rate)

            point_labels = c3d["parameters"]["POINT"]["LABELS"]["value"]
            analog_labels = c3d["parameters"]["ANALOG"]["LABELS"]["value"]

            points = c3d["data"]["points"][:3]  # x, y, z
            stim = c3d["data"]["analogs"][0, self._find_label(analog_labels, self.stim_channel, "Analog channel")]

            marker_data = np.array(
                [points[:, self._find_label(point_labels, name, "Marker"), :] for name in self.marker_names]
            )

            n_points, n_analogs = self._first_corrupted_sample(marker_data, stim, point_rate, analog_rate)
            marker_data = marker_data[:, :, :n_points]
            stim = stim[:n_analogs]

            first_point_frame = c3d["header"]["points"]["first_frame"]
            first_analog_frame = c3d["header"]["analogs"]["first_frame"]

            markers.append(marker_data)
            analogs.append(stim)
            time.append((first_point_frame + np.arange(n_points)) / point_rate)
            time_stim.append((first_analog_frame + np.arange(n_analogs)) / analog_rate)

        self.data_dict = {name: np.concatenate([m[i] for m in markers], axis=1) for i, name in enumerate(self.marker_names)}
        self.data_stim = np.concatenate(analogs)
        self.time = np.concatenate(time)
        self.time_stim = np.concatenate(time_stim)

    @staticmethod
    def _first_corrupted_sample(marker_data, stim, point_rate, analog_rate):
        """
        Return the number of marker frames and analog samples to keep, i.e. everything before the first occluded
        marker or saturated analog sample.
        """
        n_points, n_analogs = marker_data.shape[2], stim.shape[0]

        occluded = np.where(np.isnan(marker_data).any(axis=(0, 1)))[0]
        saturated = np.where(~np.isfinite(stim) | (np.abs(stim) > ANALOG_SATURATION))[0]

        if occluded.size:
            n_points = min(n_points, int(occluded[0]))
        if saturated.size:
            n_analogs = min(n_analogs, int(saturated[0]))

        # Keep both signals consistent with one another
        n_points = min(n_points, int(n_analogs * point_rate / analog_rate))
        n_analogs = min(n_analogs, int(n_points * analog_rate / point_rate))

        return n_points, n_analogs

    # ---- Elbow angle ---- #
    @staticmethod
    def _get_segment_vector(start, end):
        """Calculate the vector from start to end points."""
        return np.array(end) - np.array(start)

    @staticmethod
    def _get_angle(u, v):
        """
        Calculate the angle between two vectors, frame by frame, in radians.

        Parameters
        ----------
        u, v: np.ndarray
            The two (3, n_frames) vectors to compute the angle in between

        Returns
        -------
        The angle at every frame, in radians
        """
        dot_product = np.einsum("ij,ij->j", u, v)
        cos_theta = dot_product / (np.linalg.norm(u, axis=0) * np.linalg.norm(v, axis=0))
        return np.arccos(np.clip(cos_theta, -1.0, 1.0))

    def _get_q(self):
        """
        Compute the elbow angle in radians from the shoulder, elbow and wrist markers. The angle is expressed as a
        flexion angle: 0 rad is a fully extended elbow.
        """
        if self.Q_rad is not None:
            return self.Q_rad

        self.load_c3d()
        shoulder, elbow, wrist = self.marker_names
        forearm_position = self._get_segment_vector(start=self.data_dict[elbow], end=self.data_dict[wrist])
        humerus_position = self._get_segment_vector(start=self.data_dict[elbow], end=self.data_dict[shoulder])
        self.Q_rad = np.pi - self._get_angle(forearm_position, humerus_position)

        return self.Q_rad

    def get_q_rad(self):
        """The elbow angle in radians."""
        return self._get_q()

    def get_q_deg(self):
        """The elbow angle in degrees."""
        return np.rad2deg(self._get_q())

    def get_time(self):
        """The time vector matching the elbow angle."""
        self.load_c3d()
        return self.time

    # ---- Stimulation detection and slicing ---- #
    def _get_stimulation(self, time, stimulation_signal):
        """
        Detect the stimulation pulses from the derivative of the stimulator current.

        Parameters
        ----------
        time: np.ndarray
            The analog time vector
        stimulation_signal: np.ndarray
            The stimulator current

        Returns
        -------
        tuple
            The stimulation times and the matching analog indices
        """
        derivative = np.diff(stimulation_signal)

        threshold_positive = np.mean(np.sort(stimulation_signal)[-200:]) / 2
        threshold_negative = np.mean(np.sort(stimulation_signal)[:200]) / 2

        positive = np.where(stimulation_signal > threshold_positive)[0]
        negative = np.where(stimulation_signal < threshold_negative)[0]

        if negative.size and (not positive.size or negative[0] < positive[0]):
            derivative = -derivative

        derivative_threshold = np.mean(np.sort(derivative)[-200:]) / 2
        above_threshold = np.where(derivative > derivative_threshold)[0]
        if above_threshold.size == 0:
            raise RuntimeError(f"No stimulation could be detected on the '{self.stim_channel}' channel.")

        peaks = [above_threshold[0]]
        for index in above_threshold[1:]:
            if index - peaks[-1] > 10:
                peaks.append(index)

        time_difference = self.average_time_difference[self.frequency_stimulation]
        time_peaks = (np.array([time[peak] for peak in peaks]) + time_difference).tolist()
        peaks = (np.array(peaks) + int(time_difference * self.frequency_acquisition_stim)).tolist()

        return time_peaks, peaks

    def slice_data(self, data):
        """
        Slice the data into stimulation trains. The trains are detected by comparing the gap between two consecutive
        stimulation indices: inside a train the gap is the stimulation period, between two trains it is much larger.

        Parameters
        ----------
        data: np.ndarray
            The marker-rate data to slice (typically the elbow angle)

        Returns
        -------
        tuple
            The sliced time, the sliced data and the sliced stimulation times
        """
        self.load_c3d()
        stimulation_time, peaks_index = self._get_stimulation(self.time_stim, self.data_stim)

        sliced_time, sliced_data, sliced_stim_time = [], [], []

        temp_peaks_index = peaks_index
        i = 0
        delta = self.frequency_acquisition_stim / self.frequency_stimulation * 1.3

        while len(temp_peaks_index) != 0 and i < len(peaks_index) - 1:
            first = peaks_index[i]
            first_stim = i
            while i + 1 < len(peaks_index) and peaks_index[i + 1] - peaks_index[i] < delta:
                i += 1

            last = -1 if i + 1 >= len(peaks_index) else peaks_index[i + 1] - 1

            first = int(first * self.frequency_acquisition / self.frequency_acquisition_stim)
            last = (
                first + 2 * self.frequency_acquisition
                if last == -1
                else int(last * self.frequency_acquisition / self.frequency_acquisition_stim) + 1
            )
            last = min(last, len(self.time))

            sliced_time.append(self.time[first:last])
            sliced_data.append(data[first:last])
            sliced_stim_time.append(stimulation_time[first_stim:i])

            i += 1

            temp_peaks_index = [peaks for peaks in temp_peaks_index if peaks > last]

        return sliced_time, sliced_data, sliced_stim_time

    @staticmethod
    def _set_time_continuity(sliced_stim_time, sliced_time):
        """
        Make the sliced time vectors continuous: the first slice starts at 0 and every following slice starts where
        the previous one ended. The stimulation times are shifted accordingly.
        """
        sliced_stim_time[0] = np.array(sliced_stim_time[0]) - sliced_time[0][0]
        sliced_time[0] = np.array(sliced_time[0]) - sliced_time[0][0]

        for i in range(len(sliced_time) - 1):
            offset = sliced_time[i + 1][0] - sliced_time[i][-1]
            sliced_stim_time[i + 1] = np.array(sliced_stim_time[i + 1]) - offset
            sliced_time[i + 1] = np.array(sliced_time[i + 1]) - offset

        return sliced_time, sliced_stim_time

    def get_sliced_time_Q_rad(self):
        """The elbow angle in radians, its time vector and the stimulation times, sliced per stimulation train."""
        sliced_time, sliced_data, sliced_stim_time = self.slice_data(self._get_q())
        sliced_time, sliced_stim_time = self._set_time_continuity(sliced_stim_time, sliced_time)
        return {"q": sliced_data, "time": sliced_time, "stim_time": sliced_stim_time}

    def get_sliced_time_Q_deg(self):
        """The elbow angle in degrees, its time vector and the stimulation times, sliced per stimulation train."""
        sliced_time, sliced_data, sliced_stim_time = self.slice_data(np.rad2deg(self._get_q()))
        sliced_time, sliced_stim_time = self._set_time_continuity(sliced_stim_time, sliced_time)
        return {"q": sliced_data, "time": sliced_time, "stim_time": sliced_stim_time}

    def get_sliced_stim_time(self):
        """Same as get_sliced_time_Q_rad, but keeping the original (non-continuous) time base."""
        sliced_time, sliced_data, sliced_stim_time = self.slice_data(self._get_q())
        return {"q": sliced_data, "time": sliced_time, "stim_time": sliced_stim_time}

    @staticmethod
    def save_in_pkl(data, pkl_path):
        """Save the given data in one or several pickle files."""
        if isinstance(pkl_path, str):
            pkl_path = [pkl_path]
        elif not isinstance(pkl_path, list):
            raise ValueError("pkl_path must be a string or a list of strings.")

        for path in pkl_path:
            with open(path, "wb") as file:
                pickle.dump(data, file)


if __name__ == "__main__":
    c3d_path = (
        Path(__file__).resolve().parent.parent.parent / "data" / "exp" / "P01" / "p01_motion_20Hz_83.c3d"
    )
    c3d_to_q = C3dToQ(str(c3d_path))
    c3d_to_q.frequency_stimulation = 20

    plt.plot(c3d_to_q.get_time(), c3d_to_q.get_q_deg())
    plt.xlabel("Time (s)")
    plt.ylabel("Elbow flexion (deg)")
    plt.show()

    sliced = c3d_to_q.get_sliced_time_Q_rad()
    for i in range(len(sliced["q"])):
        plt.plot(sliced["time"][i], sliced["q"][i])
        plt.scatter(sliced["stim_time"][i], [0] * len(sliced["stim_time"][i]), color="red")
    plt.xlabel("Time (s)")
    plt.ylabel("Elbow flexion (rad)")
    plt.show()
