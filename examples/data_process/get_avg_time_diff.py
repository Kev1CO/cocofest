import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from copy import deepcopy
import heapq
import pickle

from pyomeca import Analogs


class GetAvgTimeDifference:
    """
    Perfect data for identification is no force at the beginning and a force release between each stimulation train.
    This will enable data slicing of the force response to stimulation.

    It is assumed that V1, V2, V3, V4, V5, V6 are the 6D sensor data and last input is stimulation signal
    if not give an input list of the keys position "V1", "V2", "V3", "V4", "V5", "V6", "stim"
    in the c3d file (default is [0, 1, 2, 3, 4, 5, 6]).
    """

    def __init__(self, c3d_path: str | list[str] = None, plot_brut: bool = True):
        # Conversion into list
        c3d_path_list = [c3d_path] if isinstance(c3d_path, str) else c3d_path

        for i in range(len(c3d_path_list)):
            c3d_path = c3d_path_list[i]
            if not isinstance(c3d_path, str):
                raise TypeError("c3d_path must be a str or a list of str.")
            raw_data = Analogs.from_c3d(c3d_path)
            time = raw_data.time.values.tolist()

            if plot_brut:
                plt.plot(time[175000:185000], raw_data.values[0][175000:185000])
                plt.show()

            time_peaks_muscle, peaks_muscle = self.stimulation_detection(
                time=time, stimulation_signal=raw_data.values[0]
            )

            time_peaks_measured, peaks_measured = self.stimulation_detection(
                time=time, stimulation_signal=raw_data.values[1]
            )

            if len(time_peaks_measured) == len(time_peaks_muscle):
                time_diff = np.array(time_peaks_muscle) - np.array(time_peaks_measured)

                avg_time_diff = np.mean(time_diff)
            else:
                raise ValueError("Measured data and muscle data must have same frequency")

    def stimulation_detection(self, time, stimulation_signal):
        """

        Parameters
        ----------
        time
        stimulation_signal

        Returns
        -------
        time_peaks: list
            Contains peaks' time
        peaks: list
            Contains peaks' indexes
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


if __name__ == "__main__":
    GetAvgTimeDifference(
        c3d_path="C:\\Users\\flori_4ro0b8\\Documents\\Stage_S2M\\c3d_file\\exp_id\\stim_diff_50Hz.c3d", plot_brut=False
    )
