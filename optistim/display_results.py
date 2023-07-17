from os import listdir
from os.path import isfile, join
import pickle

from math import ceil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patches





class MultiPlot:
    def __init__(self, file_path: str, max_fig_on_plot: int = 25, save_fig: bool = False):

        all_files = [f for f in listdir(file_path) if isfile(join(file_path, f))]
        pkl_files = [i for i in all_files if i.endswith(".pkl")]
        if len(pkl_files) == 0:
            raise ValueError("There isn't any pickle file in this folder")

        nb_plots = ceil(len(pkl_files) / max_fig_on_plot)
        i = 0

        for plots in range(nb_plots):
            if max_fig_on_plot == 1:
                x, y = 1, 1
            elif max_fig_on_plot == 2:
                x, y = 1, 2
            elif max_fig_on_plot <= 4:
                x, y = 2, 2
            elif max_fig_on_plot <= 6:
                x, y = 2, 3
            elif max_fig_on_plot <= 9:
                x, y = 3, 3
            elif max_fig_on_plot <= 12:
                x, y = 3, 4
            elif max_fig_on_plot <= 16:
                x, y = 4, 4
            elif max_fig_on_plot <= 20:
                x, y = 4, 5
            elif max_fig_on_plot <= 25:
                x, y = 5, 5
            else:
                raise ValueError("Can only stack 25 or less fig on same plot")

            fig, axs = plt.subplots(x, y, figsize=(19.2, 9.8))
            fig_num = 0

            for ax in axs.flat:
                with open(file_path + "/" + pkl_files[i], 'rb') as pickle_file:
                    data = pickle.load(pickle_file)

                ax.plot(data["time"], data["F"][0], label='Force')
                stim_index = [np.where(data["time"] == data['phase_time'][i]) for i in range(len(data['phase_time']))]
                force_at_stim = np.array([data["F"][0][i][0] for i in stim_index])
                ax.scatter(data["phase_time"], force_at_stim, label='Stimulation', color='gold')
                if data["force_tracking"] is not None:
                    ax.plot(data["force_tracking"][0], data["force_tracking"][1], label='Target', color='red')
                if data["end_node_tracking"] is not None:
                    ax.scatter(data["time"][-1], data["end_node_tracking"], label='Target', color='red')

                ax.set(xlabel='Time (s)', ylabel='Force (N)')

                # create the corresponding number of labels (= the text you want to display)
                labels = []

                if data['model'][0] == 'DingModelPulseDurationFrequency':
                    time_param = "fixed time" if data['time_min'] is None else "optimizing time"
                    pulse_param = "fixed pulse duration" if data['pulse_time_bimapping'] is None or False else "optimizing pulse duration"
                    optimization_param = "(" + time_param + " & " + pulse_param + ")"
                elif data['model'][0] == 'DingModelIntensityFrequency':
                    time_param = "fixed time" if data['time_min'] is None else "optimizing time"
                    pulse_param = "fixed pulse intensity" if data['pulse_intensity_bimapping'] is None or False else "optimizing pulse intensity"
                    optimization_param = "(" + time_param + " & " + pulse_param + ")"
                else:
                    optimization_param = "(fixed time)" if data['time_min'] is None else "(optimizing time)"

                labels.append("{0} {1}".format(str(data['model'][0]),
                                               optimization_param))

                labels.append("optimal solution found = {0}".format(True if data['status'] == 0 else False))
                labels.append("cost = {0:.4g}".format(float(data['cost'])))
                labels.append("nb stim = {0:.4g}".format(int(data['n_stim'])))

                if data['time_bimapping'][0]:
                    labels.append("frequency = {0:.4g}Hz".format(round(1 / data['phase_time'][1])))
                else:
                    for j in range(len(data["phase_time"])):
                        ax.annotate(text=str(round(data["phase_time"][j], 2)),
                                    xy=(data["phase_time"][j], force_at_stim[j]))
                if data['model'][0] == 'DingModelPulseDurationFrequency' and data['pulse_time_bimapping'][0]:
                    labels.append("pulse duration = {0:.4g}s".format(round(data['parameters']['pulse_duration'][0], 5)))
                elif data['model'][0] == 'DingModelPulseDurationFrequency':
                    for j in range(len(data["phase_time"])):
                        ax.annotate(text=str(round(data['parameters']['pulse_duration'][j] * 10e6, 0) + "us"),
                                    xy=(data["phase_time"][j], force_at_stim[j]+max(data["F"][0])*0.001))
                if data['model'][0] == 'DingModelIntensityFrequency' and data['pulse_intensity_bimapping'][0]:
                    labels.append("pulse intensity = {0:.4g}mA".format(round(data['parameters']['pulse_intensity'][0])))
                elif data['model'][0] == 'DingModelIntensityFrequency':
                    for j in range(len(data["phase_time"])):
                        ax.annotate(text=str(round(data['parameters']['pulse_intensity'][j], 5) + "mA"),
                                    xy=(data["phase_time"][j], force_at_stim[j]+max(data["F"][0])*0.001))

                # create a list with two empty handles (or more if needed)
                handles = [mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white",
                                                 lw=0, alpha=0)] * 10

                if data['time_min']:
                    labels.append("{0} {1}-{2}".format("stim interval min-max (s):",
                                                   data['time_min'][0], data['time_max'][0]))
                if data['model'][0] == 'DingModelPulseDurationFrequency' and data['pulse_time_min']:
                    labels.append("{0} {1}-{2}".format("stim duration min-max (s):",
                                                       data['pulse_time_min'][0], data['pulse_time_min'][0]))
                if data['model'][0] == 'DingModelIntensityFrequency' and data['pulse_intensity_min']:
                    labels.append("{0} {1}-{2}".format("stim intensity min-max (mA):",
                                                       data['pulse_intensity_min'][0], data['pulse_intensity_min'][0]))

                # create the legend, supressing the blank space of the empty line symbol and the
                # padding between symbol and label by setting handlelenght and handletextpad
                ax.legend(handles, labels, loc='best', fontsize='small',
                          fancybox=True, framealpha=0.7,
                          handlelength=0, handletextpad=0)

                i += 1
                if i == len(pkl_files):
                    for blank_fig in range(1, len(axs.flat) - fig_num):
                        axs.flat[-blank_fig].set_visible(False)
                    if save_fig:
                        image_name = 'subplot' + str(plots + 1) + '.svg'
                        fig.savefig(image_name, format='svg', dpi=1200)
                    plt.show()
                    exit()

                fig_num += 1
            # Hide x labels and tick labels for top plots and y ticks for right plots.
            for ax in axs.flat:
                ax.label_outer()

            if i == len(pkl_files):
                if save_fig:
                    image_name = 'subplot' + str(plots + 1) + '.svg'
                    fig.savefig(image_name, format='svg', dpi=1200)
                plt.show()

            else:
                if save_fig:
                    image_name = 'subplot' + str(plots + 1) + '.svg'
                    fig.savefig(image_name, format='svg', dpi=1200)
                plt.show(block=False)


if __name__ == "__main__":
    MultiPlot(file_path="../examples/temporary", max_fig_on_plot=4, save_fig=False)
