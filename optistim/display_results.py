from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt


class MultiPlot:
    def __init__(self, file_path: str):

        all_files = [f for f in listdir(file_path) if isfile(join(file_path, f))]
        pkl_files = [i for i in all_files if i.endswith(".pkl")]
        if len(pkl_files) == 0:
            raise ValueError("There isn't any pickle file in this folder")

        

        multi_start_0 = pickle.load(file)



        fig, axs = plt.subplots(2)
        fig.suptitle('Vertically stacked subplots')
        axs[0].plot(x, y)
        axs[1].plot(x, -y)

class SinglePlot():