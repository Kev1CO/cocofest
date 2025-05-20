import matplotlib.pyplot as plt
from examples.data_process.c3d_to_q import C3dToQ
from examples.data_process.c3d_to_force import C3dToForce


def force_process(c3d_path: str|list[str], saving_pkl_path: str|list[str] = None, save: bool = True, plot: bool = False):
    c3d_path_list = [c3d_path] if isinstance(c3d_path, str) else c3d_path
    saving_pkl_list = [saving_pkl_path] if isinstance(saving_pkl_path, str) else saving_pkl_path
    if len(c3d_path_list) != len(saving_pkl_list) and save:
        raise ValueError("c3d_path and saving_pkl_path must have the same length to save")

    for i in range(len(c3d_path_list)):
        c3d_to_force = C3dToForce(
            c3d_path=c3d_path_list[i],
            calibration_matrix_path="matrix.txt",
            saving_pickle_path=saving_pkl_list[i],
            frequency_acquisition=10000,
            frequency_stimulation=50,
            rest_time=1,
            model_path="C:\\Users\\flori_4ro0b8\\Documents\\Stage_S2M\\cocofest\\examples\\model_msk\\simplified_UL_Seth.bioMod",
            elbow_angle=90,
            muscle_name="BIC_long",
            dof_name="r_ulna_radius_hand_r_elbow_flex_RotX",
        )
        c3d_to_force.get_data_at_handle()
        c3d_to_force.get_force(save=save, plot=plot)


def q_process(c3d_path: str|list[str], saving_pkl_path: str|list[str] = None, save: bool = True, plot: bool = False):
    c3d_path_list = [c3d_path] if isinstance(c3d_path, str) else c3d_path
    saving_pkl_list = [saving_pkl_path] if isinstance(saving_pkl_path, str) else saving_pkl_path
    if save and len(c3d_path_list) != len(saving_pkl_list):
        raise ValueError("c3d_path and saving_pkl_path must have the same length")

    for i in range(len(c3d_path_list)):
        c3d_to_q = C3dToQ(c3d_path_list[i])
        dict = c3d_to_q.get_sliced_time_Q_rad()
        if save:
            c3d_to_q.save_to_pkl(dict, saving_pkl_list[i])
        if plot:
            for j in range(len(dict["time"])):
                plt.plot(dict["time"][j], dict["q"][j], color="blue")
                plt.scatter(dict["stim_time"][j], [0] * len(dict["stim_time"][j]), color="red")
            plt.title("Q (rad) and Stimulations")
            plt.show()


if __name__ == "__main__":
    q_process(
        c3d_path="C:\\Users\\flori_4ro0b8\\Documents\\Stage_S2M\\c3d_file\\essais_mvt_16.05.25\\lucie_50Hz_250-300-350-400-450x2_21mA_doublet.c3d",
        save=False,
        plot=True,
    )