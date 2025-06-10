"""
This example demonstrates the way of identifying an experimental muscle force model based on Ding 2007 model.
"""

import matplotlib.pyplot as plt
import numpy as np

from bioptim import SolutionMerge, OdeSolver, OptimalControlProgram, ObjectiveFcn, Node, ControlType, ObjectiveList

from cocofest import (
    DingModelPulseWidthFrequency,
    IvpFes,
    ModelMaker,
    OcpFesId, FES_plot,
)
import pickle
from cocofest.identification.identification_method import DataExtraction

from examples.data_process.c3d_to_force import C3dToForce


def set_time_to_zero(stim_time, time_list):
    first_stim = stim_time[0]
    if first_stim > time_list[0]:
        pass
        #raise ValueError("Time list should begin at the first stimulation")
    stim_time = list(np.array(stim_time) - first_stim)
    time_list = list(np.array(time_list) - first_stim)

    return stim_time, time_list

def prepare_ocp(
    model,
    final_time,
    pulse_width_values,
    key_parameter_to_identify,
    tracked_data,
    stim_time
):
    n_shooting = model.get_n_shooting(final_time)

    force_at_node = DataExtraction.force_at_node_in_ocp(tracked_data["time"], tracked_data["force"], n_shooting, final_time)

    numerical_data_time_series, stim_idx_at_node_list = model.get_numerical_data_time_series(n_shooting, final_time)
    dynamics = OcpFesId.declare_dynamics(model=model, numerical_data_timeseries=numerical_data_time_series)

    x_bounds, x_init = OcpFesId.set_x_bounds(
        model=model,
        force_tracking=force_at_node,
    )
    u_bounds, u_init = OcpFesId.set_u_bounds(
        model=model,
        control_value=pulse_width_values,
        stim_idx_at_node_list=stim_idx_at_node_list,
        n_shooting=n_shooting,
    )

    objective_functions = ObjectiveList()
    objective_functions.add(
        ObjectiveFcn.Lagrange.TRACK_STATE,
        key="F",
        weight=1,
        target=np.array(force_at_node)[np.newaxis, :],
        node=Node.ALL,
        quadratic=True,
    )
    additional_key_settings = OcpFesId.set_default_values(model)

    parameters, parameters_bounds, parameters_init = OcpFesId.set_parameters(
        parameter_to_identify=key_parameter_to_identify,
        parameter_setting=additional_key_settings,
        use_sx=True,
    )
    OcpFesId.update_model_param(model, parameters)

    return OptimalControlProgram(
        bio_model=[model],
        dynamics=dynamics,
        n_shooting=n_shooting,
        phase_time=final_time,
        x_init=x_init,
        x_bounds=x_bounds,
        u_init=u_init,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        parameters=parameters,
        parameter_bounds=parameters_bounds,
        parameter_init=parameters_init,
        control_type=ControlType.CONSTANT,
        use_sx=True,
        ode_solver=OdeSolver.RK4(n_integration_steps=10),
        n_threads=20,
    )


def main(plot=True):
    # Parameters for simulation and identification
    final_time = 2
    with open("C:\\Users\\flori_4ro0b8\\Documents\\Stage_S2M\\cocofest\\examples\\data_process\\seeds_pulse_width.pkl", "rb") as f:
        pulse_width_values_list = pickle.load(f)
    pulse_width = pulse_width_values_list[59][0]
    pulse_width_values = [pulse_width] * 50

    c3d_converter = C3dToForce(
        c3d_path="C:\\Users\\flori_4ro0b8\\Documents\\Stage_S2M\\CollecteDeDonnees\\Data\\P03\\p03_force_50Hz_59.c3d",
        calibration_matrix_path="C:\\Users\\flori_4ro0b8\\Documents\\Stage_S2M\\cocofest\\examples\\data_process\\matrix.txt",
        saving_pickle_path="p03_force_50Hz_59.pkl",
        frequency_acquisition=10000,
        frequency_stimulation=50,
        rest_time=1,
    )
    c3d_converter.get_data_at_handle()
    data = c3d_converter.handle_dictionary
    time = data["time"][0]
    muscle_force = data["x"][0]
    stim_time = data["stim_time"][0]

    stim_time, time = set_time_to_zero(stim_time, time)

    tracked_data = {"time": time, "force": muscle_force}

    #new_stim_time = list(np.linspace(0, 1, 50))
    new_stim_time = list(np.round(stim_time, 2))
    model = ModelMaker.create_model("ding2007", stim_time=new_stim_time, sum_stim_truncation=10)

    ocp = prepare_ocp(
        model,
        final_time,
        pulse_width_values,
        tracked_data=tracked_data,
        stim_time=new_stim_time,
        key_parameter_to_identify=[
            "km_rest",
            "tau1_rest",
            "tau2",
            "pd0",
            "pdt",
            "a_scale",
        ],
    )
    sol = ocp.solve()

    if plot:
        default_model = DingModelPulseWidthFrequency()

        FES_plot(data=sol).plot(
            title="Identification of experimental parameters based on Ding 2007 model",
            tracked_data=tracked_data,
            default_model=default_model,
            show_bounds=False,
            show_stim=True,
            stim_time=new_stim_time
        )


if __name__ == "__main__":
    main(plot=True)
