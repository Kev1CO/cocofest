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

from cocofest.identification.identification_method import DataExtraction

from examples.data_process.c3d_to_muscle_force import C3dToMuscleForce
from examples.data_process.muscle_force_from_pickle import MuscleForceFromPickle


def set_time_to_zero(stim_time, time_list):
    first_stim = stim_time[0]
    if first_stim > time_list[0]:
        raise ValueError("Time list should begin at the first stimulation")
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

    # stim_time_1 = stim_time + [2 * stim_time[-1]-stim_time[-2]]

    force_at_node = DataExtraction.force_at_node_in_ocp(tracked_data["time"], tracked_data["force"], n_shooting, final_time)

    #force_at_node = np.interp(stim_time, tracked_data["time"], tracked_data["force"]).tolist()

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


def main(plot=True, pickle_path=None, muscle_name=None):
    # Parameters for simulation and identification
    final_time = 4  #20
    pulse_width_values = [0.0004] * 50  #500

    # c3d_converter = C3dToMuscleForce()
    # norm_muscle_force, stim_time, time_list, stim_index_list = c3d_converter.get_force(
    #     c3d_path=c3d_path, calibration_matrix_path=calibration_matrix_path, saving_pickle_path=saving_pickle_path
    # )
    get_force = MuscleForceFromPickle(pickle_path=pickle_path, muscle_name=muscle_name)
    time, muscle_force, stim_time = get_force.read_pkl_to_force()
    stim_time, time = set_time_to_zero(stim_time[46:], time)

    tracked_data = {"time": time[:12500], "force": muscle_force[:12500]}

    last_stim = 0
    while stim_time[last_stim] < tracked_data["time"][-1]:
        last_stim += 1
    new_stim_time = stim_time[:last_stim]

    new_stim_time = list(np.linspace(0, 1, 50))
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
    main(plot=True, pickle_path="C:\\Users\\flori_4ro0b8\\Documents\\Stage_S2M\\cocofest\\examples\\data_process\\essai2_florine_force_biceps.pkl_0.pkl", muscle_name="BIC_long")
