"""
This example will do a 33 stimulation example with Ding's 2007 frequency model.
This ocp was build to produce an elbow motion from 5 to 120 degrees.
The stimulation frequency is fixed at 33hz and the pulse width is optimized to satisfy the flexion while minimizing
elbow residual torque control.
"""

import numpy as np

from bioptim import (
    Solver,
    OdeSolver,
    ObjectiveList,
    ObjectiveFcn,
    OptimalControlProgram,
    ControlType,
    Node,
    ConstraintList,
    ConstraintFcn
)
from cocofest import DingModelPulseWidthFrequency, OcpFesMsk, FesMskModel, CustomConstraint, OcpFesIdMultibody


def prepare_ocp(model: FesMskModel, final_time: float, key_parameter_to_identify, msk_info: dict):
    muscle_model = model.muscles_dynamics_model[0]
    n_shooting = muscle_model.get_n_shooting(final_time)

    numerical_time_series, stim_idx_at_node_list = muscle_model.get_numerical_data_time_series(
        n_shooting, final_time
    )

    dynamics = OcpFesMsk.declare_dynamics(model, numerical_time_series=numerical_time_series, with_contact=False)

    x_bounds, x_init = OcpFesMsk.set_x_bounds(model, msk_info)
    u_bounds, u_init = OcpFesMsk.set_u_bounds(model, msk_info["with_residual_torque"], max_bound=0.0006) #TODO

    simulation = np.linspace(0, 3, 33)

    objective_functions = ObjectiveList()
    objective_functions.add(
        ObjectiveFcn.Lagrange.TRACK_STATE,
        key="q",
        target=simulation,
        node=Node.ALL,
        weight=1,
        quadratic=True,
    )

    constraints = ConstraintList()
    constraints.add(CustomConstraint.muscles_param_constraint, key="tau1_rest", muscle=["BIC_long", "BIC_brevis"])
    constraints.add(CustomConstraint.muscles_param_constraint, key="tau2", muscle=["BIC_long", "BIC_brevis"])
    constraints.add(CustomConstraint.muscles_param_constraint, key="pd0", muscle=["BIC_long", "BIC_brevis"])
    constraints.add(CustomConstraint.muscles_param_constraint, key="pdt", muscle=["BIC_long", "BIC_brevis"])
    constraints.add(CustomConstraint.muscles_param_constraint, key="a_scale", muscle=["BIC_long", "BIC_brevis"])
    constraints.add(CustomConstraint.muscles_param_constraint, key="km_rest", muscle=["BIC_long", "BIC_brevis"])

    ocp_fed_id_multibody = OcpFesIdMultibody()
    #additional_key_settings = ocp_fed_id_multibody.set_default_values(msk_model=model)
    additional_key_settings = OcpFesIdMultibody.set_default_values(msk_model=model)

    parameters, parameters_bounds, parameters_init = ocp_fed_id_multibody.set_parameters(
        parameter_to_identify=key_parameter_to_identify,
        parameter_setting=additional_key_settings,
        use_sx=False,  # error when use_sx=True because BioRbdModel initializes in mx
    )

    model = OcpFesMsk.update_model_param(model, parameters=parameters)

    return OptimalControlProgram(
        bio_model=[model],
        dynamics=dynamics,
        n_shooting=n_shooting,
        phase_time=final_time,
        objective_functions=objective_functions,
        x_init=x_init,
        x_bounds=x_bounds,
        u_init=u_init,
        u_bounds=u_bounds,
        parameters=parameters,
        parameter_bounds=parameters_bounds,
        parameter_init=parameters_init,
        control_type=ControlType.CONSTANT,
        use_sx=False,
        ode_solver=OdeSolver.RK4(n_integration_steps=10),
        n_threads=20,
        constraints=constraints,
    )


def main(plot=True, biorbd_path="C:\\Users\\flori_4ro0b8\\Documents\\Stage_S2M\\cocofest\\examples\\model_msk\\simplified_UL_Seth_BIC_elbow.bioMod"):
    simulation_ending_time = 1
    long_biceps = DingModelPulseWidthFrequency(muscle_name="BIC_long", sum_stim_truncation=10)
    brevis_biceps = DingModelPulseWidthFrequency(muscle_name="BIC_brevis", sum_stim_truncation=10)

    model = FesMskModel(
        name=None,
        biorbd_path=biorbd_path,
        muscles_model=[long_biceps, brevis_biceps],
        stim_time=list(np.linspace(0, simulation_ending_time, 34)[:-1]),
        activate_force_length_relationship=True,
        activate_force_velocity_relationship=True,
        activate_passive_force_relationship=True,
        activate_residual_torque=False,
        external_force_set=None,  # External forces will be added later
    )

    msk_info = {
        "with_residual_torque": False,
        "bound_type": "start_end",
        "bound_data": [[0], [np.degrees(3)]],
    }

    ocp = prepare_ocp(
        model=model,
        final_time=simulation_ending_time,
        msk_info=msk_info,
        key_parameter_to_identify=[
            "km_rest_BIC_long",
            "tau1_rest_BIC_long",
            "tau2_BIC_long",
            "pd0_BIC_long",
            "pdt_BIC_long",
            "a_scale_BIC_long",
            "km_rest_BIC_brevis",
            "tau1_rest_BIC_brevis",
            "tau2_BIC_brevis",
            "pd0_BIC_brevis",
            "pdt_BIC_brevis",
            "a_scale_BIC_brevis"
        ]
    )

    sol = ocp.solve(Solver.IPOPT(show_online_optim=False, _max_iter=2000))

    if plot:
        sol.animate(viewer="pyorerun")
        sol.graphs(show_bounds=False)


if __name__ == "__main__":
    main()
