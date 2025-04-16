import numpy as np

from bioptim import (
    OdeSolver,
    OptimalControlProgram,
    ObjectiveFcn,
    Node,
    ControlType,
    ObjectiveList,
    BoundsList,
    InitialGuessList,
    InterpolationType,
)

from cocofest import (
    DingModelPulseWidthFrequency,
    FesMskModel,
    OcpFesId,
    OcpFesMsk,
    OcpFesIdMultibody,
)


def prepare_ocp(
    model,
    final_time,
    pulse_width_values,
    key_parameter_to_identify,
    use_sx=False,
):
    fes_model = model.muscles_dynamics_model
    n_shooting = fes_model[0].get_n_shooting(final_time)
    numerical_data_time_series, stim_idx_at_node_list = fes_model[0].get_numerical_data_time_series(
        n_shooting, final_time
    )

    dynamics = OcpFesMsk.declare_dynamics(
        model,
        numerical_time_series=numerical_data_time_series, with_contact=False)

    x_bounds, x_init = OcpFesMsk.set_x_bounds_fes(model)
    q_x_bounds = model.bounds_from_ranges("q")
    qdot_x_bounds = model.bounds_from_ranges("qdot")
    x_bounds.add(key="q", bounds=q_x_bounds, phase=0)
    x_bounds.add(key="qdot", bounds=qdot_x_bounds, phase=0)

    u_bounds = BoundsList()  # Controls bounds
    u_init = InitialGuessList()  # Controls initial guess
    if isinstance(fes_model[0], DingModelPulseWidthFrequency):
        if len(pulse_width_values) != 1:
            last_stim_idx = [stim_idx_at_node_list[i][-1] for i in range(len(stim_idx_at_node_list) - 1)]
            control_bounds = [pulse_width_values[last_stim_idx[i]] for i in range(len(last_stim_idx))]
        else:
            control_bounds = [pulse_width_values] * n_shooting
        for muscle in model.muscles_dynamics_model:
            u_init.add(key="last_pulse_width_" + muscle.muscle_name, initial_guess=[pulse_width_values], phase=0)
            u_bounds.add(
                "last_pulse_width_" + muscle.muscle_name,
                min_bound=np.array([control_bounds]),
                max_bound=np.array([control_bounds]),
                interpolation=InterpolationType.EACH_FRAME,
            )

    target = list(np.linspace(0, 3, n_shooting + 1))
    target = np.array(target)[np.newaxis, :]
    objective_functions = ObjectiveList()
    objective_functions.add(
        ObjectiveFcn.Lagrange.MINIMIZE_STATE,
        key="q",
        weight=1,
        target=target,
        node=Node.ALL,
        quadratic=True,
        index=[0],
    )

    x_init.add(key="q", initial_guess=target, interpolation=InterpolationType.EACH_FRAME, phase=0)
    additional_key_settings = OcpFesIdMultibody.set_default_values(msk_model=model)

    parameters, parameters_bounds, parameters_init = OcpFesId.set_parameters(
        parameter_to_identify=key_parameter_to_identify,
        parameter_setting=additional_key_settings,
        use_sx=use_sx,
    )

    model = OcpFesMsk.update_model(model, parameters=parameters, external_force_set=None)

    for param_key in parameters:
        if parameters[param_key].function:
            param_scaling = parameters[param_key].scaling.scaling
            param_reduced = parameters[param_key].cx
            fes_model_for_param_key = [
                fes_model[i] for i in range(len(fes_model)) if fes_model[i].muscle_name in param_key
            ][0]
            parameters[param_key].function(
                fes_model_for_param_key, param_reduced * param_scaling, **parameters[param_key].kwargs
            )

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
        use_sx=use_sx,
        n_threads=20,
        #ode_solver=OdeSolver.RK4(n_integration_steps=10),
    )


def main(plot=True):

    final_time = 1
    n_stim = 33
    model = FesMskModel(
        name=None,
        biorbd_path="../model_msk/simplified_UL_Seth_BIC_elbow.bioMod",
        muscles_model=[
            DingModelPulseWidthFrequency(muscle_name="BIC_long", sum_stim_truncation=10),
            DingModelPulseWidthFrequency(muscle_name="BIC_brevis", sum_stim_truncation=10),
        ],
        stim_time=list(np.linspace(0, final_time, n_stim + 1)[:-1]),
        activate_force_length_relationship=True,
        activate_force_velocity_relationship=True,
        activate_passive_force_relationship=True,
        activate_residual_torque=False,
        external_force_set=None,  # External forces will be added later
    )

    pulse_width_values = [0.0006]

    from bioptim import CostType, Solver

    ocp = prepare_ocp(
        model,
        final_time,
        pulse_width_values,
        key_parameter_to_identify=[
            "tau1_rest_BIC_brevis",
            "tau2_BIC_brevis",
            "km_rest_BIC_brevis",
            "a_scale_BIC_brevis",
            "pd0_BIC_brevis",
            "pdt_BIC_brevis",
            "tau1_rest_BIC_long",
            "tau2_BIC_long",
            "km_rest_BIC_long",
            "a_scale_BIC_long",
            "pd0_BIC_long",
            "pdt_BIC_long",
        ],
    )

    ocp.add_plot_penalty(CostType.ALL)
    sol = ocp.solve(Solver.IPOPT(_max_iter=1000))
    print(sol.parameters)
    # sol.graphs(show_bounds=True)
    # sol.animate()


if __name__ == "__main__":
    main()
