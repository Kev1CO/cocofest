import numpy as np

from bioptim import (
    OdeSolver,
    OptimalControlProgram,
    ObjectiveFcn,
    ControlType,
    ObjectiveList,
    BoundsList,
    InitialGuessList,
    InterpolationType,
    DynamicsList,
    PhaseDynamics,
    Solution,
    Shooting,
    SolutionIntegrator,
    BiorbdModel
)
from cocofest import BioRbdModelWithPassiveTorque


def prepare_ocp(
    biomodel_path,
    final_time: float,
    n_shooting: int = 30,
    use_sx=False,
):
    models = [BiorbdModel(biomodel_path)]


    objective_functions = ObjectiveList()

    # Dynamics definition
    model = BioRbdModelWithPassiveTorque()
    dynamics = DynamicsList()
    dynamics.add(
        model.declare_model_variables,
        dynamic_function=model.muscle_dynamic,
        expand_dynamics=True,
        phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        numerical_data_timeseries=None,
        with_passive_torque=True,
    )

    # States bounds and initial guess
    x_bounds, x_init = BoundsList(), InitialGuessList()

    x_init["q"] = [90]
    x_init["qdot"] = [0]

    q_x_bounds = models[0].bounds_from_ranges("q")
    q_x_bounds.min[0][1] = 0
    q_x_bounds.min[0][2] = 0
    q_x_bounds.max[0][1] = 5
    q_x_bounds.max[0][2] = 5

    qdot_x_bounds = models[0].bounds_from_ranges("qdot")

    x_bounds.add(key="q", bounds=q_x_bounds, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT)
    x_bounds.add(key="qdot", bounds=qdot_x_bounds, interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT)

    u_bounds = BoundsList()
    u_init = InitialGuessList()
    u_init.add(key="tau", value=[0], interpolation=InterpolationType.CONSTANT)
    u_bounds.add(key="tau", max_bound=100, min_bound=-100, interpolation=InterpolationType.CONSTANT)

    u_bounds.add(key="muscles", min_bound=[0, 0], max_bound=[0, 0])

    u_init["muscles"] = [0, 0]

    # Add objective functions
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau")

    return OptimalControlProgram(
        bio_model=models,
        dynamics=dynamics,
        n_shooting=n_shooting,
        phase_time=final_time,
        x_init=x_init,
        x_bounds=x_bounds,
        u_init=u_init,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        control_type=ControlType.CONSTANT,
        use_sx=use_sx,
        n_threads=20,
        ode_solver=OdeSolver.RK4(n_integration_steps=10),
    )

def main():
    ocp = prepare_ocp(biomodel_path="../model_msk/p05_scaling_scaled.bioMod", final_time=1, n_shooting=30, use_sx=False)

    # Simulation the Initial Guess
    # Interpolation: Constant
    X = InitialGuessList()
    X["q"] = [90]
    X["qdot"] = [0]

    U = InitialGuessList()
    U["tau"] = [0]
    U["muscles"] = [0, 0]

    sol_from_initial_guess = Solution.from_initial_guess(
        ocp, [np.array([1 / 30]), X, U, InitialGuessList(), InitialGuessList()]
    )
    s = sol_from_initial_guess.integrate(shooting_type=Shooting.SINGLE, integrator=SolutionIntegrator.OCP)
    print(f"Final position of q from single shooting of initial guess = {s['q'][-1]}")
    # Uncomment the next line to animate the integration
    # s.animate()

    # Interpolation: Each frame (for instance, values from a previous optimization or from measured data)
    random_u = np.random.rand(2, 30)
    U = InitialGuessList()
    U.add("tau", random_u, interpolation=InterpolationType.EACH_FRAME)

    sol_from_initial_guess = Solution.from_initial_guess(
        ocp, [np.array([1 / 30]), X, U, InitialGuessList(), InitialGuessList()]
    )
    s = sol_from_initial_guess.integrate(shooting_type=Shooting.SINGLE, integrator=SolutionIntegrator.OCP)
    print(f"Final position of q from single shooting of initial guess = {s['q'][-1]}")
    # Uncomment the next line to animate the integration
    # s.animate()

    # Uncomment the following lines to graph the solution from initial guesses
    # sol_from_initial_guess.graphs(shooting_type=Shooting.SINGLE)
    # sol_from_initial_guess.graphs(shooting_type=Shooting.MULTIPLE)

    # Simulation of the solution. It is not the graph of the solution,
    # it is the graph of a Runge Kutta from the solution
    sol = ocp.solve()
    s_single = sol.integrate(shooting_type=Shooting.SINGLE, integrator=SolutionIntegrator.OCP)
    # Uncomment the next line to animate the integration
    # s_single.animate()
    print(f"Final position of q from single shooting of the solution = {s_single['q'][-1]}")
    s_multiple = sol.integrate(shooting_type=Shooting.MULTIPLE, integrator=SolutionIntegrator.OCP)
    print(f"Final position of q from multiple shooting of the solution = {s_multiple['q'][-1]}")

    # Uncomment the following lines to graph the solution from the actual solution
    # sol.graphs(shooting_type=Shooting.SINGLE)
    # sol.graphs(shooting_type=Shooting.MULTIPLE)

    return s, s_single, s_multiple


if __name__ == "__main__":
    main()
