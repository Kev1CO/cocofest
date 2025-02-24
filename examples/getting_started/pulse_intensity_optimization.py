"""
This example will do a 10 stimulation example with Ding's 2003 frequency model associated to Hmed's 2018 work.
This ocp was build to match a force value of 200N at the end of the last node.
"""

import numpy as np
from bioptim import OdeSolver
from cocofest import DingModelPulseIntensityFrequencyWithFatigue, OcpFes

# --- Build ocp --- #
# This ocp was build to match a force value of 200N at the end of the last node.
# The stimulation won't be optimized and is already set to one pulse every 0.1 seconds (n_stim/final_time).
# Plus the pulsation intensity will be optimized between 0 and 130 mA and are not the same across the problem.


def prepare_ocp():
    final_time = 1
    model = DingModelPulseIntensityFrequencyWithFatigue(stim_time=list(np.linspace(0, final_time, 11)[:-1]))

    return OcpFes().prepare_ocp(
        model=model,
        final_time=final_time,
        pulse_intensity={"max": 130},
        objective={"end_node_tracking": 200},
        use_sx=True,
        n_threads=8,
        ode_solver=OdeSolver.RK4(n_integration_steps=10),
    )


def main():
    ocp = prepare_ocp()
    # --- Solve the program --- #
    sol = ocp.solve()
    # --- Show results --- #
    sol.graphs()


if __name__ == "__main__":
    main()
