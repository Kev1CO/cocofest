"""
This example will do a 10 stimulation example with Ding's 2007 pulse duration and frequency model.
This ocp was build to match a force value of 200N at the end of the last node.
"""
from casadi import MX
from cocofest import DingModelFrequency, OcpFes, ModelBuilder


ding_builder = ModelBuilder(model=DingModelFrequency,
                            stim_time=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                            tau2=None,
                            a_rest=4000)

# Set known stim times
models = ding_builder.build()
ocp = OcpFes.prepare_ocp(models=models,
                           n_stim=10,
                           n_shooting=20,
                           final_time=1,
                           pulse_event={"min": 0.01, "max": 0.1, "bimapping": True},
                           objective={"end_node_tracking": 200},
                           use_sx=True,
        )

sol = ocp.solve()
sol.graphs()

# Set symbolic stim times
ding_builder.set_stim_times(MX.sym('t', 4, 1))

