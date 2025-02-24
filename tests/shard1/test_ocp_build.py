import pytest
import shutil
import re

import numpy as np

from cocofest import OcpFes, ModelMaker

from bioptim import ObjectiveFcn, ObjectiveList, Node

# Force and time data coming form examples/data/hand_cycling_force.bio file
force = np.array(
    [
        0,
        33.22386956,
        64.01175842,
        58.96268087,
        54.49535426,
        36.44632893,
        29.96801641,
        30.93602607,
        29.21745363,
        24.59113196,
        23.86747368,
        23.03832502,
        24.05995233,
        24.62126537,
        26.47933897,
        34.35875801,
        35.91539197,
        35.76400291,
        40.52047879,
        46.03698741,
        48.75643065,
        53.95577793,
        56.50240254,
        62.36998819,
        62.85023295,
        61.74541602,
        64.33297518,
        63.9467184,
        64.66585644,
        60.24675264,
        46.53505388,
        41.47114645,
        37.48364824,
        36.74068556,
        39.44173399,
        39.74906276,
        33.43802423,
        25.90760263,
        15.16708131,
        12.73063647,
        22.89840483,
        25.65474343,
        23.78995719,
        24.34094537,
        21.88398197,
        22.46456012,
        23.00685366,
        23.13887312,
        24.15788808,
        23.98192192,
        32.27334539,
        41.21948216,
        46.76794658,
        48.64655786,
        53.03715513,
        50.85622133,
        49.13946943,
        46.18259705,
        44.30003259,
        45.34554766,
        46.16899136,
        47.78202516,
        46.75296973,
        43.80444159,
        40.1942265,
        36.61031425,
        36.08302422,
        32.67321347,
        29.88243224,
        25.32586748,
        23.7372641,
        18.85373501,
        15.99490173,
        15.55972989,
        13.43508441,
        8.91325156,
        5.45077189,
        2.61086563,
        2.27762137,
        4.20870452,
        7.08147898,
        8.28477706,
        8.57699962,
        10.26761919,
        15.2530161,
        22.71041396,
        30.26413335,
        36.48211366,
        41.66699745,
        44.5834331,
        42.95453371,
        45.1371186,
        44.6845018,
        46.85747254,
        48.22912681,
        50.96067339,
        50.76653352,
        49.13231127,
        53.41327896,
        53.08398207,
    ]
)

time = np.array(
    [
        0.0,
        0.01010101,
        0.02020202,
        0.03030303,
        0.04040404,
        0.05050505,
        0.06060606,
        0.07070707,
        0.08080808,
        0.09090909,
        0.1010101,
        0.11111111,
        0.12121212,
        0.13131313,
        0.14141414,
        0.15151515,
        0.16161616,
        0.17171717,
        0.18181818,
        0.19191919,
        0.2020202,
        0.21212121,
        0.22222222,
        0.23232323,
        0.24242424,
        0.25252525,
        0.26262626,
        0.27272727,
        0.28282828,
        0.29292929,
        0.3030303,
        0.31313131,
        0.32323232,
        0.33333333,
        0.34343434,
        0.35353535,
        0.36363636,
        0.37373737,
        0.38383838,
        0.39393939,
        0.4040404,
        0.41414141,
        0.42424242,
        0.43434343,
        0.44444444,
        0.45454545,
        0.46464646,
        0.47474747,
        0.48484848,
        0.49494949,
        0.50505051,
        0.51515152,
        0.52525253,
        0.53535354,
        0.54545455,
        0.55555556,
        0.56565657,
        0.57575758,
        0.58585859,
        0.5959596,
        0.60606061,
        0.61616162,
        0.62626263,
        0.63636364,
        0.64646465,
        0.65656566,
        0.66666667,
        0.67676768,
        0.68686869,
        0.6969697,
        0.70707071,
        0.71717172,
        0.72727273,
        0.73737374,
        0.74747475,
        0.75757576,
        0.76767677,
        0.77777778,
        0.78787879,
        0.7979798,
        0.80808081,
        0.81818182,
        0.82828283,
        0.83838384,
        0.84848485,
        0.85858586,
        0.86868687,
        0.87878788,
        0.88888889,
        0.8989899,
        0.90909091,
        0.91919192,
        0.92929293,
        0.93939394,
        0.94949495,
        0.95959596,
        0.96969697,
        0.97979798,
        0.98989899,
        1.0,
    ]
)

init_force = force - force[0]
init_stim_time = [0, 0.1, 0.2]
init_final_time = 0.3
init_frequency = 10
init_n_shooting = 30
init_force_tracking = [time, init_force]
init_end_node_tracking = 40

ding2003 = ModelMaker.create_model("ding2003", stim_time=init_stim_time)
ding2003_with_fatigue = ModelMaker.create_model("ding2003_with_fatigue", stim_time=init_stim_time)
ding2007 = ModelMaker.create_model("ding2007", stim_time=init_stim_time)
ding2007_with_fatigue = ModelMaker.create_model("ding2007_with_fatigue", stim_time=init_stim_time)
hmed2018 = ModelMaker.create_model("hmed2018", stim_time=init_stim_time)
hmed2018_with_fatigue = ModelMaker.create_model("hmed2018_with_fatigue", stim_time=init_stim_time)


minimum_pulse_width = ding2007.pd0
minimum_pulse_intensity = hmed2018.min_pulse_intensity()


@pytest.mark.parametrize(
    "model,"
    " pulse_width_max,"
    " pulse_intensity_max,",
    [
        (ding2003, None, None),
        (
            ding2003_with_fatigue,
            None,
            None,
        ),
        (
            ding2007,
            None,
            None,
        ),
        (
            ding2007_with_fatigue,
            None,
            None,
        ),
        (
            ding2007,
            0.0006,
            None,
        ),
        (
            ding2007_with_fatigue,
            0.0006,
            None,
        ),
        (
            hmed2018,
            None,
            130,
        ),
        (
            hmed2018_with_fatigue,
            None,
            130,
        ),
    ],
)
@pytest.mark.parametrize("use_sx", [True, False])
@pytest.mark.parametrize("final_time", [init_final_time])
@pytest.mark.parametrize(
    "force_tracking, end_node_tracking",
    [(init_force_tracking, None), (None, init_end_node_tracking)],
)
def test_ocp_building(
    model,
    final_time,
    force_tracking,
    end_node_tracking,
    pulse_width_max,
    pulse_intensity_max,
    use_sx,
):
    ocp_1 = OcpFes().prepare_ocp(
        model=model,
        final_time=0.3,
        pulse_width={"max": pulse_width_max},
        pulse_intensity={"max": pulse_intensity_max},
        objective={
            "force_tracking": force_tracking,
            "end_node_tracking": end_node_tracking,
        },
        use_sx=use_sx,
    )

#
# def test_all_ocp_fes_errors():
#     # with pytest.raises(
#     #     TypeError,
#     #     match=re.escape(
#     #         f"The current model type used is {type(None)}, it must be a FesModel type."
#     #         f"Current available models are: DingModelFrequency, DingModelFrequencyWithFatigue,"
#     #         f"DingModelPulseWidthFrequency, DingModelPulseWidthFrequencyWithFatigue,"
#     #         f"DingModelPulseIntensityFrequency, DingModelPulseIntensityFrequencyWithFatigue"
#     #     ),
#     # ):
#     #     OcpFes.prepare_ocp(model=None)
#     #
#     # with pytest.raises(TypeError, match="final_time must be a positive int or float type"):
#     #     OcpFes.prepare_ocp(
#     #         model=ding2003, stim_time=[0, 0.1, 0.2], final_time="0.3"
#     #     )
#
#     pulse_mode = "doublet"
#     with pytest.raises(
#         NotImplementedError,
#         match=re.escape(f"Pulse mode '{pulse_mode}' is not yet implemented"),
#     ):
#         OcpFes.prepare_ocp(
#             model=ding2003,
#             stim_time=[0, 0.1, 0.2],
#             final_time=0.3,
#             pulse_event={"pulse_mode": pulse_mode},
#         )
#
#     with pytest.raises(
#         ValueError,
#         match="min and max time event must be both entered or none of them in order to work",
#     ):
#         OcpFes.prepare_ocp(
#             model=ding2003,
#             stim_time=[0, 0.1, 0.2],
#             final_time=0.3,
#             pulse_event={"min": 0.1},
#         )
#
#     with pytest.raises(TypeError, match=re.escape("time bimapping must be bool type")):
#         OcpFes.prepare_ocp(
#             model=ding2003,
#             stim_time=[0, 0.1, 0.2],
#             final_time=0.3,
#             pulse_event={"min": 0.01, "max": 0.1, "bimapping": "True"},
#         )
#
#     with pytest.raises(
#         ValueError,
#         match="pulse width or pulse width min max bounds need to be set for this model",
#     ):
#         OcpFes.prepare_ocp(
#             model=ding2007,
#             stim_time=[0, 0.1, 0.2],
#             final_time=0.3,
#             pulse_width={"min": 0.001},
#         )
#
#     with pytest.raises(
#         ValueError,
#         match="Either pulse width or pulse width min max bounds need to be set for this model",
#     ):
#         OcpFes.prepare_ocp(
#             model=ding2007,
#             stim_time=[0, 0.1, 0.2],
#             final_time=0.3,
#             pulse_width={"min": 0.001, "max": 0.005, "fixed": 0.003},
#         )
#
#     minimum_pulse_width = ding2007.pd0
#     fixed_pulse_width = 0.0001
#     with pytest.raises(
#         ValueError,
#         match=re.escape(
#             f"The pulse width set ({fixed_pulse_width})"
#             f" is lower than minimum width required."
#             f" Set a value above {minimum_pulse_width} seconds "
#         ),
#     ):
#         OcpFes.prepare_ocp(
#             model=ding2007,
#             stim_time=[0, 0.1, 0.2],
#             final_time=0.3,
#             pulse_width={"fixed": fixed_pulse_width},
#         )
#
#     with pytest.raises(TypeError, match="Wrong pulse_width type, only int or float accepted"):
#         OcpFes.prepare_ocp(
#             model=ding2007,
#             stim_time=[0, 0.1, 0.2],
#             final_time=0.3,
#             pulse_width={"fixed": "0.001"},
#         )
#
#     with pytest.raises(
#         TypeError,
#         match="min and max pulse width must be int or float type",
#     ):
#         OcpFes.prepare_ocp(
#             model=ding2007,
#             stim_time=[0, 0.1, 0.2],
#             final_time=0.3,
#             pulse_width={"min": "0.001", "max": 0.005},
#         )
#
#     with pytest.raises(
#         ValueError,
#         match="The set minimum pulse width is higher than maximum pulse width.",
#     ):
#         OcpFes.prepare_ocp(
#             model=ding2007,
#             stim_time=[0, 0.1, 0.2],
#             final_time=0.3,
#             pulse_width={"min": 0.005, "max": 0.001},
#         )
#
#     pulse_width_min = fixed_pulse_width
#     with pytest.raises(
#         ValueError,
#         match=re.escape(
#             f"The pulse width set ({pulse_width_min})"
#             f" is lower than minimum width required."
#             f" Set a value above {minimum_pulse_width} seconds "
#         ),
#     ):
#         OcpFes.prepare_ocp(
#             model=ding2007,
#             stim_time=[0, 0.1, 0.2],
#             final_time=0.3,
#             pulse_width={"min": pulse_width_min, "max": 0.005},
#         )
#
#     with pytest.raises(
#         ValueError,
#         match="Pulse intensity or pulse intensity min max bounds need to be set for this model",
#     ):
#         OcpFes.prepare_ocp(model=hmed2018, stim_time=[0, 0.1, 0.2], final_time=0.3)
#
#     with pytest.raises(
#         ValueError,
#         match="Either pulse intensity or pulse intensity min max bounds need to be set for this model",
#     ):
#         OcpFes.prepare_ocp(
#             model=hmed2018,
#             stim_time=[0, 0.1, 0.2],
#             final_time=0.3,
#             pulse_intensity={"min": 20, "max": 100, "fixed": 50},
#         )
#
#     with pytest.raises(
#         ValueError,
#         match="Pulse intensity or pulse intensity min max bounds need to be set for this model",
#     ):
#         OcpFes.prepare_ocp(
#             model=hmed2018,
#             stim_time=[0, 0.1, 0.2],
#             final_time=0.3,
#             pulse_intensity={"min": 20},
#         )
#
#     minimum_pulse_intensity = hmed2018.min_pulse_intensity()
#     fixed_pulse_intensity = 1
#     with pytest.raises(
#         ValueError,
#         match=re.escape(
#             f"The pulse intensity set ({fixed_pulse_intensity})"
#             f" is lower than minimum intensity required."
#             f" Set a value above {minimum_pulse_intensity} mA "
#         ),
#     ):
#         OcpFes.prepare_ocp(
#             model=hmed2018,
#             stim_time=[0, 0.1, 0.2],
#             final_time=0.3,
#             pulse_intensity={"fixed": fixed_pulse_intensity},
#         )
#
#     with pytest.raises(TypeError, match="pulse_intensity must be int or float type"):
#         OcpFes.prepare_ocp(
#             model=hmed2018,
#             stim_time=[0, 0.1, 0.2],
#             final_time=0.3,
#             pulse_intensity={"fixed": "20"},
#         )
#
#     with pytest.raises(
#         TypeError,
#         match="pulse_intensity_min and pulse_intensity_max must be int or float type",
#     ):
#         OcpFes.prepare_ocp(
#             model=hmed2018,
#             stim_time=[0, 0.1, 0.2],
#             final_time=0.3,
#             pulse_intensity={"min": "20", "max": 100},
#         )
#
#     with pytest.raises(
#         ValueError,
#         match="The set minimum pulse intensity is higher than maximum pulse intensity.",
#     ):
#         OcpFes.prepare_ocp(
#             model=hmed2018,
#             stim_time=[0, 0.1, 0.2],
#             final_time=0.3,
#             pulse_intensity={"min": 100, "max": 1},
#         )
#
#     pulse_intensity_min = fixed_pulse_intensity
#     with pytest.raises(
#         ValueError,
#         match=re.escape(
#             f"The pulse intensity set ({pulse_intensity_min})"
#             f" is lower than minimum intensity required."
#             f" Set a value above {minimum_pulse_intensity} mA "
#         ),
#     ):
#         OcpFes.prepare_ocp(
#             model=hmed2018,
#             stim_time=[0, 0.1, 0.2],
#             final_time=0.3,
#             pulse_intensity={"min": pulse_intensity_min, "max": 100},
#         )
#
#     with pytest.raises(
#         ValueError,
#         match="force_tracking time and force argument must be same length and force_tracking " "list size 2",
#     ):
#         OcpFes.prepare_ocp(
#             model=ding2003,
#             stim_time=[0, 0.1, 0.2],
#             final_time=0.3,
#             objective={"force_tracking": [np.array([0, 1]), np.array([0, 1, 2])]},
#         )
#
#     with pytest.raises(TypeError, match="force_tracking argument must be np.ndarray type"):
#         OcpFes.prepare_ocp(
#             model=ding2003,
#             stim_time=[0, 0.1, 0.2],
#             final_time=0.3,
#             objective={"force_tracking": [[0, 1, 2], np.array([0, 1, 2])]},
#         )
#
#     with pytest.raises(TypeError, match="force_tracking must be list type"):
#         OcpFes.prepare_ocp(
#             model=ding2003,
#             stim_time=[0, 0.1, 0.2],
#             final_time=0.3,
#             objective={"force_tracking": np.array([np.array([0, 1, 2]), np.array([0, 1, 2])])},
#         )
#
#     with pytest.raises(TypeError, match="end_node_tracking must be int or float type"):
#         OcpFes.prepare_ocp(
#             model=ding2003,
#             stim_time=[0, 0.1, 0.2],
#             final_time=0.3,
#             objective={"end_node_tracking": "10"},
#         )
#
#     objective_functions = ObjectiveList()
#     objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="q", weight=10, multi_thread=False)
#     objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=10, multi_thread=False)
#     objective_functions[0].append("objective_function")
#     with pytest.raises(TypeError, match="custom_objective must be a ObjectiveList type"):
#         OcpFes.prepare_ocp(
#             model=ding2003,
#             stim_time=[0, 0.1, 0.2],
#             final_time=0.3,
#             objective={"custom": "objective_functions"},
#         )
#
#     with pytest.raises(TypeError, match="All elements in ObjectiveList must be an Objective type"):
#         OcpFes.prepare_ocp(
#             model=ding2003,
#             stim_time=[0, 0.1, 0.2],
#             final_time=0.3,
#             objective={"custom": objective_functions},
#         )
#
#     with pytest.raises(TypeError, match="ode_solver must be a OdeSolver type"):
#         OcpFes.prepare_ocp(
#             model=ding2003,
#             stim_time=[0, 0.1, 0.2],
#             final_time=0.3,
#             ode_solver="ode_solver",
#         )
#
#     with pytest.raises(TypeError, match="use_sx must be a bool type"):
#         OcpFes.prepare_ocp(
#             model=ding2003,
#             stim_time=[0, 0.1, 0.2],
#             final_time=0.3,
#             use_sx="True",
#         )
#
#     with pytest.raises(TypeError, match="n_thread must be a int type"):
#         OcpFes.prepare_ocp(
#             model=ding2003,
#             stim_time=[0, 0.1, 0.2],
#             final_time=0.3,
#             n_threads="1",
#         )
