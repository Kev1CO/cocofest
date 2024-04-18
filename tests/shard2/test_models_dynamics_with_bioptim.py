import numpy as np
import pytest

from bioptim import Solver, SolutionMerge
from cocofest import (
    DingModelFrequency,
    DingModelFrequencyWithFatigue,
    DingModelPulseDurationFrequency,
    DingModelIntensityFrequency,
    OcpFes,
)

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
init_force_tracking = [time, init_force]

minimum_pulse_duration = DingModelPulseDurationFrequency().pd0
minimum_pulse_intensity = DingModelIntensityFrequency().min_pulse_intensity()


@pytest.mark.parametrize("use_sx", [True])  # Later add False
@pytest.mark.parametrize(
    "model", [DingModelFrequency(), DingModelPulseDurationFrequency(), DingModelIntensityFrequency()]
)
@pytest.mark.parametrize("force_tracking", [init_force_tracking])
@pytest.mark.parametrize("min_pulse_duration, min_pulse_intensity", [(minimum_pulse_duration, minimum_pulse_intensity)])
def test_ocp_output(model, force_tracking, use_sx, min_pulse_duration, min_pulse_intensity):
    if isinstance(model, DingModelPulseDurationFrequency):
        ocp = OcpFes().prepare_ocp(
            model=model,
            n_shooting=20,
            n_stim=10,
            final_time=1,
            pulse_duration_dict={
                "pulse_duration_min": min_pulse_duration,
                "pulse_duration_max": 0.0006,
                "pulse_duration_bimapping": False,
            },
            objective_dict={"force_tracking": force_tracking},
            use_sx=use_sx,
        )

        ocp = ocp.solve(Solver.IPOPT(show_online_optim=False, _max_iter=1000))

        # TODO : Add a pickle file to test
        # for key in ocp.states.key():
        #     np.testing.assert_almost_equal(ocp.states[key], pickle_file.states[key])

    elif isinstance(model, DingModelIntensityFrequency):
        ocp = OcpFes().prepare_ocp(
            model=model,
            n_shooting=20,
            n_stim=10,
            final_time=1,
            pulse_intensity_dict={
                "pulse_intensity_min": min_pulse_intensity,
                "pulse_intensity_max": 130,
                "pulse_intensity_bimapping": False,
            },
            objective_dict={"force_tracking": force_tracking},
            use_sx=use_sx,
        )

        ocp = ocp.solve(Solver.IPOPT(show_online_optim=False, _max_iter=1000))

        # TODO : Add a pickle file to test
        # for key in ocp.states.key():
        #     np.testing.assert_almost_equal(ocp.states[key], pickle_file.states[key])

    elif isinstance(model, DingModelFrequency):
        ocp = OcpFes().prepare_ocp(
            model=model,
            n_shooting=20,
            n_stim=10,
            final_time=1,
            pulse_apparition_dict={"time_min": 0.01, "time_max": 1, "time_bimapping": False},
            objective_dict={"end_node_tracking": 50},
            use_sx=use_sx,
        )

        ocp = ocp.solve(Solver.IPOPT(show_online_optim=False, _max_iter=1000))

        # TODO : Add a pickle file to test
        # for key in ocp.states.key():
        #     np.testing.assert_almost_equal(ocp.states[key], pickle_file.states[key])


# TODO : add test_multi_start_ocp


@pytest.mark.parametrize("use_sx", [True])
@pytest.mark.parametrize("bimapped", [False, True])
def test_time_dependent_ocp_output(use_sx, bimapped):
    ocp = OcpFes().prepare_ocp(
        model=DingModelFrequencyWithFatigue(),
        n_stim=10,
        n_shooting=20,
        final_time=1,
        pulse_apparition_dict={"time_min": 0.01, "time_max": 0.1, "time_bimapping": bimapped},
        objective_dict={"end_node_tracking": 270},
        use_sx=use_sx,
    )

    if use_sx and not bimapped:
        return  # This test works offline but not in GitHub actions
    sol = ocp.solve(Solver.IPOPT(show_online_optim=False, _max_iter=10000))
    sol_states = sol.decision_states(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])
    sol_time = sol.decision_time(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])

    if bimapped:
        np.testing.assert_almost_equal(
            sol.parameters["pulse_apparition_time"],
            np.array(
                [
                    -1.68030676e-11,
                    4.22468456e-02,
                    8.44936912e-02,
                    1.26740537e-01,
                    1.68987382e-01,
                    2.11234228e-01,
                    2.53481074e-01,
                    2.95727919e-01,
                    3.37974765e-01,
                    3.80221610e-01,
                ]
            ),
        )

        np.testing.assert_almost_equal(
            sol_states["F"],
            np.array(
                [
                    [
                        5.36392556e-16,
                        1.87412759e00,
                        5.36077878e00,
                        9.39335896e00,
                        1.36581424e01,
                        1.80180487e01,
                        2.24003292e01,
                        2.67614126e01,
                        3.10728875e01,
                        3.53149856e01,
                        3.94732057e01,
                        4.35364211e01,
                        4.74957490e01,
                        5.13438431e01,
                        5.50744336e01,
                        5.86820190e01,
                        6.21616559e01,
                        6.55088142e01,
                        6.87192773e01,
                        7.17890751e01,
                        7.47144413e01,
                        7.47144413e01,
                        7.77280045e01,
                        8.09402585e01,
                        8.42394497e01,
                        8.75650375e01,
                        9.08801928e01,
                        9.41605759e01,
                        9.73890322e01,
                        1.00552810e02,
                        1.03641987e02,
                        1.06648515e02,
                        1.09565627e02,
                        1.12387441e02,
                        1.15108693e02,
                        1.17724554e02,
                        1.20230502e02,
                        1.22622232e02,
                        1.24895595e02,
                        1.27046553e02,
                        1.29071155e02,
                        1.30965530e02,
                        1.30965530e02,
                        1.32969164e02,
                        1.35196023e02,
                        1.37531911e02,
                        1.39915320e02,
                        1.42308917e02,
                        1.44688027e02,
                        1.47035233e02,
                        1.49337546e02,
                        1.51584815e02,
                        1.53768774e02,
                        1.55882431e02,
                        1.57919675e02,
                        1.59875009e02,
                        1.61743362e02,
                        1.63519959e02,
                        1.65200229e02,
                        1.66779737e02,
                        1.68254143e02,
                        1.69619172e02,
                        1.70870596e02,
                        1.70870596e02,
                        1.72222194e02,
                        1.73782126e02,
                        1.75450838e02,
                        1.77172879e02,
                        1.78913767e02,
                        1.80650263e02,
                        1.82365676e02,
                        1.84047371e02,
                        1.85685333e02,
                        1.87271297e02,
                        1.88798191e02,
                        1.90259770e02,
                        1.91650367e02,
                        1.92964715e02,
                        1.94197825e02,
                        1.95344899e02,
                        1.96401267e02,
                        1.97362343e02,
                        1.98223604e02,
                        1.98980566e02,
                        1.98980566e02,
                        1.99845869e02,
                        2.00928426e02,
                        2.02129603e02,
                        2.03394014e02,
                        2.04686992e02,
                        2.05985035e02,
                        2.07271175e02,
                        2.08532499e02,
                        2.09758723e02,
                        2.10941323e02,
                        2.12072976e02,
                        2.13147200e02,
                        2.14158092e02,
                        2.15100162e02,
                        2.15968203e02,
                        2.16757203e02,
                        2.17462286e02,
                        2.18078664e02,
                        2.18601615e02,
                        2.19026465e02,
                        2.19026465e02,
                        2.19567954e02,
                        2.20336167e02,
                        2.21231364e02,
                        2.22197429e02,
                        2.23199174e02,
                        2.24212698e02,
                        2.25220710e02,
                        2.26210024e02,
                        2.27170118e02,
                        2.28092254e02,
                        2.28968917e02,
                        2.29793442e02,
                        2.30559759e02,
                        2.31262218e02,
                        2.31895459e02,
                        2.32454326e02,
                        2.32933801e02,
                        2.33328963e02,
                        2.33634960e02,
                        2.33846991e02,
                        2.33846991e02,
                        2.34181173e02,
                        2.34748266e02,
                        2.35447754e02,
                        2.36223051e02,
                        2.37038639e02,
                        2.37870372e02,
                        2.38700762e02,
                        2.39516457e02,
                        2.40306791e02,
                        2.41062896e02,
                        2.41777139e02,
                        2.42442743e02,
                        2.43053537e02,
                        2.43603770e02,
                        2.44087991e02,
                        2.44500953e02,
                        2.44837553e02,
                        2.45092788e02,
                        2.45261725e02,
                        2.45339485e02,
                        2.45339485e02,
                        2.45542609e02,
                        2.45982081e02,
                        2.46557128e02,
                        2.47210989e02,
                        2.47908016e02,
                        2.48623960e02,
                        2.49341244e02,
                        2.50046435e02,
                        2.50728796e02,
                        2.51379390e02,
                        2.51990523e02,
                        2.52555356e02,
                        2.53067661e02,
                        2.53521632e02,
                        2.53911764e02,
                        2.54232758e02,
                        2.54479463e02,
                        2.54646825e02,
                        2.54729865e02,
                        2.54723660e02,
                        2.54723660e02,
                        2.54844501e02,
                        2.55203302e02,
                        2.55699397e02,
                        2.56276058e02,
                        2.56897645e02,
                        2.57539897e02,
                        2.58185220e02,
                        2.58820159e02,
                        2.59433952e02,
                        2.60017637e02,
                        2.60563489e02,
                        2.61064644e02,
                        2.61514843e02,
                        2.61908254e02,
                        2.62239343e02,
                        2.62502784e02,
                        2.62693398e02,
                        2.62806106e02,
                        2.62835903e02,
                        2.62777839e02,
                        2.62777839e02,
                        2.62847538e02,
                        2.63155657e02,
                        2.63601866e02,
                        2.64129604e02,
                        2.64703318e02,
                        2.65298800e02,
                        2.65898480e02,
                        2.66488920e02,
                        2.67059360e02,
                        2.67600837e02,
                        2.68105622e02,
                        2.68566843e02,
                        2.68978231e02,
                        2.69333942e02,
                        2.69628431e02,
                        2.69856361e02,
                        2.70012537e02,
                        2.70091871e02,
                        2.70089342e02,
                        2.69999990e02,
                    ]
                ]
            ),
            decimal=6,
        )

        np.testing.assert_almost_equal(float(sol.cost), 3.435742447837209e-05)
        np.testing.assert_almost_equal(sol_time[-1], 0.42246846)

    else:
        if use_sx:
            np.testing.assert_almost_equal(
                sol.parameters["pulse_apparition_time"],
                np.array(
                    [
                        0.0,
                        0.09999985,
                        0.19999995,
                        0.29999984,
                        0.37593192,
                        0.43452827,
                        0.47115987,
                        0.49363655,
                        0.51175686,
                        0.52992737,
                    ]
                ),
            )

            np.testing.assert_almost_equal(
                sol_states["F"],
                np.array(
                    [
                        [
                            0.0,
                            6.64762912,
                            16.70518569,
                            27.05778762,
                            37.13748215,
                            46.72193021,
                            55.69164298,
                            63.96720249,
                            71.48718717,
                            78.19939799,
                            84.05766898,
                            89.0215613,
                            93.05779189,
                            96.14275015,
                            98.26558189,
                            99.43130904,
                            99.66342183,
                            99.00540387,
                            97.52075639,
                            95.29130595,
                            92.41385668,
                            92.41385668,
                            94.95868576,
                            100.8301898,
                            107.20861361,
                            113.5094551,
                            119.49133231,
                            125.01896684,
                            129.9994953,
                            134.35961402,
                            138.03637597,
                            140.97380528,
                            143.12254411,
                            144.44133227,
                            144.89963862,
                            144.48089132,
                            143.18574072,
                            141.03475188,
                            138.06995125,
                            134.35476,
                            129.97208984,
                            125.02067486,
                            125.02067486,
                            125.93747129,
                            130.44554313,
                            135.56691258,
                            140.68950312,
                            145.55997706,
                            150.03574241,
                            154.01854976,
                            157.43068829,
                            160.20537841,
                            162.28319041,
                            163.61158803,
                            164.14635114,
                            163.85417395,
                            162.71587908,
                            160.72966772,
                            157.9138037,
                            154.30814706,
                            149.97407818,
                            144.99258611,
                            139.46060122,
                            139.46060122,
                            139.22726118,
                            141.8729121,
                            145.24570509,
                            148.83554256,
                            152.42985495,
                            155.91441437,
                            159.21676949,
                            162.28491325,
                            165.07778773,
                            167.56061885,
                            169.70252919,
                            171.47536623,
                            172.85325542,
                            173.81262923,
                            174.33258979,
                            174.39550802,
                            173.98777761,
                            173.10064342,
                            171.73101926,
                            169.88220636,
                            169.88220636,
                            169.82538981,
                            171.21993011,
                            173.14696581,
                            175.30285141,
                            177.54580224,
                            179.79606411,
                            182.0027739,
                            184.13025055,
                            186.15145985,
                            188.04457599,
                            189.79104211,
                            191.37442521,
                            192.77972584,
                            193.99296726,
                            195.0009684,
                            195.79124551,
                            196.35200938,
                            196.6722365,
                            196.74179852,
                            196.55163755,
                            196.55163755,
                            196.70347924,
                            197.31031049,
                            198.16537866,
                            199.16706512,
                            200.25735504,
                            201.39966494,
                            202.56918905,
                            203.74812797,
                            204.92310338,
                            206.08365543,
                            207.22131976,
                            208.32903461,
                            209.40074581,
                            210.43113553,
                            211.41543145,
                            212.34926978,
                            213.22859563,
                            214.04958991,
                            214.80861554,
                            215.50217828,
                            215.50217828,
                            215.95711137,
                            216.49187004,
                            217.08373372,
                            217.71707314,
                            218.38063993,
                            219.06603963,
                            219.76681693,
                            220.47787986,
                            221.19512245,
                            221.91516881,
                            222.63519452,
                            223.35279887,
                            224.06591133,
                            224.77272197,
                            225.47162862,
                            226.16119621,
                            226.84012516,
                            227.50722629,
                            228.16140091,
                            228.80162469,
                            228.80162469,
                            229.32678909,
                            229.8769511,
                            230.44621458,
                            231.03003367,
                            231.62482144,
                            232.22769073,
                            232.83627689,
                            233.44861309,
                            234.06304056,
                            234.67814263,
                            235.29269525,
                            235.90562943,
                            236.51600212,
                            237.12297356,
                            237.72578917,
                            238.32376521,
                            238.91627705,
                            239.50274972,
                            240.08265001,
                            240.65548001,
                            240.65548001,
                            241.23419333,
                            241.8262122,
                            242.42815813,
                            243.03733978,
                            243.65157271,
                            244.26905447,
                            244.88827573,
                            245.50795569,
                            246.12699408,
                            246.74443502,
                            247.35943928,
                            247.97126266,
                            248.57923909,
                            249.18276704,
                            249.78129867,
                            250.37433096,
                            250.96139848,
                            251.54206738,
                            252.11593038,
                            252.6826026,
                            252.6826026,
                            253.61138349,
                            254.55711132,
                            255.51177571,
                            256.46943304,
                            257.42551089,
                            258.37637951,
                            259.3190757,
                            260.2511181,
                            261.17037966,
                            262.07499708,
                            262.96330488,
                            263.8337865,
                            264.68503717,
                            265.51573533,
                            266.32462021,
                            267.11047409,
                            267.87210795,
                            268.60834986,
                            269.31803544,
                            269.99999994,
                        ]
                    ]
                ),
                decimal=4,
            )

            np.testing.assert_almost_equal(float(sol.cost), 8.957958763736209e-05)
            np.testing.assert_almost_equal(sol_time[-1], 0.55952257)
        else:
            np.testing.assert_almost_equal(
                sol.parameters["pulse_apparition_time"],
                np.array(
                    [
                        0.0,
                        0.09988949,
                        0.18580569,
                        0.2647265,
                        0.3370466,
                        0.40103479,
                        0.44791441,
                        0.47607527,
                        0.49633974,
                        0.51528988,
                    ]
                ),
            )

            np.testing.assert_almost_equal(
                sol_states["F"],
                np.array(
                    [
                        [
                            0.0,
                            6.63769747,
                            16.68282577,
                            27.02426936,
                            37.09441944,
                            46.67107894,
                            55.63487008,
                            63.90648691,
                            71.4246246,
                            78.13719916,
                            83.99814521,
                            88.96709187,
                            93.0107696,
                            96.10550385,
                            98.24027834,
                            99.41984124,
                            99.66729593,
                            99.02563733,
                            97.55780468,
                            95.34502996,
                            92.48353764,
                            94.3679887,
                            99.19373278,
                            104.61057248,
                            110.09175859,
                            115.41980787,
                            120.47675267,
                            125.18588679,
                            129.49014978,
                            133.34274228,
                            136.70277619,
                            139.53337259,
                            141.80113751,
                            143.47651297,
                            144.53472176,
                            144.95710379,
                            144.73265692,
                            143.85958532,
                            142.34664627,
                            140.21408843,
                            137.4940017,
                            138.00965825,
                            140.93185136,
                            144.47366896,
                            148.17416839,
                            151.83490566,
                            155.34656941,
                            158.63813086,
                            161.65733219,
                            164.36193362,
                            166.71543182,
                            168.68493277,
                            170.24020613,
                            171.35346875,
                            171.99966272,
                            172.15708594,
                            171.80826757,
                            170.94098748,
                            169.54933267,
                            167.63467516,
                            165.20645284,
                            165.14689189,
                            167.05290682,
                            169.55568886,
                            172.25972468,
                            174.98830273,
                            177.64281194,
                            180.15929418,
                            182.49124724,
                            184.60170505,
                            186.45922803,
                            188.03577597,
                            189.30559603,
                            190.24471837,
                            190.83085223,
                            191.04356675,
                            190.86468254,
                            190.2788171,
                            189.2740306,
                            187.8425161,
                            185.98127339,
                            185.74144989,
                            186.97919616,
                            188.74521107,
                            190.72669397,
                            192.77694368,
                            194.81256464,
                            196.77946938,
                            198.63884826,
                            200.36049923,
                            201.91933715,
                            203.29344838,
                            204.46297272,
                            205.4094678,
                            206.11557886,
                            206.5649173,
                            206.74209161,
                            206.63285507,
                            206.22434411,
                            205.50538541,
                            204.46684978,
                            204.27503107,
                            204.86675943,
                            205.82563562,
                            206.97718943,
                            208.23025133,
                            209.53028308,
                            210.84139375,
                            212.13816053,
                            213.40145846,
                            214.61614795,
                            215.76970766,
                            216.85138556,
                            217.8516515,
                            218.7618351,
                            219.57388266,
                            220.28019433,
                            220.87351752,
                            221.34688182,
                            221.69356539,
                            221.90708653,
                            222.12041429,
                            222.5203897,
                            223.04171662,
                            223.64498801,
                            224.30445627,
                            225.00227366,
                            225.72548147,
                            226.46430799,
                            227.2111462,
                            227.95990864,
                            228.70560381,
                            229.44404825,
                            230.17166538,
                            230.88534127,
                            231.58231911,
                            232.26012053,
                            232.91648597,
                            233.54932897,
                            234.15670055,
                            234.73676134,
                            235.17073651,
                            235.64887415,
                            236.16005492,
                            236.69610036,
                            237.25080205,
                            237.81932095,
                            238.39779936,
                            238.98310092,
                            239.57263101,
                            240.16420945,
                            240.75597809,
                            241.34633247,
                            241.93387028,
                            242.51735197,
                            243.09567015,
                            243.66782556,
                            244.23290793,
                            244.79008059,
                            245.33856805,
                            245.87764575,
                            246.38930418,
                            246.92126419,
                            247.46857986,
                            248.02739948,
                            248.59465651,
                            249.16786215,
                            249.74496203,
                            250.32423449,
                            250.90421668,
                            251.48364995,
                            252.06143857,
                            252.63661821,
                            253.20833142,
                            253.77580837,
                            254.33835152,
                            254.89532328,
                            255.446136,
                            255.99024373,
                            256.52713552,
                            257.05632969,
                            257.69919141,
                            258.35815326,
                            259.02816474,
                            259.70531807,
                            260.38651274,
                            261.06923417,
                            261.75140259,
                            262.43126705,
                            263.10732919,
                            263.77828722,
                            264.44299403,
                            265.10042522,
                            265.7496544,
                            266.38983392,
                            267.02017941,
                            267.6399575,
                            268.2484758,
                            268.84507469,
                            269.42912052,
                            270.0,
                        ]
                    ]
                ),
                decimal=6,
            )

            np.testing.assert_almost_equal(float(sol.cost), 3.719442753142962e-05)
            np.testing.assert_almost_equal(sol_time[-1], 0.5381208818685651)


def test_single_phase_time_dependent_ocp_output():
    ocp = OcpFes().prepare_ocp(
        model=DingModelFrequencyWithFatigue(), n_stim=1, n_shooting=10, final_time=0.1, use_sx=True,
    )

    sol = ocp.solve(Solver.IPOPT(show_online_optim=False, _max_iter=1000))
    sol_states = sol.decision_states(to_merge=[SolutionMerge.PHASES, SolutionMerge.NODES])
    np.testing.assert_almost_equal(sol.parameters["pulse_apparition_time"], np.array([0.0]))

    np.testing.assert_almost_equal(
        sol_states["F"],
        np.array(
            [
                [
                    0.0,
                    15.85448036,
                    36.35382298,
                    54.98028518,
                    70.84149988,
                    83.4707505,
                    92.52389842,
                    97.78058713,
                    99.22489701,
                    97.12756918,
                    92.06532562,
                ]
            ]
        ),
    )

    np.testing.assert_almost_equal(float(sol.cost), 0.0)
