"""
This example will do a 10 stimulation example with Ding's 2003 frequency model.
This ocp was build to match a force value of 270N at the end of the last node.
No optimization will be done on the stimulation, the frequency is fixed to 1Hz.
"""

from cocofest import OcpFes, ModelMaker


def prepare_ocp():
    # --- Build ocp --- #
    # This ocp was build to match a force value of 270N at the end of the last node.
    # The stimulation will be optimized between 0.01 to 0.1 seconds and are equally spaced (a fixed frequency).
    model = ModelMaker.create_model("ding2003", stim_time=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    return OcpFes().prepare_ocp(
        model=model,
        final_time=1,
        objective={"end_node_tracking": 270},
        use_sx=True,
    )


def main():
    ocp = prepare_ocp()
    sol = ocp.solve()

    # --- Show results from solution --- #
    sol.graphs(show_bounds=True)


if __name__ == "__main__":
    main()
