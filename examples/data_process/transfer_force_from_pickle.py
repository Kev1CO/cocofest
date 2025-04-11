import numpy as np
import matplotlib.pyplot as plt
import pickle
from itertools import chain

from biorbd import Model


class TransferForceFromPickle:
    def __init__(
        self,
        model_path: str = None,
        in_pickle_path: str | list[str] = None,
        out_pickle_path: str | list[str] = None,
        elbow_angle: int | float | list[int] | list[float] = 90,
        muscle_name: str | list[str] = None,
        dof_name: str = None,
    ):

        if model_path is None:
            raise ValueError("Please provide a path to the model.")
        if not isinstance(model_path, str):
            raise TypeError("Please provide a str type model path.")

        self.model_path = model_path

        if in_pickle_path is None:
            raise ValueError("Please provide a path to the pickle file(s).")
        if not isinstance(in_pickle_path, str) and not isinstance(in_pickle_path, list):
            raise TypeError(
                "Please provide a pickle_path list of str type or a str type path."
            )
        if not isinstance(out_pickle_path, str) and not isinstance(
            out_pickle_path, list
        ):
            raise TypeError(
                "Please provide a out_pickle_path list of str type or a str type path."
            )
        if out_pickle_path is not None:
            if isinstance(out_pickle_path, str):
                out_pickle_path = [out_pickle_path]
            if len(out_pickle_path) != 1:
                if len(out_pickle_path) != len(in_pickle_path):
                    raise ValueError(
                        "If not str type, out_pickle_path must be the same length as pickle_path."
                    )

        self.path = in_pickle_path
        self.time = None
        self.stim_time = None
        self.local_data = None
        self.model = None
        self.Q = None
        self.Qdot = None
        self.Qddot = None
        self.dof_name = dof_name
        self.muscle_name = muscle_name
        self.muscle_moment_arm = None
        self.muscle_force_vector = None

        # Load the model
        self.load_model(elbow_angle)

        # Saving muscle/dof names and indexes as dict
        self.muscle_name_index = {}
        for i in range(len(self.model.muscleNames())):
            self.muscle_name_index[self.model.muscleNames()[i].to_string()] = i

        self.dof_name_index = {}
        for i in range(len(self.model.nameDof())):
            self.dof_name_index[self.model.nameDof()[i].to_string()] = i

        if self.muscle_name not in self.muscle_name_index.keys():
            raise ValueError(
                f"Please provide a muscle name in the muscle_index dictionary : {list(self.muscle_name_index.keys())}."
            )

        if self.dof_name not in self.dof_name_index.keys():
            raise ValueError(
                f"Please provide a dof name in the dof_index dictionary : {list(self.dof_name_index.keys())}."
            )

        in_pickle_path_list = (
            [in_pickle_path] if isinstance(in_pickle_path, str) else in_pickle_path
        )

        for i in range(len(in_pickle_path_list)):
            force_data, torque_data = self.read_pkl_to_force_vector(
                in_pickle_path_list[i]
            )

            plt.plot(self.time, force_data[0], label="force x", color="blue")
            plt.plot(self.time, force_data[1], label="force y", color="orange")
            plt.plot(self.time, force_data[2], label="force z", color="green")
            plt.legend()
            plt.show()

            local_force_data = self.local_sensor_to_local_hand(force_data)

            plt.plot(self.time, local_force_data[0], label="force x", color="blue")
            plt.legend()
            plt.show()

            local_torque_data = self.local_sensor_to_local_hand(torque_data)
            self.local_data = np.concatenate((local_force_data, local_torque_data))
            self.select_muscle_and_dof()
            self.get_muscle_force(local_data=self.local_data)

            self.muscle_force_vector = (
                np.array(self.muscle_force_vector) - self.muscle_force_vector[0]
            )

            plt.plot(
                self.time, self.muscle_force_vector, label="muscle force", color="red"
            )
            plt.legend()
            plt.show()

            if out_pickle_path:
                if len(out_pickle_path) == 1:
                    if out_pickle_path[:-4] == ".pkl":
                        save_pickle_path = out_pickle_path[:-4] + "_" + str(i) + ".pkl"
                    else:
                        save_pickle_path = out_pickle_path[0] + "_" + str(i) + ".pkl"
                else:
                    save_pickle_path = out_pickle_path[i]

                dictionary = {
                    "time": self.time,
                    self.muscle_name: self.muscle_force_vector,
                    "stim_time": self.stim_time,
                }
                with open(save_pickle_path, "wb") as file:
                    pickle.dump(dictionary, file)

    def read_pkl_to_force_vector(self, in_pickle_path):
        """
        This function is used to read the pickle file and extract the force and torque data.
        Parameters
        ----------
        in_pickle_path: str
            The path to the pickle file

        Returns
        -------
        force_data: np.array
            The force data in the local muscle axis
        torque_data: np.array
            The torque data in the local muscle axis
        """
        if isinstance(in_pickle_path, str):
            # pickle_path = in_pickle_path[:-4] + ".pkl_0" + in_pickle_path[-4:]
            with open(in_pickle_path, "rb") as f:
                data = pickle.load(f)
                x = list(chain.from_iterable(data["x"]))
                y = list(chain.from_iterable(data["y"]))
                z = list(chain.from_iterable(data["z"]))
                mx = list(chain.from_iterable(data["mx"]))
                my = list(chain.from_iterable(data["my"]))
                mz = list(chain.from_iterable(data["mz"]))

                force_data = np.array([x, y, z])
                torque_data = np.array([mx, my, mz])

                self.time = list(chain.from_iterable(data["time"]))
                self.stim_time = data["stim_time"]
        else:
            raise ValueError("pickle_path must be a string")

        return force_data, torque_data

    @staticmethod
    def local_sensor_to_local_hand(sensor_data: np.array) -> np.array:
        """
        This function is used to convert the sensor data from the local axis to the local hand axis.
        This a rotation along x axis whatever the elbow angle
        Parameters
        ----------
        sensor_data: np.array
            The sensor data in the local axis

        Returns
        -------
        new_sensor_data: np.array
            The sensor data in the local hand axis
        """
        rotation_angle = np.pi / 2
        rotation_matrix = np.array(
            [
                [1, 0, 0],
                [0, np.cos(rotation_angle), -np.sin(rotation_angle)],
                [0, np.sin(rotation_angle), np.cos(rotation_angle)],
            ]
        )
        new_sensor_data = rotation_matrix @ sensor_data

        return new_sensor_data

    @staticmethod
    def local_hand_to_local(force_data: np.array, elbow_angle: float) -> np.array:
        """
        This function is used to convert the sensor data from the local hand axis to the local muscle axis.
        This is a rotation of elbow angle along z axis
        Parameters
        ----------
        force_data: np.array
            The sensor data in the local hand axis
        elbow_angle: float
            Must be in radians

        Returns
        -------
        new_force_data: np.array
            The sensor data in the local muscle axis
        """
        rotation_angle = np.pi - np.radians(elbow_angle)
        rotation_matrix = np.array(
            [
                [np.cos(rotation_angle), -np.sin(rotation_angle), 0],
                [np.sin(rotation_angle), np.cos(rotation_angle), 0],
                [0, 0, 1],
            ]
        )
        new_force_data = rotation_matrix @ force_data

        return new_force_data

    def load_model(self, elbow_angle: int | float):
        """
        This function is used to load the model and set the initial position, velocity and acceleration.
        Parameters
        ----------
        elbow_angle: int | float
            The elbow angle in degrees. It must be between 0 and 180 degrees.
            If elbow_angle is a list, the function will use the first element of the list.
        """
        # Load a predefined model
        self.model = Model(self.model_path)  # TODO : relative path
        # Get number of q, qdot, qddot
        nq = self.model.nbQ()
        nqdot = self.model.nbQdot()
        nqddot = self.model.nbQddot()

        # Choose a position/velocity/acceleration to compute dynamics from
        if nq != 2:
            raise ValueError("The number of degrees of freedom has changed.")  # 0
        self.Q = np.array(
            [0.0, np.radians(elbow_angle)]
        )  # "0" arm along body and "1.57" 90° forearm position  |__.
        self.Qdot = np.zeros((nqdot,))  # speed null
        self.Qddot = np.zeros((nqddot,))  # acceleration null

    def select_muscle_and_dof(self):
        """
        This function is used to select the muscle and dof from the model.
        """
        dof_index = self.dof_name_index[self.dof_name]
        muscle_index = self.muscle_name_index[self.muscle_name]
        self.muscle_moment_arm = -self.model.musclesLengthJacobian(self.Q).to_array()[
            muscle_index
        ][dof_index]

    def get_muscle_force(self, local_data):
        """
        This function is used to compute the muscle force from the local data.
        Parameters
        ----------
        local_data: array
            Contains the force and torque data in the local muscle axis

        Returns
        -------
        The muscle force
        """
        self.muscle_force_vector = []
        for i in range(len(local_data[0])):
            spatial_vector = np.array(
                [
                    local_data[0][i],
                    local_data[1][i],
                    local_data[2][i],
                    local_data[3][i],
                    local_data[4][i],
                    local_data[5][i],
                ]
            )

            external_force = self.model.externalForceSet()
            external_force.addInSegmentReferenceFrame(
                segmentName="r_ulna_radius_hand",
                vector=spatial_vector,
                pointOfApplication=np.array([0, 0, 0]),
            )

            tau = self.model.InverseDynamics(
                self.Q, self.Qdot, self.Qddot, external_force
            ).to_array()
            dof_index = self.dof_name_index[self.dof_name]
            tau = tau[dof_index]
            muscle_force = tau / self.muscle_moment_arm
            self.muscle_force_vector.append(muscle_force)


if __name__ == "__main__":
    force_transfer = TransferForceFromPickle(
        model_path="C:\\Users\\flori_4ro0b8\\Documents\\Stage_S2M\\cocofest\\examples\\model_msk\\simplified_UL_Seth.bioMod",
        in_pickle_path="essai2_florine_50Hz_400us_15mA_TR3s01_alltrains.pkl_9.pkl",
        out_pickle_path="essai2_florine_force_biceps_alltrains.pkl",
        elbow_angle=90,
        muscle_name="BIC_long",
        dof_name="r_ulna_radius_hand_r_elbow_flex_RotX",
    )
