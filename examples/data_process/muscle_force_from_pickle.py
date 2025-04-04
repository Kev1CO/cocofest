import pickle
import matplotlib.pyplot as plt


class MuscleForceFromPickle:
    def __init__(self, pickle_path: str = None, muscle_name: str = None):
        if pickle_path is None:
            raise ValueError("Please provide a path to the pickle file(s).")
        if not isinstance(pickle_path, str) :
            raise TypeError("Please provide a str type path.")
        if not isinstance(muscle_name, str) and not isinstance(muscle_name, list):
            raise TypeError("Please provide a str type path.")

        self.path = pickle_path
        self.muscle_name = muscle_name

        self.time, self.muscle_force, self.stim_time = self.read_pkl_to_force()

    def read_pkl_to_force(self):
        with open(self.path, "rb") as f:
            data = pickle.load(f)
        return data["time"], data[self.muscle_name], data["stim_time"]


if __name__ == "__main__":
    muscle_force = MuscleForceFromPickle(pickle_path="essai1_florine_force_biceps.pkl_0.pkl", muscle_name="BIC_long")

    plt.plot(muscle_force.time, muscle_force.muscle_force)
    plt.title("Muscle Force")
    plt.xlabel("Time (s)")
    plt.ylabel("Force (N)")
    plt.show()