from scipy import integrate
import matplotlib.pyplot as plt

def dynamics(t, q):
    pass

sol = integrate.solve_ivp(dynamics, [0, 1], [90, 0], method="RK45")

plt.plot(sol.t, sol.y[0], label='Angle (rad)')
plt.plot(sol.t, sol.y[1], label='Angular velocity (rad/s)')
plt.xlabel('Time (s)')
plt.ylabel('State variables')
plt.legend()
plt.show()