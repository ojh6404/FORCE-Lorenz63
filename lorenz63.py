import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from utils import plot_3d


class Lorenz63():
    def __init__(self, dt=0.02, T=125, init_state=[0.1, 0.1, 0.1], rho=28., sigma=10., beta=(8./3.)):
        self.rho = rho
        self.sigma = sigma
        self.beta = beta
        self.dt = dt
        self.T = T
        self.t = np.arange(0, T, self.dt)
        self.init_state = init_state
        self.trajectory = self.generate_trajectory(self.init_state, self.T)
        self.num_points = len(self.trajectory)

    def generate_trajectory(self, init_state, T):
        def f(state, t):
            x, y, z = state
            return self.sigma * (y - x), x * (self.rho - z) - y, x * y - self.beta * z
        trajectory = odeint(f, init_state, self.t)
        self.trajectory = np.expand_dims(trajectory, axis=-1)
        return self.trajectory


def main():
    lorenz63 = Lorenz63(dt=0.02)
    trajectory = lorenz63.trajectory
    plot_3d(trajectory)


if __name__ == "__main__":
    main()
