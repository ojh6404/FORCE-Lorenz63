import numpy as np
from scipy import sparse
from lorenz63 import Lorenz63
from utils import *


class Network():
    def __init__(self, lorenz63: Lorenz63, ratio=0.8, alpha=1., beta=1., tau=1., Nr=1000, Nz=3, g=1.5, pr=0.1, pz=1., seed=42):
        # def __init__(self, lorenz63: Lorenz63, alpha=10.0, tau=0.01, N_G=1000, g_GG=1.5, g_Gz=1.2, p_GG=0.025, p_z=1., seed=42, z_dim=3):
        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        self.Nr = Nr
        self.Nz = Nz
        self.g = g
        self.pr = pr
        self.pz = pz
        self.seed = seed

        self.dt = lorenz63.dt
        self.t = lorenz63.t
        self.T = lorenz63.T
        self.trajectory = lorenz63.trajectory
        self.ratio = ratio

        self.train_trajectory = self.trajectory[:int(
            len(self.trajectory) * self.ratio)]
        self.target_trajectory = self.trajectory[1:int(
            len(self.trajectory) * self.ratio) + 1]
        self.test_trajectory = self.trajectory[int(
            len(self.trajectory) * self.ratio):]

        np.random.seed(self.seed)

        self.Wr_std = 1. / np.sqrt(self.pr * self.Nr)
        self.Wo_std = 1. / np.sqrt(self.Nr)
        self.Wi_std = 1. / np.sqrt(self.Nz)

        self.Wr = self.Wr_std * sparse.random(self.Nr, self.Nr, density=pr,
                                              random_state=self.seed, data_rvs=np.random.randn).toarray()
        self.Wf = np.random.uniform(-1.0, 1.0, (self.Nr, self.Nz))
        self.Wo = self.beta * np.random.randn(self.Nr, self.Nz)
        self.Wi = self.beta * np.random.randn(self.Nr, self.Nz)

        self.r = 0.5 * np.random.randn(self.Nr, 1)
        self.z = 0.5 * np.random.randn(self.Nz, 1)

        self.P = np.eye(self.Nr) / self.alpha

    def step(self, u_in, f=None, train=False):
        dr = (self.dt / self.tau) * (-self.r + np.tanh(self.g *
                                                       self.Wr @ self.r + self.Wf @ self.z + self.Wi @ u_in))
        self.r += dr
        self.z = self.Wo.T @ self.r

        if (train is True) and (f is not None):
            Pr = self.P @ self.r
            rPr = self.r.T @ Pr
            k = Pr.T / (1. + rPr)
            self.P -= Pr @ k
            e_minus = self.z - f

            dWo = - np.outer(k, e_minus)
            self.Wo += dWo


def main():
    lorenz63 = Lorenz63(dt=0.02)
    network = Network(lorenz63=lorenz63)


if __name__ == "__main__":
    main()
