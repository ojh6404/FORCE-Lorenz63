from lorenz63 import Lorenz63
import numpy as np
from network import Network
from utils import *
import matplotlib.pyplot as plt


def main():
    dt = 0.02

    lorenz63 = Lorenz63(dt=dt)
    network = Network(lorenz63=lorenz63)

    train_trajectory = network.train_trajectory
    target_trajectory = network.target_trajectory
    test_trajectory = network.test_trajectory

    num_train_steps = len(train_trajectory)
    num_test_steps = len(test_trajectory)

    z_save_train = []
    z_save_pred = []

    for i in range(num_train_steps):
        if i % 100 == 0:
            print(f"mode : train, step : {i}/{num_train_steps}")

        if i % 2 == 0:
            network.step(train_trajectory[i],
                         f=target_trajectory[i], train=True)
        else:
            network.step(train_trajectory[i])
        z_save_train.append(network.z)

    for i in range(num_test_steps):
        if i % 100 == 0:
            print(f"mode : test, step : {i}/{num_test_steps}")
        network.step(network.z)
        z_save_pred.append(network.z)

    z_train = np.array(z_save_train).squeeze()
    z_pred = np.array(z_save_pred).squeeze()
    z_test = test_trajectory.squeeze()

    fig3d = plt.figure()
    ax = fig3d.add_subplot(projection='3d')
    ax.plot(z_pred[:, 0], z_pred[:, 1], z_pred[:, 2],
            linewidth=1, color='r', label='Generated from Resorvoir')
    ax.plot(z_test[:, 0], z_test[:, 1], z_test[:, 2],
            linewidth=1, color='b', label='Lorenz63')
    ax.legend()
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')
    plt.draw()
    plt.show()

    t_test = np.arange(0, num_test_steps) * dt
    print(t_test)

    figxyz = plt.figure()
    ax1 = figxyz.add_subplot(3, 1, 1)
    ax1.plot(t_test, z_pred[:, 0], linewidth=1, color='r')
    ax1.plot(t_test, z_test[:, 0], linewidth=1, color='b')
    ax1.set_xlabel('time step')
    ax1.set_ylabel('$x$')
    ax2 = figxyz.add_subplot(3, 1, 2)
    ax2.plot(t_test, z_pred[:, 1], linewidth=1, color='r')
    ax2.plot(t_test, z_test[:, 1], linewidth=1, color='b')
    ax2.set_xlabel('time step')
    ax2.set_ylabel('$y$')
    ax3 = figxyz.add_subplot(3, 1, 3)
    ax3.plot(t_test, z_pred[:, 2], linewidth=1, color='r')
    ax3.plot(t_test, z_test[:, 2], linewidth=1, color='b')
    ax3.set_xlabel('time step')
    ax3.set_ylabel('$z$')

    plt.draw()
    plt.show()


if __name__ == "__main__":
    main()
