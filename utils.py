import numpy as np
import matplotlib.pyplot as plt


def plot_3d(traj):
    traj = traj.squeeze()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], linewidth=1, label='Lorenz63')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')
    ax.legend()
    plt.draw()
    plt.show()


def plot_xyz(traj):
    traj = traj.squeeze()
    fig = plt.figure()
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)
    ax1.plot(traj[:, 0])
    ax2.plot(traj[:, 1])
    ax3.plot(traj[:, 2])
    plt.draw()
    plt.show()
