import os
import numpy as np

import matplotlib.pyplot as plt
from geo_fab.envs import add_obstacle

def solve_euler(q, dq, dt):
    return q + dq * dt


class PointEnv():
    """
    Particle Simple Environment.
    """
    def __init__(self, reward_type=0, time_step=1 / 240., dynamics=True, n_obstacles=0):
        ## Particle params
        self.fq = 1/time_step
        self.n_substeps = 1.
        self.T = 100
        self.qlimits = np.array([[0., 10.], [0., 10.]])
        ## Obstacles
        self.n_obstacles = n_obstacles
        self._set_obstacles(n_obstacles)
        ## Robot Initial State
        self.q_home = np.array([0., 0.])
        self.q_0 = np.array([0., 0.])
        self.v_0 = np.zeros(2)
        self.trj = self.q_0[None,:]

    @property
    def dt(self):
        return 1 / self.fq * self.n_substeps

    def reset(self, q0=None):
        ## Set Obstacles
        self._set_obstacles(self.n_obstacles)
        ## Initialize Particle Environment
        self.q_0 = np.random.rand(2)*self.qlimits[0][1]
        self.q_0 = np.zeros(2)
        self.v_0 = np.zeros(2)

        self.trj = self.q_0[None,:]
        self._visualize_trj()
        return self._get_observations()

    def step(self, action, vel=True):
        ## Compute Linear Dynamics
        if vel:
            v_1 = action
            q_1 = self.q_0 + action*self.dt
        else:
            v_1 = self.v_0 + action*self.dt
            q_1 = self.q_0 + self.v_0*self.dt + 0.5*(action**2)*(self.dt**2)

        ## Bound q_1##
        for i in range(q_1.shape[0]):
            q_1[i] = np.clip(q_1[i], self.qlimits[i, 0], self.qlimits[i, 1])

        ## Add element to trajectory
        self.trj = np.concatenate((self.trj, q_1[None,:]), 0)
        self.trj = self.trj[-self.T:,:]

        ## Set state
        self.q_0 = q_1
        self.v_0 = v_1

        ## Visualize trj
        self._visualize_trj()
        return self._get_observations()

    def _get_observations(self):
        return self.q_0, self.v_0

    def _set_obstacles(self, n_obstacles):
        # self.obstacles = np.zeros((0,3))
        # self.obstacles = np.array([[2., 1.8, 1.],[2., 3., .7], [2., 4., 1.2],[1.8, 6., 1.2],
        #                            [3., 1.8, 1.],[4., 1.8, 1.],])

        self.obstacles = np.array([[5., 5., 1.],])

        for i in range(n_obstacles):
            obs = np.random.rand(2)*self.qlimits[0][1]
            obs_size = np.random.rand()*1. +0.3
            obs_full = np.concatenate((obs, np.array([obs_size])),0)
            self.obstacles = np.concatenate((self.obstacles,obs_full[None,:]),0)

    def _visualize_trj(self):
        plt.clf()
        fig, ax = plt.subplots(num=1, figsize=(12, 6))

        ## visualize obstacles
        for i in range(self.obstacles.shape[0]):
            obs = self.obstacles[i,...]
            c = plt.Circle((obs[0], obs[1]), obs[2], color='r')
            ax.add_patch(c)

        ## visualize trajectories
        ax.plot(self.trj[:,0], self.trj[:,1], linewidth=4.)
        ax.scatter(self.trj[-1,0], self.trj[-1,1])

        plt.xlim(self.qlimits[0])
        plt.ylim(self.qlimits[1])
        plt.draw()
        plt.pause(0.000001)

