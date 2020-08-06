
from functools import partial
import autograd.numpy as np
from autograd import value_and_grad
from autograd.scipy.misc import logsumexp
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from envs.gridworld import GridworldEnv
from envs.windy_gridworld import WindyGridworldEnv
import os

import gym

def build_twostate_MDP():
    """
    MDP with transition probabilities
    P(s_0 | s_0, a_0) = 0.5
    P(s_1 | s_0, a_0) = 0.5
    P(s_0 | s_0, a_1) = 0
    P(s_1 | s_0, a_1) = 1
    P(s_1 | s_0, a_2) = 0
    P(s_1 | s_1, a_2) = 1
    Rewards: r(s_0, a_0) = 5, r(s_0, a_1) = 10, r(s_1, a_2) = -1
    Discount factor : 0.95
    :return:
    """
    P = np.zeros((3, 2, 2))
    P[0, 0] = [0.5, 0.5]
    P[1, 0] = [0, 1]
    P[2, 0] = [1, 0]  # no op
    P[2, 1] = [0, 1]
    P[1, 1] = [0, 1]
    P[0, 1] = [0, 1]
    #T = {0: {0: [0.5, 0.5], 1: [0, 1]}, 1: {2: [0, 1]}}
    gamma = 0.9
    R = np.zeros((2, 3))
    R[0, 0] = 5
    R[0, 1] = 10
    R[1, 2] = -1

    initial_distribution = np.array([0.5,0.5])

    return P, R, gamma, initial_distribution


def mdp_imani_counterexample():
    """
    MDP counter example given in Fig 1a of Imani, et al.
    "An Off-policy Policy Gradient Theorem Using Emphatic Weightings."
    Neurips 2018
    :return:
    """
    # |S| = 4, |A| = 2
    STATES = 4
    ACTIONS = 2
    P = np.zeros((ACTIONS, STATES, STATES))
    P[0, 0] = [0, 1, 0, 0]
    P[1, 0] = [0, 0, 1, 0]
    P[0, 1] = [0, 0, 0, 1]
    P[1, 1] = [0, 0, 0, 1]
    P[0, 2] = [0, 0, 0, 1]
    P[1, 2] = [0, 0, 0, 1]
    P[0, 3] = [0, 0, 0, 1]
    P[1, 3] = [0, 0, 0, 1]

    gamma = 0.9

    R = np.zeros((STATES, ACTIONS))
    R[0, 0] = 0
    R[0, 1] = 1
    R[1, 0] = 2
    R[1, 1] = 0
    R[2, 0] = 0
    R[2, 1] = 1

    initial_distribution =np.array([1, 0, 0, 0])

    return P, R, gamma, initial_distribution

def mdp_fig2d():
  """ Figure 2 d) of
  ''The Value Function Polytope in Reinforcement Learning''
  by Dadashi et al. (2019) https://arxiv.org/abs/1901.11524
  """
  P = np.array([[[0.7 , 0.3 ], [0.2 , 0.8 ]],
              [[0.99, 0.01], [0.99, 0.01]]])
  R = np.array(([[-0.45, -0.1 ],
               [ 0.5 ,  0.5 ]]))
  initial_distribution = np.ones(P.shape[-1])/P.shape[-1]
  return P, R, 0.9, initial_distribution

def build_gridworld():
    env = GridworldEnv()
    P = np.zeros((env.action_space.n, env.observation_space.n, env.observation_space.n))
    R = np.zeros((env.observation_space.n, env.action_space.n))
    for s in range(env.observation_space.n):
        for a in range(env.action_space.n):
            for p, ns, r, _ in env.P[s][a]:
                P[a, s, ns] += p
                R[s, a] += p*r
    initial_distribution = np.ones(env.observation_space.n)/np.sum(np.ones(env.observation_space.n))
    gamma = 0.9
    return P, R, gamma, initial_distribution


def build_windy_gridworld():
    env = WindyGridworldEnv()
    P = np.zeros((env.action_space.n, env.observation_space.n, env.observation_space.n))
    R = np.zeros((env.observation_space.n, env.action_space.n))
    for s in range(env.observation_space.n):
        for a in range(env.action_space.n):
            for p, ns, r, _ in env.P[s][a]:
                P[a, s, ns] += p
                R[s, a] += p*r
    initial_distribution = np.ones(env.observation_space.n)/np.sum(np.ones(env.observation_space.n))
    gamma = 0.9
    return P, R, gamma, initial_distribution

def build_FrozenLake():
    env  = gym.make("FrozenLake-v0")
    P = np.zeros((env.action_space.n, env.observation_space.n, env.observation_space.n))
    R = np.zeros((env.observation_space.n, env.action_space.n))
    for s in range(env.observation_space.n):
        for a in range(env.action_space.n):
            for p, ns, r, _ in env.env.P[s][a]:
                P[a, s, ns] += p
                R[s, a] += p*r
    initial_distribution = np.zeros(env.observation_space.n)
    initial_distribution[0] = 1
    gamma=0.9
    return P, R, gamma, initial_distribution



def build_FrozenLake8():
    env  = gym.make("FrozenLake8x8-v0")
    P = np.zeros((env.action_space.n, env.observation_space.n, env.observation_space.n))
    R = np.zeros((env.observation_space.n, env.action_space.n))
    for s in range(env.observation_space.n):
        for a in range(env.action_space.n):
            for p, ns, r, _ in env.env.P[s][a]:
                P[a, s, ns] += p
                R[s, a] += p*r
    initial_distribution = np.zeros(env.observation_space.n)
    initial_distribution[0] = 1
    gamma=0.9
    return P, R, gamma, initial_distribution


def build_SB_example35():
    """
    Example 3.5 from (Sutton and Barto, 2018) pg 60 (March 2018 version).
    A rectangular Gridworld representation of size 5 x 5.
    Quotation from book:
    At each state, four actions are possible: north, south, east, and west, which deterministically
    cause the agent to move one cell in the respective direction on the grid. Actions that
    would take the agent off the grid leave its location unchanged, but also result in a reward
    of −1. Other actions result in a reward of 0, except those that move the agent out of the
    special states A and B. From state A, all four actions yield a reward of +10 and take the
    agent to A'. From state B, all actions yield a reward of +5 and take the agent to B'
    """
    size = 5
    P = build_simple_grid(size=size, p_success=1)
    P=P.reshape(4,25,-1) #reshaping to match the format accepted by the solver.
    # modify P to match dynamics from book.
    P[:, 1, :] = 0 # first set the probability of all actions from state 1 to zero
    P[:, 1, 21] = 1 # now set the probability of going from 1 to 21 with prob 1 for all actions

    P[:, 3, :] = 0  # first set the probability of all actions from state 3 to zero
    P[:, 3, 13] = 1  # now set the probability of going from 3 to 13 with prob 1 for all actions

    # TODO: add rewards for walking off the grid
    R = np.zeros((P.shape[1], P.shape[0])) # initialize a matrix of size |S|x|A|

    for state in range(P.shape[1]):
        for action in [actions.UP, actions.LEFT, actions.RIGHT, actions.DOWN]:
            if not check_can_take_action(action, state, size):
                R[state, action] = -1

    R[1, :] = +10
    R[3, :] = +5
    initial_distribution = np.ones(P.shape[1])/P.shape[1] # uniform starting probability (assumed)
    gamma = 0.9
    return P, R, gamma, initial_distribution


def build_CliffWalking():
    env  = gym.make("CliffWalking-v0")
    P = np.zeros((env.action_space.n, env.observation_space.n, env.observation_space.n))
    R = np.zeros((env.observation_space.n, env.action_space.n))
    for s in range(env.observation_space.n):
        for a in range(env.action_space.n):
            for p, ns, r, _ in env.P[s][a]:
                P[a, s, ns] += p
                R[s, a] += p*r
    initial_distribution = np.zeros(env.observation_space.n)
    initial_distribution[0] = 1
    gamma=0.9
    return P, R, gamma, initial_distribution
