import numpy as np
import copy
from gym import error, spaces, utils
from gym.utils import seeding
from gym.envs.registration import register
import gym

class counterexampleMDP(gym.Env):
	def __init__(self):
		
		self.start_state=0
		self.current_state=copy.deepcopy(self.start_state)
		self.name="2stateMDP"
		self.discount=0.9
		self.action_space=spaces.Discrete(2)
		self.observation_space=spaces.Discrete(4)
		self.P = np.zeros((2,4,4))
		self.P[0, 0] = [0, 1, 0, 0]
		self.P[1, 0] = [0, 0, 1, 0]
		self.P[0, 1] = [0, 0, 0, 1]
		self.P[1, 1] = [0, 0, 0, 1]
		self.P[0, 2] = [0, 0, 0, 1]
		self.P[1, 2] = [0, 0, 0, 1]
		self.P[0, 3] = [0, 0, 0, 1]
		self.P[1, 3] = [0, 0, 0, 1]

		self.R = np.zeros((4, 2))
		self.R[0, 0] = 0
		self.R[0, 1] = 1
		self.R[1, 0] = 2
		self.R[1, 1] = 0
		self.R[2, 0] = 0
		self.R[2, 1] = 1
	def step(self,action):

		
		next_state = np.argmax(self.P[action,self.current_state])

		reward = self.R[self.current_state,action]

		self.current_state = next_state

		if self.current_state==3:
			done=True
		else:
			done=False

		return next_state,reward,done,{}


	def reset(self):
		self.current_state=copy.deepcopy(self.start_state)
		return self.current_state

register(
    id='counterexampleMDP-v0',
    entry_point='envs.counterexample:counterexampleMDP',
    timestep_limit=300,
    
)

