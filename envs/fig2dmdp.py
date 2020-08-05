import numpy as np
import copy
from gym import error, spaces, utils
from gym.utils import seeding
from gym.envs.registration import register
import gym

class fig2dMDP(gym.Env):
	def __init__(self):
		
		self.start_state=0
		self.current_state=copy.deepcopy(self.start_state)
		self.name="fig2dMDP"
		self.discount=0.9
		self.action_space=spaces.Discrete(2)
		self.observation_space=spaces.Discrete(2)
		self.P = np.array([[[0.7 , 0.3 ], [0.2 , 0.8 ]],
              [[0.99, 0.01], [0.99, 0.01]]])
		self.R = np.array(([[-0.45, -0.1 ],
               [ 0.5 ,  0.5 ]]))
	def step(self,action):

		
		next_state = np.random.multinomial(1,self.P[action,self.current_state]).argmax()

		reward = self.R[self.current_state,action]

		self.current_state = next_state

		

		return next_state,reward,False,{}


	def reset(self):
		self.current_state=copy.deepcopy(self.start_state)
		return self.current_state

register(
    id='fig2dMDP-v0',
    entry_point='envs.fig2dmdp:fig2dMDP',
    timestep_limit=300,
    
)

