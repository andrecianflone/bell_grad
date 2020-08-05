import numpy as np
import copy
from gym import error, spaces, utils
from gym.utils import seeding
from gym.envs.registration import register
import gym

class twostateMDP(gym.Env):
	def __init__(self):
		self.start_state_prob=0.5*np.ones(2)
		self.start_state=np.random.multinomial(1,self.start_state_prob).argmax()
		self.current_state=copy.deepcopy(self.start_state)
		self.name="2stateMDP"
		self.discount=0.9
		self.action_space=spaces.Discrete(3)
		self.observation_space=spaces.Discrete(2)
		self.probs_state_actions= np.zeros([2,2,3])
		self.probs_state_actions[0,0,0]=0.5
		self.probs_state_actions[1,0,0]=0.5
		self.probs_state_actions[0,0,1]=0
		self.probs_state_actions[1,0,1]=1
		self.probs_state_actions[1,0,2]=0
		self.probs_state_actions[1,1,2]=1

	def step(self,action):

		
		next_state = np.random.multinomial(1,self.probs_state_actions[:,self.current_state,action]).argmax()

		if action==0 and self.current_state==0:
			reward = 5
		if action==1 and self.current_state==0:
			reward = 10
		if action==2 and self.current_state==1:
			reward=-1
		else:
			reward=0

		self.current_state=next_state

		return next_state,reward,False,{}


	def reset(self):
		self.current_state=copy.deepcopy(self.start_state)
		return self.current_state

register(
    id='twostateMDP-v0',
    entry_point='envs.twostateMDP:twostateMDP',
    timestep_limit=300,
    
)

