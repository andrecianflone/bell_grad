import numpy as np
import random
import os
import time
import json

create_folder = lambda f: [os.makedirs(f) if not os.path.exists(f) else False]
class Logger(object):
    def __init__(self, experiment_name='', environment_name='',
                                                      folder='./results' ):
        """
        Saves experimental metrics for use later.
        :param experiment_name: name of the experiment
        :param folder: location to save data
        : param environment_name: name of the environment
        """
        self.rewards = []
        filename = time.strftime('%Y%m%d-%H-%M')
        save_folder = os.path.join(folder, experiment_name,
                    environment_name, filename)
        self.save_folder = self.check_folder(save_folder)

    def check_folder(self, base, path=None, count=0):
        path = base if path is None else path
        count += 1
        # If base folder exists, incr
        if os.path.exists(path):
            return self.check_folder(base, f'{base}_{count}', count)
        else:
            # If run in parallel, possibly another process simultaneously locks
            try:
                os.makedirs(path)
            except:
                return self.check_folder(base, f'{base}_{count}', count)
            return path

    def record_reward(self, reward_return):
        self.returns_eval = reward_return

    def training_record_reward(self, reward_return):
        self.returns_train = reward_return

    def save(self):
        np.save(os.path.join(self.save_folder, "returns_eval.npy"), self.returns_eval)

    def save_2(self, returns_PGT):
        np.save(os.path.join(self.save_folder, "all_returns.npy"), returns_PGT)

    def save_args(self, args):
        """
        Save the command line arguments
        """
        with open(os.path.join(self.save_folder, 'params.json'), 'w') as f:
              json.dump(dict(args._get_kwargs()), f)

def get_env():
    if args.env == 'GridWorld':
        from envs.gridworld import GridworldEnv
        env = GridworldEnv()
        eval_env = GridworldEnv()
        mdp = environments.build_gridworld()

    elif args.env == 'WindyGridWorld':
        from envs.windy_gridworld import WindyGridworldEnv
        env = WindyGridworldEnv()
        eval_env = WindyGridworldEnv()
        mdp = environments.build_windy_gridworld()

    elif args.env == 'CliffWalking':
        env = gym.make("CliffWalking-v0")
        eval_env = gym.make("CliffWalking-v0")
    elif args.env == 'FrozenLake':
        env = gym.make("FrozenLake-v0")
        eval_env = gym.make("FrozenLake-v0")
        mdp = environments.build_FrozenLake()

    elif args.env == 'FrozenLake8':
        env = gym.make("FrozenLake8x8-v0")
        eval_env = gym.make("FrozenLake8x8-v0")
    elif args.env == 'Taxi':
        env = gym.make("Taxi-v2")
        eval_env = gym.make("Taxi-v2")
    elif args.env=='twostateMDP':
        from envs.twostateMDP import twostateMDP
        env = gym.make('twostateMDP-v0')
        eval_env=gym.make('twostateMDP-v0')
        mdp = environments.mdp_fig2d()
        args.env = environments.mdp_fig2d
    return env, eval_env, mdp

def one_hot_ify(state, num_states):
    res = torch.zeros(1, num_states)
    res[0,state] = 1
    return res

def softmax(vals, temp=1.):
    """Batch softmax
    Args:
        vals (np.ndarray): S x A. Applied row-wise
        t (float, optional): Defaults to 1.. Temperature parameter
    Returns:
        np.ndarray: S x A
    """
    return np.exp(  (1./temp) * vals - logsumexp(  (1./temp) * vals, axis=1, keepdims=True) )

if __name__ == "__main__":
    logger = Logger(experiment_name="test", environment_name="test_env")
    logger.save_args(args)


