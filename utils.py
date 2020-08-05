import numpy as np
import random
import os
import time
import json
import torch
import torch.nn as nn

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

if __name__ == "__main__":
    logger = Logger(experiment_name="test", environment_name="test_env")
    logger.save_args(args)

