import sys
import torch as T
from torch.utils.tensorboard import SummaryWriter

from rl.utils import mkdir


class Recorder(SummaryWriter):
    """Class for recording data in Tensorboard
    """
    def __init__(self, log_dir='./logs', skip_steps=1):
        # Creating logging directory
        mkdir(log_dir)
        self.log_dir = log_dir

        # Getting summary writer in log_dir
        super().__init__(log_dir=log_dir)

        # Vars to record info
        self.data = []

        # Only record every skip steps
        self.skip_steps = skip_steps

    def record(self, reward):
        """Stores reward of an episode step

        Parameters
        ----------
        reward : float
            Reward of episode step
        """
        self.data.append(reward)

    def log(self, episode):
        """Save average episode reward into tensorboard

        Parameters
        ----------
        episode : int
            Episode number
        """
        # Averaging episode rewards
        rewards = T.tensor(self.data, dtype=T.float)
        rewards_avg = rewards.mean().item()

        # Logging to tensorboard only every skip_steps
        if episode % self.skip_steps == 0:
            self.add_scalar('train/reward', rewards_avg, episode)

        # Reseting episode rewards
        self.data = []

        return rewards_avg



    


