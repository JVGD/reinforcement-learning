import sys

import argparse
import gym
import torch as T
from tqdm import tqdm
from IPython import embed

sys.path.append('.')
from rl.utils import get_logger
from rl.metrics import Recorder
from rl.weights import format_weights_name, save_weights
from rl.agents.qlearning import QNetworkAgent


def train(episodes=3000, log_every=100, log_dir='cartpole-v1'):
    """Trains an agent to play CartPole-v1
    """
    log = get_logger(__name__)
    
    # Building environment & reseting to init state
    log.info('Building environment')
    env = gym.make('CartPole-v1')
    init_state = env.reset()
    
    # Creating the agent
    log.info('Creating agent')
    N_actions = env.action_space.n
    agent = QNetworkAgent(init_state, init_state.size, N_actions)

    # Metrics for storing in tensorboard every log_every episodes
    metrics = Recorder(log_dir=log_dir, skip_steps=log_every)
    log.info('Logs in: %s', metrics.log_dir)

    # Training loop
    log.info('Start training')
    progress_bar = tqdm(range(episodes), unit='episode')

    for episode in progress_bar:

        # Reseting environment for new episode
        state = env.reset()
        ended = False

        while not ended:
            # Playing
            action = agent(state)
            state, reward, ended, info = env.step(action)
            agent.update(state, reward)
            
            # Metrics
            metrics.record(reward)

        # Logging
        episode_reward = metrics.log(episode)
        progress_bar.set_description('Episode reward: %.3f' % episode_reward)
    
    # Saving weights after training
    weights = format_weights_name(episode, episode_reward, 'taxiv3')
    weights_path = save_weights(agent, weights, path=log_dir)
    log.info('Saved weights: %s', weights_path)


if __name__ == '__main__':
    train()