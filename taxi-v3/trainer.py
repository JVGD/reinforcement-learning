import sys

import gym
import torch as T
from tqdm import tqdm

sys.path.append('.')
from rl.utils import get_logger
from rl.metrics import Recorder
from rl.weights import format_weights_name, save_weights
from rl.agents.qlearning import QLearningAgent


def train(episodes=3000, log_every=100):
    """Trains an agent to play Taxi-v3
    """
    log = get_logger(__name__)
    
    # Building environment & reseting to init state
    log.info('Building environment')
    env = gym.make('Taxi-v3')
    init_state = env.reset()
    state = init_state

    # Creating the agent
    log.info('Creating agent')
    N_states = env.observation_space.n
    N_actions = env.action_space.n
    agent = QLearningAgent(init_state, N_states, N_actions)

    # Metrics for storing in tensorboard every log_every episodes
    metrics = Recorder(log_dir='taxi-v3', skip_steps=log_every)
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
    save_weights(agent, weights)
    log.info('Saved weights: %s', weights)
    

if __name__== '__main__':
    train()
