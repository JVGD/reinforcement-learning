import sys
import logging

import gym
import torch as T
from tqdm import tqdm

sys.path.append('.')
from rl.utils import get_logger
from rl.weights import format_weights_name, save_weights
from rl.agents.qlearning import QLearningAgent


def train(episodes=1000):
    """Trains an agent to play Taxi-v3
    """
    log = get_logger(__name__)
    
    # Building environment & reseting to init state
    log.warning('Building environment')
    env = gym.make('Taxi-v3')
    init_state = env.reset()
    state = init_state

    # Creating the agent
    log.info('Creating agent')
    N_states = env.observation_space.n
    N_actions = env.action_space.n
    agent = QLearningAgent(init_state, N_states, N_actions)

    # Training loop
    log.info('Start training')
    progress_bar = tqdm(range(episodes), unit='episode')

    for episode in progress_bar:

        # Reseting environment for new episode
        state = env.reset()
        ended = False
        rewards = []

        while not ended:
            # Playing
            action = agent(state)
            state, reward, ended, info = env.step(action)
            agent.update(state, reward)
            
            # Metrics
            rewards.append(reward)

        # Logging
        rewards_avg = T.tensor(rewards, dtype=T.float).mean().item()
        progress_bar.set_description('Episode reward: %.3f' % rewards_avg)
    
    # Saving weights after training
    weights = format_weights_name(episode, rewards_avg, 'taxiv3')
    save_weights(agent, weights)
    log.info('Saved weights: %s', weights)
    

if __name__== '__main__':
    train()
