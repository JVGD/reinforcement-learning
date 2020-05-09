import os
import time
import sys

import gym
import torch as T
from tqdm import tqdm

sys.path.append('.')
from rl.utils import get_logger
from rl.metrics import Recorder
from rl.weights import format_weights_name, load_weights
from rl.agents.qlearning import QLearningAgent


def render(environment, n, reward, time_delta=0.2):
    """Helps the rendering in terminal
    """
    # Clearing terminal
    os.system('clear')

    # Logging & rendering
    print('Iteration: {} Step reward: {:.2f}'.format(n, reward))
    environment.render()

    # Sleeping for a while to see evolution
    time.sleep(time_delta)


def play(weights):
    """Evaluate an agent playing Taxi-v3
    """
    log = get_logger(__name__)
    
    # Building environment & reseting to init state
    log.info('Building environment')
    env = gym.make('Taxi-v3')
    state = env.reset()

    # Creating the agent
    log.info('Creating agent')
    N_states = env.observation_space.n
    N_actions = env.action_space.n
    agent = QLearningAgent(state, N_states, N_actions)

    # Loading agent weights
    log.info('Loading agent weights: %s', weights)
    load_weights(agent, weights)

    # Reseting environment for new episode
    state = env.reset()
    ended = False
    iteration = 0 

    while not ended:
        # Playing
        action = agent(state)
        state, reward, ended, info = env.step(action)
        agent.update(state, reward)

        # Rendering
        render(env, iteration, reward, time_delta=0.2)

        # Updating iterations
        iteration += 1


if __name__== '__main__':
    play('episode9999_reward-6.00_taxiv3.pt')
