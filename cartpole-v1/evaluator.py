import os
import time
import sys

import argparse
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


def play():
    """Evaluate an agent playing CartPole-v1
    """
    log = get_logger(__name__)
    
    # Building environment & reseting to init state
    log.info('Building environment')
    env = gym.make('CartPole-v1')
    state = env.reset()

    # # Creating the agent
    # log.info('Creating agent')
    # N_states = env.observation_space.n
    # N_actions = env.action_space.n
    # agent = QLearningAgent(state, N_states, N_actions)

    # # Loading agent weights
    # log.info('Loading agent weights: %s', weights)
    # load_weights(agent, weights)

    # Reseting environment for new episode
    state = env.reset()
    ended = False
    iteration = 0 

    while not ended:
        # Playing
        action = env.action_space.sample()
        env.step(action)

        # action = agent(state)
        # state, reward, ended, info = env.step(action)
        # agent.update(state, reward)

        # Rendering
        # render(env, iteration, reward, time_delta=0.2)
        render(env, iteration, 0, time_delta=0.01)

        # Updating iterations
        iteration += 1


if __name__== '__main__':
    # # Parse args
    # description = 'Taxi-v3 agent evaluator'
    # parser = argparse.ArgumentParser(description=description)
    # parser.add_argument('weights', 
    #                     help='Path to weigths to load for the agent')
    # args = parser.parse_args()

    # # Playing
    # play(weights=args.weights)
    play()