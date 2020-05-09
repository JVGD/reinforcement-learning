import sys

import argparse
import gym
import torch as T
from tqdm import tqdm

sys.path.append('.')
from rl.utils import get_logger
from rl.metrics import Recorder
from rl.weights import format_weights_name, save_weights
from rl.agents.qlearning import QLearningAgent, SARSAAgent


def train(episodes=20000, log_every=100, log_dir='frozenlake-v0'):
    """Trains an agent to play Frozen Lake
    """
    log = get_logger(__name__)
    
    # Building environment & reseting to init state
    log.info('Building environment')
    env = gym.make('FrozenLake-v0')
    init_state = env.reset()
    state = init_state

    # Creating the agent
    log.info('Creating agent')
    N_states = env.observation_space.n
    N_actions = env.action_space.n
    agent = SARSAAgent(init_state, N_states, N_actions)

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

            # Adding synthetic if agent did not died since 
            # rewards in FrozenLake-v0 are very sparse
            if not ended:
                reward += 1e-3

            agent.update(state, reward)
            
            # Metrics
            metrics.record(reward)

        # Logging
        episode_reward = metrics.log(episode)
        progress_bar.set_description('Episode reward: %.3f' % episode_reward)
    
    # Saving weights after training
    weights = format_weights_name(episode, episode_reward, 'frozenlakev0')
    weights_path = save_weights(agent, weights, path=log_dir)
    log.info('Saved weights: %s', weights_path)
    

if __name__== '__main__':
    # Parse args
    description = 'FrozenLake-v0 agent trainer'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        '--episodes', 
        default=20000,
        type=int,
        help='Number of episodes to train the agent')
    parser.add_argument(
        '--log_freq',
        default=100,
        type=int,
        help='Tensorboard logging every log_freq episodes')
    parser.add_argument(
        '--log_dir',
        default='taxi-v3',
        help='Default directory under ./logs/ to store logs')
    args = parser.parse_args()

    # Do the training
    train(episodes=args.episodes, 
          log_every=args.log_freq, 
          log_dir=args.log_dir)
