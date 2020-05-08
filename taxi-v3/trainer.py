import gym
import torch as T
from tqdm import tqdm
from agents import QLearningAgent


def train(episodes=1000):
    """Trains an agent to play Taxi-v3
    """
    # Building environment & reseting to init state
    env = gym.make('Taxi-v3')
    init_state = env.reset()
    state = init_state

    # Creating the agent
    N_states = env.observation_space.n
    N_actions = env.action_space.n
    agent = QLearningAgent(init_state, N_states, N_actions)

    # Training loop
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
        if episode % 10 == 0:
            print('%.3f' % rewards_avg)



if __name__== '__main__':
    train()
