import logging
from os import name, system
from time import sleep

import gym
import numpy as np
from numpy import random
from tqdm import tqdm


def clear():
    """Clear terminal"""
    # for windows
    if name == 'nt':
        _ = system('cls')

    # for mac and linux
    else:
        _ = system('clear')


def get_nice_logger(debug=False, name=__file__):
    """Get nice logger :)
    """
    # Log format
    formating = '%(msecs)d [%(levelname)s] %(message)s'
    formatter = logging.Formatter(formating)

    # Handler for logging to stdout
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    if debug:
        console_handler.setLevel(logging.DEBUG)
    else:
        console_handler.setLevel(logging.INFO)
    
    # Getting the logger that will handle multiple handlers
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)
    return logger


def play():
    """Playing the game"""

    # Reseting environment
    env.reset()

    for episode in range(5):
        state = env.reset()
        step = 0
        done = False

        for step in range(max_steps):
            
            # Clear terminal
            clear()

            # Print steps
            print("EPISODE ", episode)
            
            env.render()
            
            # Take the action (index) that have the maximum expected future   
            # reward given that state
            action = np.argmax(Q[state,:])
            
            new_state, reward, done, info = env.step(action)
            
            if done:
                # Rendering final step to see final state
                clear()
                # Print steps
                print("EPISODE ", episode)
                env.render()

                # Printing reports
                print(" ")
                print("Finish Report:")
                print("Steps:    " + str(step))
                print("Position: " + str(new_state))
                if new_state == 15:
                    print("Success:  " + "GOAL REACHED! :)")
                else:
                    print("Success:  " + "Game Over :(")
                print(" ")
                input("Press enter...")
                break

            state = new_state

            # Time between simulations
            sleep(0.1)
            
    env.close()


if __name__ == "__main__":
        
    # Getting a nice logger to print stuff
    log = get_nice_logger(debug=False, name='rocker-logger')

    # Loading environment
    log.info('Loading environment')
    env = gym.make("FrozenLake-v0")

    # 0 left, 1 down, 2 right, 3 up
    actions = ["LEFT", "DOWN", "RIGHT", "UP"]

    # Getting environment data for Q-table
    # will have Q dimensions(N x M)
    #   M : number of possible actions
    #   N : number of possible states
    M = env.action_space.n
    N = env.observation_space.n

    # Defining our Q-table
    log.info('Initializing Q-Table with random values')
    Q = np.random.rand(N, M)

    total_episodes = int(1e5)     # Total episodes
    learning_rate = 0.8           # Learning rate
    max_steps = 100               # Max steps per episode
    gamma = 0.95                  # Discounting rate

    # Exploration parameters
    epsilon = 1.0                 # Exploration rate
    max_epsilon = 1.0             # Exploration probability at start
    min_epsilon = 0.01            # Minimum exploration probability 
    decay_rate = 0.01             # Exponential decay rate for exploration

    # List of rewards
    rewards = []

    # Looping through different episodes
    for episode in tqdm(range(total_episodes)):
        
        # Reseting environment for new episode
        state = env.reset()
        step = 0
        done = False
        total_rewards = 0

        if episode % 500 == 0:
            log.info("Training: %.2f%%", (episode / total_episodes) * 100)
            log.info("Score   : %.2f", np.mean(rewards))

        # Looping for actions in episode
        for step in range(max_steps):
            exploration_explotation_tradeoff = random.uniform(0,1)
            
            # Decision
            # Determining next action to take
            if exploration_explotation_tradeoff > epsilon:
                # Exploitation, returns the index of the best action for state
                action = np.argmax(Q[state,:])
            else:
                # Exploration Takes a random action from Q
                action = env.action_space.sample()

            # Observation
            # Getting the new state or stepping into that state
            # Rewards will be only given when reached goal, this a situation
            # where only sparse rewards are given
            new_state, reward, dead, info = env.step(action)

            # Debug info
            log.debug("Q[%r,%r] = " % (str(state), actions[action]) +
                      "Q[state, action] + learning_rate * (reward "
                      "+ gamma * max(Q[new_state, :]) - "
                      "Q[state, action])")

            # Valoration
            # Updating Q-table
            Q[state, action] = (
                Q[state, action] + learning_rate * 
                (reward + gamma * max(Q[new_state, :]) - Q[state, action])
            )
            
            # Cumulative reward for this episode
            # total_rewards += reward
            
            # Updating the state
            state = new_state
            
            # Logging info
            log.debug("Action: %r", actions[action])
            log.debug("Next State: %r", new_state)
            log.debug("Reward: %r", reward)
            log.debug("max(Q[new_state, :]) = %r", np.max(Q[new_state, :]))
            log.debug("New_Q[%r,%r] = %r", state, 
                      actions[action], Q[state, action])
            
            # Since rewards are very sparse and only happens when the agent
            # reaches its final state, we help the agent a bit by giving him
            # a positive intermediate reward if didnt die in the step taken
            # If not dead, update reward
            if not dead:
                total_rewards += reward*10 + 1
                if reward > 0:
                    print(reward)

            # If deat update with negative reward
            if dead: 
                total_rewards += -10
                log.debug("Agent died... :(")
                break
        
        # Updating number of episodes
        episode += 1
        
        # Reduce epsilon (because we need less and less exploration)
        epsilon = (
            min_epsilon + 
            (max_epsilon - min_epsilon) * np.exp(- decay_rate * episode)
        )
        rewards.append(total_rewards)

    # Clearing terminal and showing reports
    log.info("Reports: ")
    log.info("Score over time:   " + str(sum(rewards) / total_episodes))
    log.info("Total episodes:    " + str(total_episodes))
    log.info("Steps per episode: " + str(max_steps))
    log.info("Q-table (numbers might be small)")
    log.info(str(Q))
