import argparse
import logging
from os import name, system
from time import sleep

import gym
import numpy as np
from numpy import random
from tqdm import tqdm


def parse_args():
    description = 'Frozen Lake Agent Trainer and Player'

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--train", action='store_true',
                        help="Train the agent")
    parser.add_argument("--total_episodes",
                        help="Number of episodes for the agent to learn")
    parser.add_argument("--learning_rate",
                        help="Learning rate for the training")
    parser.add_argument("--max_steps",
                        help="Max steps to take for the agent in 1 episode")
    parser.add_argument("--play", action='store_true',
                        help="Play the frozen lake game")
    parser.add_argument("--policy",
                        help="Policy to load to play with")
    parser.add_argument("--debug", action='store_true',
                        help="Wheter to print debug messages")
    args = parser.parse_args()

    # Defatult configuration
    conf = {
        'train': True,
        'play': True,
        'total_episodes': 10000,
        'learning_rate': 0.8,
        'max_steps': 100,
        'policy': None,
        'debug': False
    }
    
    # Overriding conf with args
    if args.train:
        conf['play'] = False
    if args.total_episodes:
        conf['total_episodes'] = int(args.total_episodes)
    if args.learning_rate:
        conf['learning_rate'] = int(args.learning_rate)
    if args.max_steps:
        conf['max_steps'] = int(args.max_steps)

    if args.play:
        conf['train'] = False    
        if args.policy:
            conf['policy'] = args.policy
        else:
            raise ValueError('For --play you must provide a '
                             'valid policy with --policy')
    if args.debug:
        conf['debug'] = args.debug
    
    return conf


def clear():
    '''Clear terminal, useful for rendering'''
    # for windows
    if name == 'nt':
        _ = system('cls')

    # for mac and linux
    else:
        _ = system('clear')


def get_nice_logger(debug=False, name=__file__):
    '''Get nice logger :)'''
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


def train(total_episodes=10000, learning_rate=0.8, max_steps=100):

    # Loading environment
    log.info('Loading environment')
    env = gym.make('FrozenLake-v0')

    # 0 left, 1 down, 2 right, 3 up
    actions = ['LEFT', 'DOWN', 'RIGHT', 'UP']

    # Getting environment data for Q-table
    # will have Q dimensions(N x M)
    #   M : number of possible actions
    #   N : number of possible states
    M = env.action_space.n
    N = env.observation_space.n

    # Defining our Q-table
    log.info('Initializing Q-Table with random values')
    Q = np.random.rand(N, M)

    # Exploration parameters
    gamma = 0.95                  # Discounting rate
    epsilon = 1.0                 # Exploration rate
    max_epsilon = 1.0             # Exploration probability at start
    min_epsilon = 0.01            # Minimum exploration probability 
    decay_rate = 0.01             # Exponential decay rate for exploration

    # List of rewards
    rewards = []

    # Fancy progress bar
    pbar = tqdm(range(total_episodes))

    # Looping through different episodes
    for episode in pbar:
        
        # Reseting environment for new episode
        state = env.reset()
        step = 0
        done = False
        total_rewards = 0

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
            log.debug('Q[%r,%r] = ' % (str(state), actions[action]) +
                      'Q[state, action] + learning_rate * (reward '
                      '+ gamma * max(Q[new_state, :]) - '
                      'Q[state, action])')

            # Valoration
            # Updating Q-table
            Q[state, action] = (
                Q[state, action] + learning_rate * 
                (reward + gamma * max(Q[new_state, :]) - Q[state, action])
            )
            
            # Updating the state
            state = new_state

            # Logging info
            log.debug('Action: %r', actions[action])
            log.debug('Next State: %r', new_state)
            log.debug('Reward: %r', reward)
            log.debug('max(Q[new_state, :]) = %r', np.max(Q[new_state, :]))
            log.debug('New_Q[%r,%r] = %r', state, 
                      actions[action], Q[state, action])

            # Since rewards are very sparse and only happens when the agent
            # reaches its final state, we help the agent a bit by giving him
            # a positive intermediate reward if didnt die in the step taken
            # If not dead, update reward
            if not dead:
                total_rewards += reward + 1
                if reward > 0:
                    print(reward)

            # If dead we could update with negative reward
            if dead: 
                total_rewards += 0
                log.debug('Agent died... :(')
                break

        # Reduce epsilon (because we need less and less exploration)
        epsilon = (
            min_epsilon + 
            (max_epsilon - min_epsilon) * np.exp(- decay_rate * episode)
        )
        rewards.append(total_rewards)

        # Updating progress bar with expected cumulative rewards
        pbar.set_description('Score: %.2f' % np.mean(rewards))


    # Clearing terminal and showing reports
    log.info('Reports: ')
    log.info('Score over time:   ' + str(sum(rewards) / total_episodes))
    log.info('Total episodes:    ' + str(total_episodes))
    log.info('Steps per episode: ' + str(max_steps))
    log.info('Q-table (numbers might be small)')
    log.info(str(Q))

    log.info('Saving Q policy')
    q_policy_name = 'Q_policy' + str(np.mean(rewards)) + '.npy'
    np.save(q_policy_name, Q)

    return q_policy_name


def play(Q, episodes=5, max_steps=100):
    '''Playing the game'''

    # Loading policy
    Q = np.load(Q)

    # Reseting environment
    log.info('Playing the game')

    # Loading environment
    log.info('Loading environment')
    env = gym.make('FrozenLake-v0')

    for episode in range(episodes):
        state = env.reset()
        step = 0
        done = False

        for step in range(max_steps):
            
            # Waiting after simulation and clearing terminal
            sleep(0.5)
            clear()

            # Print steps
            log.info('Episode: %d', episode)
            env.render()
            
            # Take the action (index) that have the maximum 
            # expected future reward given that state
            action = np.argmax(Q[state,:])
            
            new_state, reward, done, info = env.step(action)
            
            if done:
                # Print terminal state
                sleep(0.5)
                clear()
                log.info('Episode: %d', episode)
                env.render()

                # Printing reports
                log.info('Finish Report:')
                log.info('Steps:    %s', step)
                log.info('Position: %s', new_state)
                if new_state == 15:
                    log.info('Success: GOAL REACHED! :)')
                else:
                    log.info('Success: Game Over :(')

                input('Press enter...')
                break

            state = new_state
            
    env.close()


def main(conf):

    # Default behaviour: train and play with results from training
    if conf['play'] and conf['train']:
        Q = train(total_episodes=conf['total_episodes'], 
                  learning_rate=conf['learning_rate'], 
                  max_steps=conf['max_steps'])
        play(Q)

    # Just train and save result policy
    elif conf['train']:
        Q = train(total_episodes=conf['total_episodes'], 
                  learning_rate=conf['learning_rate'], 
                  max_steps=conf['max_steps'])
    
    # Just play with given policy
    elif conf['play']:
        play(conf['policy'])


# Getting a nice logger to print stuff
global log

# Entry point
if __name__ == '__main__':
    # Getting conf
    conf = parse_args()

    # Configuring logger
    log = get_nice_logger(debug=conf['debug'], name='rocker-logger')
    
    # Running program
    main(conf)
