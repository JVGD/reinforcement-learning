from numpy import *
import gym
import random
import pdb
import time

# For printing in terminal
from os import system, name
from time import sleep
 
def clear():
	"""Clear terminal"""
	# for windows
	if name == 'nt':
		_ = system('cls')

	# for mac and linux(here, os.name is 'posix')
	else:
		_ = system('clear')


# Adding some variables to measure the
# learning, this will only cause effect
# if debug = True
debug = False

# 0 left, 1 down, 2 right, 3 up
actions = ["LEFT", "DOWN", "RIGHT", "UP"]

# Loading environment
env = gym.make("FrozenLake-v0")

# Getting environment data for Q-table
# will have Q dimensions(N x M)
# - M : number of possible actions
# - N : number of possible states
M = env.action_space.n
N = env.observation_space.n

qtable = zeros([N,M])

total_episodes = 10000        # Total episodes
learning_rate = 0.8           # Learning rate
max_steps = 100               # Max steps per episode
gamma = 0.95                  # Discounting rate

# Exploration parameters
epsilon = 1.0                 # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability 
decay_rate = 0.01             # Exponential decay rate for exploration prob

# Learning
# List of rewards
rewards = []



# Time of training
t_start = time.time()

# Looping through different episodes
for episode in range(total_episodes):
    
    # Reseting environment for new episode
    state = env.reset()
    step = 0
    done = False
    total_rewards = 0

    # Printing train status
    if episode > 0:
        if episode % 100 == 0:
            clear()
            print("Training Status: " +str(episode/total_episodes * 100)+"%")
            print("Score over time: " +  str(sum(rewards)/episode))

    
    # Looping for actions in episode
    for step in range(max_steps):
        exploration_explotation_tradeoff = random.uniform(0,1)
        
        # Decision
        # Determining next action to take
        if exploration_explotation_tradeoff > epsilon:
            # Exploitation
            # Returns the index of the best action for state_i
            action = argmax(qtable[state,:])
        else:
            # Exploration
            # Takes a random action from qtable
            # TODO: NOT SAMPLING PROPERLY
            # Forcing to sampling right
            action = env.action_space.sample()

        # Observation
        # Getting the new state or stepping into that state
        new_state, reward, dead, info = env.step(action)

        # Debug info
        if debug:
            print("qtable["+str(state)+","+actions[action]+"] = "
                "qtable[state, action] + learning_rate * (reward "
                "+ gamma * max(qtable[new_state, :]) - qtable[state, action])")

        # Valoration
        # Updating Q-table
        qtable[state, action] = qtable[state, action] + learning_rate * \
                (reward + gamma * max(qtable[new_state, :]) - qtable[state, action])
        
        # Cumulative reward for this episode
        total_rewards += reward
        
        # Updating the state
        state = new_state
        
        if debug:
            print("\tAction: "+actions[action])
            print("\tNext State: " + str(new_state))
            print("\tReward:", str(reward))
            print("\tmax(qtable[new_state, :]) = " + str(max(qtable[new_state, :])) ) 
            print("New_qtable["+str(state)+","+actions[action]+"] = " + str(qtable[state, action]))
            input("Press a key to continue...")
            print("")

        # If dead, finish episode
        if dead == True: 
            break
    
    # Updating number of episodes
    episode += 1
    
    # Reduce epsilon (because we need less and less exploration)
    epsilon = min_epsilon + (max_epsilon - min_epsilon)* exp(-decay_rate*episode) 
    rewards.append(total_rewards)

# Train end timming
t_train = time.time() - t_start

# Clearing terminal and showing reports
clear()
print ("Reports: ")
print ("Time to train:     " + str(t_train) + " s")
print ("Score over time:   " +  str(sum(rewards)/total_episodes))
print ("Total episodes:    " +  str(total_episodes))
print ("Steps per episode: " +  str(max_steps))

print (" ")
print ("Q-table (numbers might be small)")
print(str(qtable))

input("Press enter....")

def play():

    # Playing
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
            
            # Take the action (index) that have the maximum expected future reward given that state
            action = argmax(qtable[state,:])
            
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
    play()