import sys
import torch as T
from torch import nn


class QLearningAgent(nn.Module):
    """Time Difference learning algorithm: Q learning
    """
    def __init__(self, init_state, N_states, N_actions, 
                 leraning_rate=0.5, epsilon=0.01, 
                 discount_rate=0.99):
        # Init torch module
        super(QLearningAgent, self).__init__()
        
        # Saving parameters
        self.leraning_rate = leraning_rate
        self.epsilon = epsilon
        self.discount_rate = discount_rate

        # Building Q table as torch parameters
        # so we can save it and load with state_dict
        self.N_states = N_states
        self.N_actions = N_actions
        Q_table = T.randn((N_states, N_actions))
        self.Q_table = nn.Parameter(data=Q_table, requires_grad=False)

        # Params for storing agent-env evolution
        self.action = None
        self.state = init_state
        
    def policy_greedy(self, state):
        # Random distrib in [0,1)
        if T.rand(1) > self.epsilon:
            # Take action with max value for state
            action = T.argmax(self.Q_table[state, :])
        else:
            # Take random action
            action = T.randint(low=0, high=self.N_actions, size=(1,))
        return action

    def __call__(self, current_state):
        # Saving runtime info
        self.state = current_state

        # Getting action to perform
        next_action = self.policy_greedy(current_state)
        
        # Saving action to take and returning it
        self.action = next_action
        return next_action.item()

    def update(self, state_next, reward):        
        # For redability
        r = reward
        s_next = state_next
        Q = self.Q_table
        s = self.state
        a = self.action
        α = self.leraning_rate
        γ = self.discount_rate
        
        # Update rule
        Q[s, a] = Q[s, a] + α * (r + γ * T.max(Q[s_next,:]) - Q[s, a])
        self.Q_table = Q

        # Updating current state
        self.state = state_next


class SARSAAgent(nn.Module):
    """State-Action-Reward-State-Action (SARSA)
    """
    def __init__(self, init_state, N_states, N_actions, 
                 leraning_rate=0.5, epsilon=0.01, 
                 discount_rate=0.99):
        # Init torch module
        super(SARSAAgent, self).__init__()
        
        # Saving parameters
        self.leraning_rate = leraning_rate
        self.epsilon = epsilon
        self.discount_rate = discount_rate

        # Building Q table as torch parameters
        # so we can save it and load with state_dict
        self.N_states = N_states
        self.N_actions = N_actions
        Q_table = T.randn((N_states, N_actions))
        self.Q_table = nn.Parameter(data=Q_table, requires_grad=False)

        # Params for storing agent-env evolution
        self.action = None
        self.state = init_state
        
    def policy_greedy(self, state):
        # Random distrib in [0,1)
        if T.rand(1) > self.epsilon:
            # Take action with max value for state
            action = T.argmax(self.Q_table[state, :])
        else:
            # Take random action
            action = T.randint(low=0, high=self.N_actions, size=(1,))
        return action

    def __call__(self, current_state):
        # Saving runtime info
        self.state = current_state

        # Getting action to perform
        next_action = self.policy_greedy(current_state)
        
        # Saving action to take and returning it
        self.action = next_action
        return next_action.item()

    def update(self, state_next, reward):        
        # For redability
        r = reward
        s_next = state_next
        Q = self.Q_table
        s = self.state
        a = self.action
        α = self.leraning_rate
        γ = self.discount_rate
        
        # Choosing next action with policy epsilon greedy
        a_next = self.policy_greedy(s_next)

        # SARSA update rule
        Q[s, a] = Q[s, a] + α * (r + γ * Q[s_next, a_next] - Q[s, a])
        self.Q_table = Q

        # Updating current state
        self.state = state_next