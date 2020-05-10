import sys
import torch as T
from torch import nn
from torch import optim

from rl.agents.networks import QNetwork
from rl.agents.memory import ExperienceReplayBuffer


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


class QNetworkAgent(nn.Module):
    def __init__(self, init_state, state_size, N_actions,
                 memory_capacity=20000, batch_size=128,
                 update_steps=100, learning_rate=1e-2, 
                 epsilon=0.01, discount_rate=0.99):
        # Init torch module
        super(QNetworkAgent, self).__init__()

        # Saving parameters
        self.batch_size = batch_size
        self.update_steps = update_steps
        self.epsilon = epsilon
        self.discount_rate = discount_rate

        # Params for storing agent-env evolution
        self.iterations = 0
        self.action = None
        self.state = init_state

        # Getting Q networks as state-action value approximator
        # one Q net for sampling environment and the other as the
        # target to approximate
        self.N_actions = N_actions
        self.Q_sample = QNetwork(state_size, N_actions)
        self.Q_target = QNetwork(state_size, N_actions)
        self.optimizer = optim.RMSprop(self.Q_sample.parameters(), 
                                       lr=learning_rate)

        # Experience replay memory buffer
        self.memory = ExperienceReplayBuffer(memory_capacity)

    def policy_greedy(self, actions_value):
        if T.rand(1) > self.epsilon:
            # Take action with max value for state
            action = T.argmax(actions_value)
        else:
            # Take random action
            action = T.randint(low=0, high=self.N_actions, size=(1,))
        return action

    def __call__(self, current_state):

        # To torch tensor
        current_state = T.tensor(current_state, dtype=T.float).detach()
        
        # Storing current state
        self.state = current_state

        # Getting value for each action in current_state
        # We set not to records gradient since for this operation 
        # since gradients for loss are not cumputed with respect
        # to this transition but for the sampled exp replay buffer
        with T.no_grad():
            actions_value = self.Q_sample(current_state)

        # Getting action with greedy policy
        action = self.policy_greedy(actions_value)

        # Storing selected action
        self.action = action

        # Returning action as int instead of torch tensor / scalar
        return action.item()

    def update(self, state_next, reward):
        # Storing transition in memory for experience replay
        self.memory.store(state=self.state, action=self.action, 
                          state_next=state_next, reward=reward)

        # Sampling the memory buffer for transitions to compute loss
        # transitions : [B, (s, a, s', r)]
        transitions = self.memory.sample(self.batch_size)

        # For redability
        s = transitions[:, 0]
        a = transitions[:, 1]
        s_next = transitions[:, 2]
        r = transitions[:, 3]
        γ = self.discount_rate
        Qs = self.Q_sample
        Qt = self.Q_target

        # Computing loss (forward pass in DL)
        loss = (r + γ * T.max(Qt(s_next), dim=-1) - Qs(s)[a]) ** 2

        # Optimization (backward pass in DL)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # After update_steps iterations we update Q_target  
        # weights with weights from Q_sample
        if (self.iterations % self.update_steps) == 0:
            weights_Q_sample = self.Q_sample.state_dict()
            self.Q_target.load_state_dict(weights_Q_sample)

        # Updating iterations
        self.iterations =+ 1