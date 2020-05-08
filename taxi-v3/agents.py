import torch as T


class QLearningAgent(object):
    """Time Difference learning algorithm: Q learning
    """
    def __init__(self, init_state, N_states, N_actions, 
                 leraning_rate=0.5, epsilon=0.01, 
                 discount_rate=0.99):
        # Saving parameters
        self.leraning_rate = leraning_rate
        self.epsilon = epsilon
        self.discount_rate = discount_rate

        # Building Q table
        self.N_states = N_states
        self.N_actions = N_actions
        self.Q_table = T.randn((N_states, N_actions))

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