#%% [markdown]
# # Simple 4x4 Grid Example
# 
# This example is a simple 4x4 grid to get the basic understanding of the environment and the variables in the Markov Decision Processes (MDP)
# 

#%%
import numpy as np

#%% [markdown]
# The environment has 4 states:
# 
# $S = \left\{ s_0, s_1, s_2, s_3 \right\} $
# 
# The agent only knows how to take this for actions:
# 
# $A = \left\{ a_0(\uparrow), a_1(\downarrow), a_2(\leftarrow), a_3(\rightarrow) \right\} $
# 
# The transition function is deterministic, and is all zeros except for the allowed transitions, for example, being in $s_0$ and taking action $a_1(\downarrow)$ will get you to the state $s_2$

#%%
S = [0, 1, 2, 3]
S_N = len(S)

A = [0, 1, 2, 3]
A_N = len(A)

T = np.zeros((S_N, A_N, S_N))
T[ S[0], A[1], S[2] ] = 1
T[ S[0], A[3], S[1] ] = 1

T[ S[1], A[2], S[0] ] = 1
T[ S[1], A[1], S[3] ] = 1

T[ S[2], A[0], S[0] ] = 1
T[ S[2], A[3], S[3] ] = 1

T[ S[3], A[0], S[1] ] = 1
T[ S[3], A[2], S[0] ] = 1


#%%
# Initial random policy pi:
# policy internal storage
# function pi for selecting optimal action
# pi: SxA -> A
# policy = SxA
Q = np.array([[0, 0, 0, 1],
              [0, 0, 1, 0],
              [1, 0, 0, 0],
              [1, 0, 0, 0]], dtype=np.float)

pi = np.argmax(Q, axis=1)

#%%
# Reward function: optimal path will be s0, s2, s3
# R:S -> Real number
R = np.array([0, 2, 4, 6])
# R = np.array([-6, -4, -2, -0])

# Value function
# V:S -> Real number
V = np.array([0, 0, 0, 0])

# Initial sigma and delta value for convergence or stop condition
sigma = 100
delta = 0.2

# Learning rate
gamma = 0.5

# Each iteration step evaluates how good is to be in
# state taking into account N iterations in the future,
# so we evaluate the value function for 20 iterations, or
# for a future prevision (horizon) of 20 steps
iteration = 0
future_horizon = 1

# While V_new very different from V continue evaluation till
# the difference is lower than delta
while delta < sigma and iteration < future_horizon:

    # Calculation of first iteration of V(s)
    # Copy of value function array with same elements
    V_new = np.array(V)
    
    for s in S:
        for s_next in S:
            # Implementing the summatory
            V_new[s] += T[s, pi[s], s_next] * ( R[s] + gamma * V[s_next] )
            
        #     # Debug
        #     print("T[s%d, pi(s%d)=a%d, s%d_k+1]=%.2f * ( R[s]=%d + 0.1 * V[s_k+1]=%.2f )" 
        #             %(s,s,pi(s),s_next, T[s, pi(s), s_next], R[s], V[s_next]) )

        # # Debug
        # print("")

    # Difference calculation for convergence
    delta = np.mean(np.abs(V - V_new))

    # Updating
    V = np.array(V_new)

    # Debug
    iteration += 1
    print("%d: V(s) = %s, delta = %s" %(iteration, V_new, delta))
    

# Policy Improvement
print("pi: %s" % pi)
print("Q:\n %s" %Q)
Q_new = np.array(Q)

for s in S:
    for a in A:
        for s_next in S:
            Q_new[s,a] = T[s, a, s_next] * ( R[s] + gamma * V[s_next] )

# Updating Q function
Q = np.array(Q_new)

# Updating policy
pi = np.argmax(Q, axis=1)
print("pi: %s" % pi)
print("Q:\n %s" %Q)

