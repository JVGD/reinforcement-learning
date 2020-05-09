## Overview

### Details
* Name: FrozenLake-v0  
* Category: Classic Control

### Description
The goal of this game is to go from the starting state (S) to the goal state (G) by walking only on frozen tiles (F) and avoid holes (H). However, the ice is slippery, so you won't always move in the direction you intend (stochastic environment)

### Source
Came from this Colab and blog
[Blog](https://colab.research.google.com/drive/1oqon14Iq8jzx6PhMJvja-mktFTru5GPl#scrollTo=5aQKQMJTJBPH)


## Environment

### Observation
Type: Discrete (16)

Num | Observation (State)
---|---
0 - 15 | For 4x4 square, counting each position from left to right, top to bottom

### Actions
Type: Discrete(4)

Num | Action
--- | ---
0 | Move Left
1 | Move Down
2 | Move Right
3 | Move Up

### Reward
Reward is 0 for every step taken, 0 for falling in the hole, 1 for reaching the final goal

### Starting State
Starting state is at the top left corner

### Episode Termination
1. Reaching the goal or fall into one of the holes

### Solved Requirements
Reaching the goal without falling into hole over 100 consecutive trials.