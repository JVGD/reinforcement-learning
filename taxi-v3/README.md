# The Taxi Problem

From "Hierarchical Reinforcement Learning with the MAXQ Value FunctionDecomposition" by Tom Dietterich

## Description

There are four designated locations in the grid world indicated by R(ed), G(reen), Y(ellow), and B(lue). When the episode starts, the taxi starts off at a random square and the passenger is at a random location. The taxi drives to the passenger's location, picks up the passenger, drives to the passenger's destination (another one of the four specified locations), and then drops off the passenger. Once the passenger is dropped off, the episode ends.

### Observations

There are 500 discrete states since there are 25 taxi positions, 5 possible locations of the passenger (including the case when the passenger is in the taxi), and 4 destination locations.  

Passenger locations:

* 0: R(ed)
* 1: G(reen)
* 2: Y(ellow)
* 3: B(lue)
* 4: in taxi

Destinations:

* 0: R(ed)
* 1: G(reen)
* 2: Y(ellow)
* 3: B(lue)

Actions: There are 6 discrete deterministic actions:

* 0: move south
* 1: move north
* 2: move east 
* 3: move west 
* 4: pickup passenger
* 5: dropoff passenger

Rewards: There is a reward of -1 for each action and an additional reward of +20 for delivering the passenger. There is a reward of -10 for executing actions "pickup" and "dropoff" illegally.

Rendering:

* blue: passenger
* magenta: destination
* yellow: empty taxi
* green: full taxi
* other letters (R, G, Y and B): locations for passengers and destinations

State space is represented by:

```
(state, reward, end, info) = env.step(action)
```
