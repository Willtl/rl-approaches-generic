# Reinforcement Learning - Grid Game

*This project was developed in order to improve my skills with reinforcement learning, as well as to demonstrate a situation where deep reinforcement learning is more viable than Q-learning.*

The game was developed to easily (with the increment of rows and columns) increase the number of states. Since the seed spawns randomly, if we consider *x* as the number of rows and columns, the number of states is equal to *x<sup>2</sup>*, and four being the number of actions (i.e., up, right, down, left). 

The project's idea relies on the fact that the number of states increases exponentially with the number of rows and columns. In this sense, Q-learning falters with increasing numbers of states/actions since the likelihood of the agent visiting a particular state and performing a particular action is increasingly small. One solution is to use an artificial neural network as a function approximator.

## Q-learning
The agent's performance given different number of iterations (each action is one iteration)
| 100000 | 250000 | 500000 | 1000000 |
| --- | --- | --- | --- |
| ![](images/100000.gif) | ![](images/250000.gif) | ![](images/500000.gif) | ![](images/1000000.gif) | 
 
