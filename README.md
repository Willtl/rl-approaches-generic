# Reinforcement Learning - Grid Game

*This project was developed in order to improve my skills with reinforcement learning, as well as to demonstrate a situation where deep reinforcement learning is more viable than Q-learning.*

The game simply consists of a grid of cells with one seed and one player (the game accepts multiple players/seeds but this is out of the scope of the project). The game starts with the *player* (blue) and the *seed* (green) at a random cell such that the row and columns of both are not the same. The player can move up, right, down, and left. Every time it overlaps with the seed it scores one point and a new seed is spawned at a random cell.

The mechanics of the game were developed to easily (with the increment of rows and columns) increase the number of states. Since the seed spawns randomly, if we consider *x* as the number of rows and columns, the number of states is equal to *x<sup>2</sup>*, and four being the number of actions (i.e., up, right, down, left). Therefore, the Q-table's size is equal to *x<sup>2</sup> * 4*.

The project's idea relies on the fact that the number of states increases exponentially with the number of rows and columns. In this sense, Q-learning falters with increasing numbers of states/actions since the likelihood of the agent visiting a particular state and performing a particular action is increasingly small. It is pretty clear that we can’t infer the Q-value of new states from already explored states. 

One solution is to approximate these Q-values with machine learning models such as an artificial neural network. Function approximation may speed up learning in finite problems, due to the fact that the model can generalize earlier experiences to previously unseen states.

Below, the performance of each strategy is demonstrated and described to different game' instances sizes.

## Q-learning
The agent's performance with Q-learning on a *5x5* instance of the game given different number of iterations (each action is one iteration)
| 100000 | 250000 | 500000 | 1000000 |
| --- | --- | --- | --- |
| ![](images/100000.gif) | ![](images/250000.gif) | ![](images/500000.gif) | ![](images/1000000.gif) | 
 
