***This peoject is for learning Deep Q-learning***

referring to the project guide : https://towardsdatascience.com/develop-your-first-ai-agent-deep-q-learning-375876ee2472#c87e

**steps:**
DQL:

A step refers to a cycle. An episode refers to multiple steps that form a complete training cycle (stop when reaching the target, or force stop after reaching max_step)

Environment setup:

Construct environment class and reset() method
environment class:
The initialization attribute is grid size, and the method is reset: use numpy to give a 0 matrix. And (if 3x3) np.zero((3,3)) uses tuple (3,3) to represent the shape.


Construct methods for adding agent and setting goal:
Use random to randomly select a point on the 0 matrix as 1 (i.e. agent)
Use random, set to -1 at a point that is not 1

(random.randint(x,y) is used to generate a random integer from x to y)

Both methods finally return the location of the processing point.

"Render" visual interface:
This is not a real rendering, it is simulated rendering by printing. Print the results on the console to visualize the process:
Convert the grid into a list of integers: print out each row, and these rows are finally used as the grid. This is a render. To distinguish, each render is separated by a blank line.


Use astype and tolist to convert here:





Integrate and execute:
After completing the environment setup, perform a test:

Initialize the environment——》Set agent and goal——》Render display


Enrich the implementation functions of the reset method:
Use the test process just used as the reset method process:
Let it have the function of rendering decision and position reset: detect whether render is needed when reset is triggered. And the positions of the agent and goal are assigned by add agent and add goal to complete the reset and prepare for a new round of episode. The reset here refers to completing the random selection again.

Change the state of the grid from two dimensions to one dimension as the state after the environment changes (here after reset), so that the neural network can be used as input:
Add a get_state method as the output of reset.


Agent (neural network):

Define agent class and network
The agent class has a model attribute. Call this attribute to create a model directly.
When building the model, you need to know grid_size, which is the input size.
Call tensorflow to create several layers of neural networks. (To flatten (elongate) two-dimensional input into one-dimensional input)
Dense of keras is used to build the neural node layer, and sequential is used to build the network structure of the layer (because tensorflow is too lazy to use, I switched to pytorch to continue)

Define a method for policy decisions:

A typical policy function in reinforcement learning selects an action based on the state and an estimated Q-value. It first unfolds the state into a single instance in a batch, then uses a neural network model to estimate the Q-value, and finally selects the action with the highest Q-value as the output. This process is part of Q-learning and is used to determine the action an agent should take in a given state.
In addition to the greedy strategy, there is also the Epsilon-Greedy strategy, where epsilon is the probability of choosing a random operation. Sets the decay value for epsilon decay after each agent execution. For example: Step 1000: epsilon = 1 * (0.998)^1000 = 0.135, but you also need to define a stopping point to prevent it from lowering all the time.
Now, update the agent to introduce epsilon:

In the action decision-making area, first introduce a random number of 0-1 and compare it with epsilon, and perform two actions based on the results: random action or the action with the highest Q value.


Improve environment settings:

Define rewards:
  There are two types of rewards:
1. Sparse reward: Generally used in complex environments (such as autonomous driving), there are few opportunities to obtain positive rewards.
2. Dense reward: In a simple environment, every step can be rewarded.
A dense reward structure allows agents to train faster and behave more predictably.


Agents interact with the environment:
Allows the agent to interact with the environment and ensures that the agent is active within a valid scope. The environment takes the agent's action as input, judges the action (one of the four states of 0123), and uses a small change in move to update the old position.

But a method must be used to determine whether the movement is within the defined range. If only.

On the premise that feedback is required and it is judged whether the new position is a reward position, True or False is returned as feedback.

Implement sparse rewards: (reward rationing only for goal achievement of results)
100 points will be added for reaching the reward position, 1 point will be deducted for not reaching the reward position, and 3 points will be deducted for not being within the valid range.
and! Redefine the return value. The move_agent method has two return values, one is reward and the other is the Boolean value done.

Implement intensive rewards: (give feedback more frequently, achieve goals in a progressive manner, pay attention to more dynamic details)
Calculate whether the Manhattan distance is increasing or decreasing at each step to give feedback.
For convenience, we only need to make changes to places that are not in goal but are effective. First determine whether it is at the goal position. If not, calculate the difference between the Manhattan distance before and after and use it as reward (in this case it will only be 1 or -1, which can be appropriately adjusted to a value suitable for the task).

Step penalty:
The implemented reward structure has many network neutral loops. It can move back and forth between the two locations forever without accruing any penalties. We can solve this problem by subtracting a small value at each step. (There is some confusion about the definition of this part)





Now connect the things in the environment and apply actions to the environment:
Enter action, return reward and state, and done (a boolean value)


Add memory function:

Define class: ExperienceReplay

Need to define memory(deque), batch_size, and Experience(namedtuple)



Defining the agent’s learning process: Fitting a neural network

We hope to add subsequent rewards to the training, so the subsequent Q values are also put into the network for training.


Discounting future rewards – the role of gamma
When thinking about the future, we don't want it to be equally important (weighted) as the present. The degree to which we discount the future, or reduce its impact on each decision, is defined by gamma (often represented by the Greek letter γ).
Gamma can be adjusted, with higher values encouraging planning and lower values encouraging more short-sighted behavior. We will use the default value of 0.99. The discount factor is almost always between 0 and 1.

Implement gamma and define target Q-value
In the context of training a neural network, the process depends on two key elements: the input data we provide and the corresponding output we want the network to learn to predict.
We need to provide the network with some target Q-values that are updated based on the reward given by the environment at that particular state and action, and the discounted (in terms of gamma) predicted reward for the best action in the next state.
Define target_q_values, find the q value corresponding to the action of the i-th experience through the [i, action [i]] index, and set the target Q value at this time to the Q value of the next state plus reward (because the Q value of the next state The value depends on the action made based on the Q value at this time. Since we need to take a long-term view, we replace it with the Q of the next state plus the reward of the action at this time), and take gamma into account. If it ends (the time is up or the task is completed), there is no Q value for the next state, and the reward at this time is directly used as the target.



Define a learn method for the agent:
There are i experiences among the experiences in a batch, which correspond to the i experiences in the batch in turn. Finally, it is saved as target_q_values with i q-value tables.




Set the main function:

To define an instance, first set episodes and steps. Then each step performs the following loop:

Get the action, input the action into env, get the output of env, and then put the output into exp,
Then keep adding to exp. When the amount is enough, start learning from exp.
Update state. Save the weight file after each episode
