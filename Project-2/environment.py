import numpy as np
import random

class Environment:
     def __init__(self, grid_size, render_mode = False):
         self.grid_size = grid_size
         self.grid = []
         self.render_mode = render_mode
         self.agent_location = ()
         self.goal_location = ()

     def reset(self):
         self.grid = np.zeros((self.grid_size, self.grid_size))
         self.agent_location = self.add_agent()
         self.goal_location = self.add_goal()

         if self.render_mode == True :
             self.render()

         return self.get_state()


     def get_state(self):
         state = self.grid.flatten()
         return state



     def add_agent(self):
         location = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
         self.grid[location[0], location[1]] = 1

         return location

     def add_goal(self):
         location = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))

         while self.grid[location[0], location[1]] == 1:
             location = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))

         self.grid[location[0], location[1]] = -1

         return location


     def render(self):
         grid = self.grid.astype(int).tolist()

         for row in grid:
             print(row)

         print("")




     def is_valid_location(self, location):
         if (0 <= location[0] < self.grid_size and 0 <= location[1] < self.grid_size):
             return True
         else:
             return False

     def move_agent(self, action): #return reward and done(boolean)
         move = {
             0:(-1, 0),
             1:(1, 0),
             2:(0, -1),
             3:(0, 1)
         }

         new_move = move[action]
         previous_location = self.agent_location
         new_location = (previous_location[0] + new_move[0], previous_location[1] + new_move[1])

         done = False
         reward = 0

         if self.is_valid_location(new_location):
             self.grid[previous_location[0]][previous_location[1]] = 0
             self.grid[new_location[0]][new_location[1]] = 1
             self.agent_location = new_location

             if self.agent_location == self.goal_location:
                reward = 100
                done = True
             else:
                 previous_distance = np.abs(self.goal_location[0] - previous_location[0]) + np.abs(self.goal_location[1] - previous_location[1])
                 new_distance = np.abs(self.goal_location[0] - new_location[0]) + np.abs(self.goal_location[1] - new_location[1])

                 reward = (previous_distance - new_distance) - 0.1
         else:
             reward = -3
         return reward, done


     def step(self, action):
         reward, done = self.move_agent(action)
         next_state = self.get_state()

         if self.render_mode:
             self.render()

         return reward, next_state, done





















########
# test

# env = Environment(5, render_mode=True)
# env.reset()
# print("agent:{}\n goal:{}".format(env.agent_location, env.goal_location))
#
#
# env.reset()
# print("agent:{}\n goal:{}".format(env.agent_location, env.goal_location))
#
#
# env.reset()
# print("agent:{}\n goal:{}".format(env.agent_location, env.goal_location))







