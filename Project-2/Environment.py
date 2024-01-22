import numpy as np
import random

class Envvironment:
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



########
# test

env = Envvironment(5, render_mode=True)
env.reset()
print("agent:{}\n goal:{}".format(env.agent_location, env.goal_location))


env.reset()
print("agent:{}\n goal:{}".format(env.agent_location, env.goal_location))


env.reset()
print("agent:{}\n goal:{}".format(env.agent_location, env.goal_location))







