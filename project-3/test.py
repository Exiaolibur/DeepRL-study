from train import model
import gymnasium as gym

env_name = "LunarLander-v2"
env = gym.make(env_name, render_mode = "human")
episodes = 10

for episode in range(episodes + 1):

   obs, info = env.reset()
   terminated = False
   score = 0

   while not terminated:

      action, next_state = model.predict(obs)
      obs, reward, terminated, truncated, info = env.step(action)
      score += reward

   print("Episode:{}, Score:{}".format(episode, score))