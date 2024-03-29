import gymnasium as gym
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

env = gym.make('CartPole-v0')
env = DummyVecEnv([lambda:env])
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=60000)


env = gym.make('CartPole-v0', render_mode = 'human')
obs, info = env.reset()

terminated = False
truncated = False
score = 0


for episode in range(3):
    while not terminated:
        action, next_state = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        score = score + reward
print("Score:{}".format(score))

pygame.display.quit()
env.close()









