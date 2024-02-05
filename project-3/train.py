import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv



env_name = "LunarLander-v2"
env = gym.make(env_name)
env = DummyVecEnv([lambda : env])

model = DQN(
    "MlpPolicy",
    env=env,
    learning_rate=5e-4,
    batch_size=128,
    buffer_size=50000,
    learning_starts=0,
    target_update_interval=250,
    policy_kwargs={"net_arch" : [256, 256]},
    verbose=1,
    tensorboard_log="./tensorboard/LunarLander-v2/"
)

model.learn(total_timesteps = 1e5)

env.close()



















# episodes = 10
#
# for episode in range(episodes + 1):
#
#    observation, info = env.reset()
#    terminated = False
#    score = 0
#
#    while not terminated:
#
#       action = env.action_space.sample()
#       obs, reward, terminated, truncated, info = env.step(action)
#       score += reward
#
#    print("Episode:{}, Score:{}".format(episode, score))

