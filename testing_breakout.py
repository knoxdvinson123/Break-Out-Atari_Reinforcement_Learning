import os
import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env

environment_name = 'Breakout-v4'
env = gym.make(environment_name, render_mode = "human")
env = make_atari_env('Breakout-v4', n_envs=1, seed=0)
env = VecFrameStack(env, n_stack=4)

A2C_path = os.path.join('Training', 'SavedModels', 'A2C_Breakout_Model')
model = A2C.load(A2C_path, env=env)

#Example of running random actions for 5 episodes and printing score--------------------------------------
episodes = 5
for episode in range(1, episodes+1):
    obs = env.reset()
    done = False
    score = 0

    while not done:
      env.render()
      #action, _states = model.predict(obs)
      action, _ = model.predict(obs) #now using the trained model
      obs, reward, done, info = env.step(action)
      score += reward
    print('Episode:{} Score:{}'.format(episode, score))