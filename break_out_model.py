import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
import os

#this uses A2C as the algorithm
#uses atari environments
#this uses vector frame stack and will use 4 vectorized frames (cart pole used 1 at a time)


#python -m atari_py.import_roms .\ROMS\ROMS

environment_name = 'Breakout-v4'
env = gym.make(environment_name, render_mode = "human")

#env.reset()
print(env.action_space)
print(env.observation_space)

# episodes = 5
# for episode in range(1, episodes+1):
#     obs = env.reset()
#     done = False
#     score = 0
#
#     while not done:
#         env.render()
#         action = env.action_space.sample()
#         obs, reward, done, truncated, info = env.step(action)
#         score += reward
#     print('Episode:{} Score:{}'.format(episode, score))

    #Vectorise Environment and Train Model
#print(env.reset())

env = make_atari_env('Breakout-v4', n_envs=4, seed=0)
env = VecFrameStack(env, n_stack=4)
env.reset()
#env.render()

#TRAINING the model
log_path = os.path.join('Training', 'Logs')
model = A2C('CnnPolicy', env, verbose=1, tensorboard_log=log_path)

model.learn(total_timesteps=100000)

a2c_path = os.path.join('Training', 'SavedModels', 'A2C_Breakout_Model')
model.save(a2c_path)

# #to delete and reload later...
# #del model
# #model = A2C.load(a2c_path, env)
#
# #EVALUATE model
# env = make_atari_env('Breakout-v4', n_envs=1, seed=0)
# env=VecFrameStack(env, n_stack=4)
#
# evaluate_policy(model, env, n_eval_episodes=10, render=True)

