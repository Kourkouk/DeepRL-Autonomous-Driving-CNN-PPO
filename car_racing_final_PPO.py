import sys
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
import model_final_PPO

# # The saved .zip model hardcoded the original module name ('model_test_PPO') during training.
# We use sys.modules to redirect it to our updated 'model_final_PPO' file without retraining.
# Bypassing...
sys.modules['model_test_PPO'] = model_final_PPO

# Creating the environment where the user sees the trained results(render_mode="human").
print("Creating the environment with graphics...")
def make_env():
    return gym.make("CarRacing-v3", continuous=True, render_mode="human")

# Frame stacking - 4(short memory for the model).
env = DummyVecEnv([make_env])
env = VecFrameStack(env, n_stack=4)
env = VecTransposeImage(env)

print("Importing trained enhanced model_PPO...")
# Loading the best model.
model = PPO.load("best_model.zip", env=env)
# Loading the ppo_carracing_final_test_model model
#model = PPO.load("ppo_carracing_final_test_model.zip", env=env)

obs = env.reset()

# The loop keeps the simulation running until the episode naturally terminates (track completion, driving out of bounds, or timeout).
done = [False]
while not done[0]:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)

env.close()
print("The race is over!")