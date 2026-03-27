import os
import shutil
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from model_final_PPO import CarRacingCNN

# Directories for logging evaluation metrics and saving the best performing model.
log_dir = "ppo_carracing_final_log_test/"
best_model_dir = "best_model_saved/"

# Removes old logs to prevent overlapping data from previous training sessions.
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)
os.makedirs(best_model_dir, exist_ok=True)

# Initializing the agent environment without the graphics(render_mode="human") to run in the background faster.
def make_env():
    env = gym.make("CarRacing-v3", continuous=True)
    env = Monitor(env) # Enables the ep_rew_mean(reward for the model-reinforced learning).
    return env

# Using the frame stacking technique to provide the agent with temporal context (velocity/direction) by observing 4 consecutive frames.
env = DummyVecEnv([make_env])
env = VecFrameStack(env, n_stack=4)
env = VecTransposeImage(env)

# Initializing a separate evaluation environment which will test the model to find the best one the model achieves.
eval_env = DummyVecEnv([make_env])
eval_env = VecFrameStack(eval_env, n_stack=4)
eval_env = VecTransposeImage(eval_env)

# Using linear learning rate greatly improves the overall learning rate of the model as it accelerates in the beginning and settles down in the end.
def linear_schedule(initial_value: float, final_value: float):
    def func(progress_remaining: float) -> float:
        return final_value + (initial_value - final_value) * progress_remaining
    return func

# Setting up the model build in the model_final_PPO.
policy_kwargs = dict(
    features_extractor_class=CarRacingCNN,
    features_extractor_kwargs=dict(features_dim=256), # The PPO trainer model takes 256 neurons output for training.
)

# Setting up EvalCallback.
# Every 10.000 steps it tests the model in 5 races. It saves the one with the best score yet.
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=best_model_dir,
    log_path=log_dir,
    eval_freq=10000,
    n_eval_episodes=5,
    deterministic=True,
    render=False
)

# Setting up the parameters for the PPO trainer.
# We use the linear learning rate approach.
print("Creating Highly Optimized PPO Agent...")
model = PPO(
    "CnnPolicy",
    env,
    learning_rate=linear_schedule(0.0002, 0.00007),
    ent_coef=0.01,       # Encourages exploration to prevent the agent from getting stuck in lazy/safe driving habits.
    clip_range=0.15,     # Prevents to some extent the Policy Collapse.
    policy_kwargs=policy_kwargs, # model_final_PPO.
    verbose=1,
    tensorboard_log=log_dir
)

# Training the model for 500.000 timesteps(roughly 2 hours waiting time-CPU).
total_timesteps = 500000
print(f"Starts training for {total_timesteps} timesteps...")

# Adding the EvalCallback in the training.
model.learn(total_timesteps=total_timesteps, callback=eval_callback)

# The ppo_carracing_final_test_model.zip is created and being overwritten after training.
model.save("ppo_carracing_final_test_model")
print("Training is complete and the final model is saved.")
env.close()
eval_env.close()