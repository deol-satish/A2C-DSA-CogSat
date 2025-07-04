# %%
import os

# Folder name
saved_folder = "saved_data"

# Create the folder if it doesn't exist
if not os.path.exists(saved_folder):
    os.makedirs(saved_folder)
    print(f"Folder '{saved_folder}' created.")
else:
    print(f"Folder '{saved_folder}' already exists.")


import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class RewardLoggerCallback(BaseCallback):
    def __init__(self, epoch_length, verbose=0):
        super().__init__(verbose)
        self.epoch_length = epoch_length
        self.epoch_rewards = []
        self.epoch_mean_rewards = []
        self.epoch_median_rewards = []
        self.epoch_all_rewards = []
        self.current_rewards = []

    def _on_step(self) -> bool:
        if self.locals.get("rewards") is not None:
            self.current_rewards.append(self.locals["rewards"][0])
        
        # Every epoch_length steps, calculate mean and reset
        if self.num_timesteps % self.epoch_length == 0:
            if self.current_rewards:
                mean_reward = np.mean(self.current_rewards)
                median_reward = np.median(self.current_rewards)
                self.epoch_rewards.append(mean_reward)
                self.epoch_mean_rewards.append(mean_reward)
                self.epoch_median_rewards.append(median_reward)
                self.epoch_all_rewards.append(self.current_rewards.copy())
                self.current_rewards = []

        return True

    def _on_training_end(self):
        # Save rewards to file (optional)
        # Save the numpy arrays using f-strings
        np.save(f'{saved_folder}/epoch_rewards.npy', self.epoch_rewards)
        np.save(f'{saved_folder}/epoch_mean_rewards.npy', self.epoch_mean_rewards)
        np.save(f'{saved_folder}/epoch_median_rewards.npy', self.epoch_median_rewards)
        np.save(f'{saved_folder}/epoch_all_rewards.npy', self.epoch_all_rewards)


# %%
import gymnasium
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from utils.env import CogSatEnv

# set the seed
seed = 42

gymnasium.register(
    id='CogSatEnv-v1',  # Use the same ID here as you used in the script
    entry_point='utils.env:CogSatEnv',
)
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Initialize the environment
env_id = "CogSatEnv-v1"
env = make_vec_env(env_id, n_envs=1, seed=seed)

epoch_length = 122 ## got through experiment
epoch_numbers = 4000

# Instantiate callback
reward_logger = RewardLoggerCallback(epoch_length=epoch_length)

# Specify the policy network architecture, here we are using the default MIP
model = A2C("MultiInputPolicy", env, verbose=1,ent_coef=0.01, tensorboard_log="./a2c_dsa_tensorboard/",
            seed=seed)

# Define the total number of timesteps to train the model
total_timesteps = epoch_length*epoch_numbers

# Train the model
model.learn(total_timesteps=total_timesteps,callback=reward_logger)

# Save the model
model.save("a2c_cogsatenv_1")

env.close()


