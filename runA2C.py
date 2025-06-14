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

# Initialize the environment
env_id = "CogSatEnv-v1"
env = make_vec_env(env_id, n_envs=1, seed=seed)

epoch_length = 62 ## got through experiment
epoch_numbers = 5

# Set up the checkpoint callback
checkpoint_callback = CheckpointCallback(save_freq=epoch_length, save_path='./logs/', name_prefix='rl_model_A2C')

# Specify the policy network architecture, here we are using the default MIP
model = A2C("MultiInputPolicy", env, ent_coef=0.01, verbose=1, tensorboard_log="./a2c_leogeo_tensorboard/",
            seed=seed, learning_rate=0.0001)

# Define the total number of timesteps to train the model
total_timesteps = epoch_length*epoch_numbers

# Train the model
model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

# Save the model
model.save("a2c_cogsatenv_1")

env.close()