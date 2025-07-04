# train_a2c.py
import os
import argparse
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
import gymnasium
import logging



# Register your custom environment
gymnasium.register(
    id='CogSatEnv-v1',
    entry_point='utils.env:CogSatEnv',
)

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--channels', type=int, required=True)
parser.add_argument('--leo_users', type=int, required=True)
parser.add_argument('--geo_users', type=int, required=True)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

# Config
env_config = {
    "no_channels": args.channels,
    "no_leo_user": args.leo_users,
    "no_geo_user": args.geo_users,
    "saved_folder": f"saved_data_{args.channels}_channels_{args.leo_users}_leo_user_{args.geo_users}_geo_user"
}



# Configure the logger
logging.basicConfig(
    filename=f"train_log_{args.channels}_channels_{args.leo_users}_leo_user_{args.geo_users}_geo_user",
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    filemode='w'  # Overwrites the file each time
)

# Create save folder
os.makedirs(env_config["saved_folder"], exist_ok=True)

# Set environment variable
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Create environment
env = make_vec_env("CogSatEnv-v1", n_envs=1, seed=args.seed, env_kwargs={"env_config": env_config})

# Model parameters
epoch_length = 122
epoch_numbers = 4000
total_timesteps = epoch_length * epoch_numbers

model = A2C("MultiInputPolicy", env, verbose=1,
            ent_coef=0.01,
            tensorboard_log=os.path.join(env_config["saved_folder"], "tensorboard"),
            seed=args.seed)

# Train
model.learn(total_timesteps=total_timesteps)

# Save
model.save(os.path.join(env_config["saved_folder"], "a2c_model"))

env.close()
