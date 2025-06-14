import gymnasium
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3 import A2C
from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit

from utils.env import CogSatEnv  # Ensure this import path is correct

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ---- Subclass A2C to Record Loss ----
class A2CWithLossTracking(A2C):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.losses = {
            "policy_loss": [],
            "value_loss": [],
            "entropy_loss": []
        }

    def train(self):
        super().train()
        log_data = self.logger.name_to_value
        if "train/value_loss" in log_data:
            self.losses["value_loss"].append(log_data["train/value_loss"])
        if "train/policy_gradient_loss" in log_data:
            self.losses["policy_loss"].append(log_data["train/policy_gradient_loss"])
        if "train/entropy_loss" in log_data:
            self.losses["entropy_loss"].append(log_data["train/entropy_loss"])


# ---- Callback to Track Rewards & Losses ----
class RewardAndLossCallback(BaseCallback):
    def __init__(self, model_ref, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.model_ref = model_ref

    def _on_step(self) -> bool:
        for i, done in enumerate(self.locals.get("dones", [])):
            if done:
                ep_info = self.locals["infos"][i].get("episode")
                if ep_info is not None:
                    self.episode_rewards.append(ep_info["r"])
        return True

    def _on_training_end(self):
        rewards = self.episode_rewards
        self.mean_rewards = [np.mean(rewards[max(0, i - 10):i]) for i in range(1, len(rewards) + 1)]
        self.median_rewards = [np.median(rewards[max(0, i - 10):i]) for i in range(1, len(rewards) + 1)]
        self.policy_losses = self.model_ref.losses["policy_loss"]
        self.value_losses = self.model_ref.losses["value_loss"]
        self.entropy_losses = self.model_ref.losses["entropy_loss"]


def main():
    # ---- Register and Initialize Environment ----
    seed = 42
    env_id = 'CogSatEnv-v1'

    gymnasium.register(
        id=env_id,
        entry_point='utils.env:CogSatEnv',
    )
    env = make_vec_env(env_id, n_envs=1, seed=seed)

    # ---- Hyperparameters ----
    epoch_length = 62
    epoch_numbers = 2
    total_timesteps = epoch_length * epoch_numbers

    # ---- Model and Callbacks ----
    model = A2CWithLossTracking("MultiInputPolicy", env,
                                ent_coef=0.01,
                                verbose=1,
                                seed=seed,
                                learning_rate=0.0001,
                                tensorboard_log="./a2c_leogeo_tensorboard/")

    reward_loss_callback = RewardAndLossCallback(model)
    checkpoint_callback = CheckpointCallback(save_freq=epoch_length, save_path='./logs/', name_prefix='rl_model_A2C')

    # ---- Train ----
    model.learn(total_timesteps=total_timesteps, callback=[reward_loss_callback, checkpoint_callback])
    model.save("a2c_cogsatenv_1")
    env.close()

    # ---- Plot ----
    plt.figure(figsize=(12, 6))

    # Plot rewards
    plt.subplot(1, 2, 1)
    plt.plot(reward_loss_callback.mean_rewards, label="Mean Reward")
    plt.plot(reward_loss_callback.median_rewards, label="Median Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Rewards over Episodes")
    plt.grid(True)
    plt.legend()

    # Plot losses
    plt.subplot(1, 2, 2)
    plt.plot(reward_loss_callback.policy_losses, label="Policy Loss")
    plt.plot(reward_loss_callback.value_losses, label="Value Loss")
    plt.plot(reward_loss_callback.entropy_losses, label="Entropy Loss")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Losses during Training")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig("reward_loss_plot.png")
    plt.show()


if __name__ == "__main__":
    main()
