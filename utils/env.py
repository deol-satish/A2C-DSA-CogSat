import numpy as np
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Dict, Box

class CogSatDSAEnv(gym.Env):
    def __init__(self, env_config=None, render_mode=None):
        super(CogSatDSAEnv, self).__init__()
        
        # # Set static LEO/GEO counts, or pull from env_config if using Ray
        # self.n_leo = env_config.get("n_leo", 3) if env_config else 3
        # self.n_leo_users = env_config.get("n_leo_users", 2) if env_config else 2
        # self.n_geo = env_config.get("n_geo", 2) if env_config else 2
        # self.n_geo_users = env_config.get("n_geo_users", 1) if env_config else 1

        self.n_leo = 3
        self.n_leo_users = 2
        self.n_geo = 1
        self.n_geo_users = 2

        # Register env spec if missing (useful for SB3 compatibility)
        if not hasattr(self, 'spec') or self.spec is None:
            self.spec = gym.envs.registration.EnvSpec("CogSatDSAEnv-v1")

        # Action space: 4 devices picking from 11 channels (0 = no transmission)
        self.action_space = MultiDiscrete([11] * 4)

        # Observation space structure
        self.observation_space = Dict({
            "utc_time": Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.int64),
            "leo_pos": Box(low=-np.inf, high=np.inf, shape=(self.n_leo *2,), dtype=np.float64),
            "geo_freq": Box(low=-np.inf, high=np.inf, shape=(self.n_geo), dtype=np.float64),
            "leo_freq": Box(low=-np.inf, high=np.inf, shape=(self.n_leo,), dtype=np.float64),
        })

        self.terminated = False
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.terminated = False

        obs = {
            "utc_time": np.array([0], dtype=np.int64),
            "leo_pos": np.random.randn(self.n_leo).astype(np.float32),
            "leo_rssi": np.random.randn(self.n_leo * self.n_leo_users).astype(np.float32),
            "leo_sinr": np.random.randn(self.n_leo * self.n_leo_users).astype(np.float32),
            "geo_rssi": np.random.randn(self.n_geo * self.n_geo_users).astype(np.float32),
            "geo_sinr": np.random.randn(self.n_geo * self.n_geo_users).astype(np.float32),
        }
        return obs, {}

    def step(self, action):
        self.current_step += 1
        reward = 0.0  # Placeholder
        self.terminated = self.current_step >= 300

        obs = {
            "utc_time": np.array([self.current_step], dtype=np.int64),
            "leo_pos": np.random.randn(self.n_leo).astype(np.float32),
            "leo_rssi": np.random.randn(self.n_leo * self.n_leo_users).astype(np.float32),
            "leo_sinr": np.random.randn(self.n_leo * self.n_leo_users).astype(np.float32),
            "geo_rssi": np.random.randn(self.n_geo * self.n_geo_users).astype(np.float32),
            "geo_sinr": np.random.randn(self.n_geo * self.n_geo_users).astype(np.float32),
        }

        return obs, reward, self.terminated, False, {}

    def render(self):
        pass  # Optional: plot satellite movement, SINR, channel use, etc.

    def close(self):
        pass
