import gymnasium
import numpy as np
import matlab.engine
from gymnasium.spaces import Dict, Box
from datetime import datetime
import time
import os

# Folder name
saved_folder = "saved_data"

# Create the folder if it doesn't exist
if not os.path.exists(saved_folder):
    os.makedirs(saved_folder)

 
class CogSatEnv(gymnasium.Env):
    """Gymnasium environment for MATLAB-based Cognitive Satellite Simulation"""
 
    def __init__(self, env_config=None, render_mode=None):
        super(CogSatEnv, self).__init__()
        if not hasattr(self, 'spec') or self.spec is None:
            self.spec = gymnasium.envs.registration.EnvSpec("CogSatEnv-v1")

        self.episode_number = 0
            
        # Start MATLAB engine and set path
        self.eng = matlab.engine.start_matlab()
        self.eng.cd(self.eng.pwd(), nargout=0)
        self.eng.addpath(r'./matlab_code', nargout=0)

        # Initialize the MATLAB scenario
        self.eng.eval("initialiseScenario", nargout=0)
        self.eng.eval("P06_Intf_Eval", nargout=0)
        self.eng.eval("resetScenario", nargout=0)

        self.mat_filename = "A2C"
        self.tIndex = 0
        self.timelength = self.eng.eval("length(ts)", nargout=1)
        self.NumLeoUser = int(self.eng.workspace['NumLeoUser'])
        self.NumGeoUser = int(self.eng.workspace['NumGeoUser'])
        self.LeoChannels = int(self.eng.workspace['numChannels'])
        self.GeoChannels = int(self.eng.workspace['numChannels'])

        self.intial_obs = {
            "utc_time": np.array([0], dtype=np.int64),
            "freq_lgs_leo": np.random.uniform(1.0, self.LeoChannels, size=(self.NumLeoUser,)).astype(np.int64),
            "freq_ggs_geo": np.random.uniform(1.0, self.GeoChannels, size=(self.NumGeoUser,)).astype(np.int64),
            "leo_pos": np.random.uniform(0, 20, size=(self.NumLeoUser*2,)).astype(np.float32),
        }
        self.eng.eval("stepScenario", nargout=0)
        self.save_npy_data('Baseline')        
        
        # Define action and observation space
        self.action_space = gymnasium.spaces.MultiDiscrete([self.LeoChannels] * self.NumLeoUser)

        # Observation space structure
        self.observation_space = Dict({
            "utc_time": Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.int64),
            "freq_lgs_leo": Box(low=1, high=self.LeoChannels+1, shape=(self.NumLeoUser,), dtype=np.int64),
            "freq_ggs_geo": Box(low=1, high=self.GeoChannels+1, shape=(self.NumGeoUser,), dtype=np.int64),
            "leo_pos": Box(low=-np.inf, high=np.inf, shape=(self.NumLeoUser*2,), dtype=np.float32),
        })

        self.ep_start_time = time.time()

    def save_npy_data(self,extra_tag="Original"):
        SINR = np.array(self.eng.workspace['SINR'])
        Intf = np.array(self.eng.workspace['Intf'])
        SINR_mW_dict = np.array(self.eng.workspace['SINR_mW_dict'])
        Intf_mW_dict = np.array(self.eng.workspace['Intf_mW_dict'])
        Thrpt = np.array(self.eng.workspace['Thrpt'])
        SE = np.array(self.eng.workspace['SE'])
        FreqAlloc = np.array(self.eng.workspace['FreqAlloc'])

        np.save(f'{saved_folder}/{extra_tag}_SINR.npy', SINR)
        np.save(f'{saved_folder}/{extra_tag}_Intf.npy', Intf)
        np.save(f'{saved_folder}/{extra_tag}_SINR_mW_dict.npy', SINR_mW_dict)
        np.save(f'{saved_folder}/{extra_tag}_Intf_mW_dict.npy', Intf_mW_dict)
        np.save(f'{saved_folder}/{extra_tag}_Thrpt.npy', Thrpt)
        np.save(f'{saved_folder}/{extra_tag}_SE.npy', SE)
        np.save(f'{saved_folder}/{extra_tag}_FreqAlloc.npy', FreqAlloc)
    
    def get_matlab_ts(self):
        """
        Get the MATLAB timestamp as a list of strings.
        """
        ts_str = self.eng.eval("cellstr(datestr(ts, 'yyyy-mm-ddTHH:MM:SS'))", nargout=1)
        python_datetimes = [datetime.fromisoformat(s) for s in ts_str]
        timestamps = [dt.timestamp() for dt in python_datetimes]
        return timestamps
    
    def get_state_from_matlab(self):
        self.ts = self.get_matlab_ts()
        self.FreqAlloc = np.array(self.eng.workspace['FreqAlloc'])
        self.LEOFreqAlloc = self.FreqAlloc[:10,:]
        self.GEOFreqAlloc = self.FreqAlloc[10:20,:]

        cur_obs = self.intial_obs.copy()
        cur_obs["utc_time"] = np.array([self.ts[self.tIndex]], dtype=np.int64)
        cur_obs["freq_lgs_leo"] = np.array(self.LEOFreqAlloc[:,self.tIndex], dtype=np.int64)
        cur_obs["freq_ggs_geo"] = np.array(self.GEOFreqAlloc[:,self.tIndex], dtype=np.int64)
        leo_loc = np.array(self.eng.workspace['LEO_LOC'])
        cur_obs["leo_pos"] = np.array(leo_loc[0:self.NumLeoUser,self.tIndex].flatten(), dtype=np.float32)

        assert self.observation_space.contains(cur_obs), "cur_obs doesn't match the observation space!"
        return cur_obs
    
    def step(self, action):
        start_time = time.time()

        # Convert action from 0-based to 1-based indexing for MATLAB
        action = action + 1 

        # Get MATLAB workspace variables
        channel_list_leo = np.array(self.eng.workspace['ChannelListLeo'])
        Serv_idxLEO = np.array(self.eng.workspace['Serv_idxLEO'])
        self.tIndex = int(self.tIndex)

        # Apply actions
        for i in range(self.NumLeoUser):
            sat = Serv_idxLEO[i, self.tIndex].astype(int) - 1
            channel_list_leo[i, sat, self.tIndex] = action[i]

        self.eng.workspace['ChannelListLeo'] = matlab.double(channel_list_leo)
        self.eng.eval("stepScenario", nargout=0)

        next_observation = self.get_state_from_matlab()  

        # Calculate interference penalty
        FreqAlloc = np.array(self.eng.workspace['FreqAlloc'])[:,self.tIndex]
        unique_values, counts = np.unique(FreqAlloc, return_counts=True)
        num_repeated = np.sum(counts > 1)   

        # Calculate reward
        SINR = np.array(self.eng.workspace['SINR_mW_dict'])
        SINR_of_LEO_users = SINR[:self.NumLeoUser, self.tIndex]
        reward = np.sum(np.log10(SINR_of_LEO_users)) - num_repeated

        # Check if episode is done
        self.tIndex += 1
        terminated = False
        if self.tIndex >= self.timelength:
            terminated = True
            self.save_npy_data(f'Episode_{self.episode_number}')
            filename = f"{saved_folder}/{self.mat_filename}_workspace_Saved_data_Close.mat"
            save_cmd = f"save('{filename}')"
            self.eng.eval(save_cmd, nargout=0)
            self.eng.eval("P08_SaveData", nargout=0)

        # Timing information
        end_time = time.time()
        step_duration = end_time - start_time
        print("=== Time taken for timestep: %.4f seconds ===" % step_duration)

        return next_observation, reward, terminated, False, {}
 
    def reset(self, *, seed=None, options=None):
        self.episode_number = self.episode_number + 1
        super().reset(seed=seed) 

        self.tIndex = 0
        
        self.ep_end_time = time.time()
        self.ep_step_duration = self.ep_end_time - self.ep_start_time
        print("=== Episode Time taken: %.4f seconds ===" % self.ep_step_duration)
        
        self.ep_start_time = time.time()

        observation = self.get_state_from_matlab()
        return observation, {}

    def save_env_state(self):
        return {
            "tIndex": self.tIndex,
            "ChannelListLeo": np.array(self.eng.workspace["ChannelListLeo"]),
        }

    def restore_env_state(self, state):
        self.tIndex = state["tIndex"]
        self.eng.workspace["ChannelListLeo"] = matlab.double(state["ChannelListLeo"].tolist())
        self.eng.workspace["FreqAlloc"] = matlab.double(state["FreqAlloc"].tolist())

        self.intial_obs["utc_time"] = np.array([self.ts[self.tIndex]], dtype=np.int64)
        self.intial_obs["freq_lgs_leo"] = np.array(state["FreqAlloc"][:self.NumLeoUser, self.tIndex], dtype=np.int64)
        self.intial_obs["freq_ggs_geo"] = np.array(state["FreqAlloc"][self.NumLeoUser:self.NumLeoUser+ self.NumGeoUser, self.tIndex], dtype=np.int64)
 
    def render(self):
        pass
 
    def close(self):
        filename = f"{saved_folder}/{self.mat_filename}_workspace_Saved_data_Close.mat"
        save_cmd = f"save('{filename}')"
        self.eng.eval(save_cmd, nargout=0)
        self.eng.quit()