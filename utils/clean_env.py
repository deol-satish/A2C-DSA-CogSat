import gymnasium
import numpy as np
import matlab.engine
from gymnasium.spaces import MultiDiscrete, Dict, Box
import logging
import json
import math
from datetime import datetime, timedelta, timezone
import time
import os

# Folder name
saved_folder = "saved_data"

# Create the folder if it doesn't exist
if not os.path.exists(saved_folder):
    os.makedirs(saved_folder)
    print(f"Folder '{saved_folder}' created.")
else:
    print(f"Folder '{saved_folder}' already exists.")

env_name = "CogSatEnv-v1"

# Configure the logger
logging.basicConfig(
    filename='train_log.txt',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    filemode='w'  # Overwrites the file each time
)

 
class CogSatEnv(gymnasium.Env):
    """Gymnasium environment for MATLAB-based Cognitive Satellite Simulation"""
 
    def __init__(self, env_config=None, render_mode=None):
        super(CogSatEnv, self).__init__()
        if not hasattr(self, 'spec') or self.spec is None:
            self.spec = gymnasium.envs.registration.EnvSpec("CogSatEnv-v1")


        self.episode_number = 0
            
 
        # Start MATLAB engine and set path
        self.eng = matlab.engine.start_matlab()
        self.eng.cd(self.eng.pwd(), nargout=0)  # ensure working directory is set

        self.eng.addpath(r'./matlab_code', nargout=0)

        # Initialize the MATLAB scenario and Save Baseline
        self.eng.eval("initialiseScenario", nargout=0)
        self.eng.eval("P06_Intf_Eval", nargout=0)
        self.eng.eval("resetScenario", nargout=0)
        self.save_npy_data('Baseline')       

        filename = f"{saved_folder}/Baseline_workspace_Saved_data.mat"
        save_cmd = f"save('{filename}')"
        self.eng.eval(save_cmd, nargout=0)


        self.mat_filename = "A2C"       


        self.tIndex = 0
        self.timelength = self.eng.eval("length(ts)", nargout=1)
        self.NumLeoUser = int(self.eng.workspace['NumLeoUser'])
        self.NumGeoUser = int(self.eng.workspace['NumGeoUser'])
        self.reward = 0
        self.LeoChannels = int(self.eng.workspace['numChannels'])
        self.GeoChannels = int(self.eng.workspace['numChannels'])


        self.cur_obs = {
            "utc_time": np.array([0], dtype=np.int64),
            "freq_lgs_leo": np.random.uniform(1.0, self.LeoChannels, size=(self.NumLeoUser,)).astype(np.int64),
            "freq_ggs_geo": np.random.uniform(1.0, self.GeoChannels, size=(self.NumGeoUser,)).astype(np.int64),
            "leo_pos": np.random.uniform(0, 20, size=(self.NumLeoUser*2,)).astype(np.float32),
            
        }
      
        
 
        # Define action and observation space
        self.action_space = gymnasium.spaces.MultiDiscrete([self.LeoChannels] * self.NumLeoUser)  # Select a channel index for each LEO user


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
        # Log cur_state_from_matlab
        #logging.info("=== Current State ===")
        # logging.info(json.dumps(cur_state_from_matlab, indent=2))
        """Reset the environment and initialize the buffer."""

        self.ts = self.get_matlab_ts()

        self.FreqAlloc = np.array(self.eng.workspace['FreqAlloc'])
        self.LEOFreqAlloc = self.FreqAlloc[:10,:]
        self.GEOFreqAlloc = self.FreqAlloc[10:20,:]

        self.cur_obs["utc_time"] = np.array([self.ts[self.tIndex]], dtype=np.int64)
        self.cur_obs["freq_lgs_leo"] = np.array(self.LEOFreqAlloc[:,self.tIndex], dtype=np.int64)
        self.cur_obs["freq_ggs_geo"] = np.array(self.GEOFreqAlloc[:,self.tIndex], dtype=np.int64)
        leo_loc = np.array(self.eng.workspace['LEO_LOC'])
        self.cur_obs["leo_pos"] = np.array(leo_loc[0:self.NumLeoUser,self.tIndex].flatten(), dtype=np.float32)
        # Log current observation
        # t = np.array(self.eng.workspace['t'])
        # logging.info("===get_state_from_matlab: t from matlab: %s ===",t)
        # logging.info("===get_state_from_matlab: t from python env: %s ===",self.tIndex)
        # logging.info("get_state_from_matlab:self.FreqAlloc: %s",self.FreqAlloc)

        # # Log utc_time
        # logging.info("utc_time: %s", cur_obs["utc_time"].tolist())

        # # Log freq_lgs_leo
        # logging.info("freq_lgs_leo: %s", cur_obs["freq_lgs_leo"].tolist())
        # logging.info("freq_ggs_geo: %s", cur_obs["freq_ggs_geo"].tolist())

        #logging.info("cur_obs: %s", self.cur_obs)
        #print("cur_obs",self.cur_obs)

        # (Optional) Validate against observation_space
        assert self.observation_space.contains(self.cur_obs), "cur_obs doesn't match the observation space!"

        return self.cur_obs
    

 
    def step(self, action):
        """
        Apply action and return (observation, reward, terminated, truncated, info)
        """
        self.eng.workspace['t'] = int(self.tIndex) + 1
        start_time = time.time()

        # Action start from 0 and ends before self.LeoChannels, that means it is not included
        # For example, if self.LeoChannels is 5, action can be 0, 1, 2, 3, or 4.
        # This is because MATLAB uses 1-based indexing, so we need to convert it to 0-based indexing for Python.

        #logging.info("=== Step Started ===")
        t = np.array(self.eng.workspace['t'])
        #logging.info("===step: t from matlab: %s ===",t)
        #logging.info("=== step: t from python env: %s ===",self.tIndex)
        #logging.info("step: self.FreqAlloc: %s",self.FreqAlloc)
        



        #print("Action taken: ", action)
        # logging.info("=== Action Taken === %s", action)
        action = action + 1 
        #print("After +1 Operation: Action taken: ", action)
        #logging.info("=== After +1 Operation: Action Taken === %s", action)
        


        # Access the variable from MATLAB workspace
        # Convert MATLAB array to NumPy array
        channel_list_leo = np.array(self.eng.workspace['ChannelListLeo'])
        Serv_idxLEO = np.array(self.eng.workspace['Serv_idxLEO'])




        # Store original values for comparison
        before_vals = []
        for i in range(self.NumLeoUser):
            sat = Serv_idxLEO[i, self.tIndex].astype(int) - 1
            before_vals.append(channel_list_leo[i, sat, self.tIndex])

        # Now do the assignment 
        for i in range(self.NumLeoUser):
            sat = Serv_idxLEO[i, self.tIndex].astype(int) - 1
            channel_list_leo[i, sat, self.tIndex] = action[i]

        # Compare
        for i in range(self.NumLeoUser):
            sat = Serv_idxLEO[i, self.tIndex].astype(int) - 1
            #print(f"User {i}: Before = {before_vals[i]}, After = {channel_list_leo[i, sat, self.tIndex]}")
            #logging.info(f"User {i}: Before = {before_vals[i]}, After = {channel_list_leo[i, sat, self.tIndex]}")



        self.eng.workspace['ChannelListLeo'] = matlab.double(channel_list_leo)




        self.eng.eval("stepScenario", nargout=0)



        next_observation = self.get_state_from_matlab()  

        FreqAlloc = np.array(self.eng.workspace['FreqAlloc'])[:,self.tIndex]
        #logging.info("=== FreqAlloc === %s",FreqAlloc)
        # Example: FreqAlloc = np.array([...])
        unique_values, counts = np.unique(FreqAlloc, return_counts=True)

        # Count how many values are repeated (i.e., count > 1)
        num_repeated = np.sum(counts > 1)   

        # Split into LEO (first 10) and GEO (last 10)
        leo_vals = FreqAlloc[:10]
        geo_vals = FreqAlloc[10:]

        # Find common elements
        common_vals = np.intersect1d(leo_vals, geo_vals)

        # Count
        num_common = len(common_vals)

        # print("Common values:", common_vals)

        #logging.info("=== Count of Common values === %s",num_common)

        
        #print("Next Observation: ", next_observation)
        #logging.info("=== Next Observation === %s", next_observation)   
         
        terminated = False
        truncated = False

        SINR = np.array(self.eng.workspace['SINR_mW_dict'])
        Intf = np.array(self.eng.workspace['Intf_mW_dict'])
        Thrpt = np.array(self.eng.workspace['Thrpt'])/(1024*1024)


        

        # Compute reward: sum of SINR across all users at current time step
        Interference_to_geo_users = Intf[self.NumLeoUser:self.NumGeoUser+ self.NumLeoUser, self.tIndex]
        SINR_of_LEO_users = SINR[:self.NumLeoUser, self.tIndex]
        Thrpt_of_LEO_users = Thrpt[:self.NumLeoUser, self.tIndex]

        #print('Intf: ', Interference_to_geo_users*1e8)
        #print('SINR: ', SINR_of_LEO_users)
        #print('Thrpt: ', Thrpt_of_LEO_users)
        #logging.info("=== Interference === %s", Interference_to_geo_users*1e8)
        #logging.info("=== SINR === %s", SINR_of_LEO_users)
        #logging.info("=== Throughput === %s", Thrpt_of_LEO_users)
        #logging.info("=== num_repeated === %s", num_repeated)

        # reward = np.sum(np.log10(SINR_of_LEO_users)) -  num_repeated

        # reward = np.mean(Thrpt_of_LEO_users) - (0.1*(np.std(Thrpt_of_LEO_users))) - (2*num_repeated)
        # reward = np.mean(SINR_of_LEO_users) - (0.1*(np.std(SINR_of_LEO_users))) - (2*num_repeated)


        # print("Count of common values:", num_common)

        #reward = np.sum(np.log10(Thrpt_of_LEO_users)) -  num_repeated

        reward = np.sum(np.log10(SINR_of_LEO_users)) -  num_repeated


        self.reward = reward
        #logging.info("=== Reward === %s", reward)

        




        self.tIndex += 1
        
        if self.tIndex >= self.timelength - 1:
            terminated = True
            #print("Episode finished after {} timesteps".format(self.tIndex))
            #logging.info("=== Episode finished after %s timesteps ===", self.tIndex)
            self.save_npy_data(f'Episode_{self.episode_number}')
            # filename = f"{saved_folder}/{self.mat_filename}_{self.episode_number}_workspace_Saved_data_Close.mat"
            # save_cmd = f"save('{filename}')"
            # self.eng.eval(save_cmd, nargout=0)
            # self.eng.eval("P08_SaveData", nargout=0)

        info = {}

        #print("*-"*50)

        end_time = time.time()
        step_duration = end_time - start_time
        #logging.info("=== Time taken for timestep: %.4f seconds ===", step_duration)
        #print("=== Time taken for timestep: %.4f seconds ===", step_duration)

 
        return next_observation, reward, terminated, truncated, info
 
    def reset(self, *, seed=None, options=None):
        self.episode_number = self.episode_number + 1
        #print("Resetting environment for episode: ", self.episode_number)
        #logging.info("=== Resetting Environment for Episode %s ===", self.episode_number)
        super().reset(seed=seed) 
        # Reset the scenario
        self.eng.eval("resetScenario", nargout=0)
        self.eng.eval("stepScenario", nargout=0)
        self.eng.eval("resetScenario", nargout=0)

        self.tIndex = 0
        self.done = 0
        
        self.ep_end_time = time.time()
        self.ep_step_duration = self.ep_end_time - self.ep_start_time
        # logging.info("=== Episode Time taken for timestep: %.4f seconds ===", self.ep_step_duration)
        #logging.info("=== Episode Index: {} Time taken for timestep: {:.4f} seconds ===".format(self.episode_number, self.ep_step_duration))
        print("=== Episode Index: {} Time taken for timestep: {:.4f} seconds ===".format(self.episode_number, self.ep_step_duration))

        self.ep_start_time = time.time()

        observation = self.get_state_from_matlab()
        #print("++++===== ENV RESET+++===")
 
        return observation, {}

    def save_env_state(self):
        """
        Saves the current environment state, including MATLAB workspace variables.
        """
        return {
            "tIndex": self.tIndex,
            "ChannelListLeo": np.array(self.eng.workspace["ChannelListLeo"]),
        }

    def restore_env_state(self, state):
        """
        Restores the environment state from the saved state dictionary.
        """
        # Restore tIndex
        self.tIndex = state["tIndex"]

        # Restore MATLAB workspace variables
        self.eng.workspace["ChannelListLeo"] = matlab.double(state["ChannelListLeo"].tolist())
        self.eng.workspace["FreqAlloc"] = matlab.double(state["FreqAlloc"].tolist())

        # Sync observation state
        self.cur_obs["utc_time"] = np.array([self.ts[self.tIndex]], dtype=np.int64)
        self.cur_obs["freq_lgs_leo"] = np.array(state["FreqAlloc"][:self.NumLeoUser, self.tIndex], dtype=np.int64)
        self.cur_obs["freq_ggs_geo"] = np.array(state["FreqAlloc"][self.NumLeoUser:self.NumLeoUser+ self.NumGeoUser, self.tIndex], dtype=np.int64)

        #logging.info("=== Environment Restored to tIndex %s ===", self.tIndex)

 
    def render(self):
        pass
 
    def close(self):
        #print("Saving MATLAB Data.")
        #logging.info("=== Saving MATLAB Data ===")
        # self.eng.eval("P08_SaveData", nargout=0)
        filename = f"{saved_folder}/{self.mat_filename}_workspace_Saved_data_Close.mat"
        save_cmd = f"save('{filename}')"
        self.eng.eval(save_cmd, nargout=0)
        self.eng.quit()
    