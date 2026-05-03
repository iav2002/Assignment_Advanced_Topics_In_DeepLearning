import gymnasium as gym
import numpy as np
from gymnasium import spaces

class WarehouseEnv(gym.Env):
    def __init__(self, grid_map, embeddings_dict):
        super(WarehouseEnv, self).__init__()
        
        # grid_map IDs aligned with doc: 0=Floor, 1=Wall, 2=Pallet, 3=Sign
        self.grid_map = np.array(grid_map)
        self.grid_size = self.grid_map.shape[0]
        
        # Find the pallet location dynamically from the map
        # This aligns with the "collect pallets" task in the doc [cite: 71]
        self.target_coords = np.argwhere(self.grid_map == 2)[0] 
        
        self.embeddings_dict = embeddings_dict
        self.embed_dim = list(embeddings_dict.values())[0].shape[0]
        
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(2 + self.embed_dim,), 
            dtype=np.float32
        )
        
        self.agent_pos = np.array([0, 0])

    def reset(self, seed=None, options=None):
        # Skeleton was missing reset(), required by Gymnasium API
        # so episodes can restart properly
        super().reset(seed=seed)
        self.agent_pos = np.array([0, 0])
        return self._get_obs(), {}

    def step(self, action):
        prev_pos = self.agent_pos.copy()
        hit_wall = False
        
        new_pos = self.agent_pos.copy()
        if action == 0:   new_pos[1] = min(self.grid_size - 1, new_pos[1] + 1) # Up
        elif action == 1: new_pos[1] = max(0, new_pos[1] - 1)                 # Down
        elif action == 2: new_pos[0] = max(0, new_pos[0] - 1)                 # Left
        elif action == 3: new_pos[0] = min(self.grid_size - 1, new_pos[0] + 1) # Right
        
        # Check for Wall (ID 1) as defined in Part 2 [cite: 51]
        if self.grid_map[new_pos[1], new_pos[0]] == 1:
            hit_wall = True
        else:
            self.agent_pos = new_pos

        # Mission: Reach the Pallet (ID 2) 
        terminated = np.array_equal(self.agent_pos, self.target_coords)
        truncated = False 
        
        reward = self._calculate_reward(prev_pos, self.agent_pos, terminated, hit_wall)
        
        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        obj_id = self.grid_map[self.agent_pos[1], self.agent_pos[0]]
        embedding = self.embeddings_dict[obj_id]
        
        return np.concatenate([
            self.agent_pos.astype(np.float32), 
            embedding.astype(np.float32)
        ])

    def _calculate_reward(self, prev_pos, current_pos, terminated, hit_wall):
        # Provisional reward: minimal version to get the env running
        # Final version comes from the ablation study later in the notebook
        if terminated:
            return 10.0     # reaching the pallet
        if hit_wall:
            return -1.0     # bumping into a wall
        return -0.05        # small step penalty for efficiency