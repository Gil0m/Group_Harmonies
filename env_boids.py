# env_boids.py
from gymnasium import spaces
from boids import Flock
import numpy as np


class BoidEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}


    def __init__(self, n=30, controlled_id=0, bounds=[20,20,20]):
        super().__init__()
        self.flock = Flock(n=n, bounds=bounds, seed=None)
        self.controlled_id = controlled_id
        # observation: positions and velocities of k nearest neighbors + self velocity
        self.k = 6
        obs_dim = (self.k * 6) + 6 # for neighbors: (rel_pos(3)+rel_vel(3)), plus self pos+vel
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        # action: 3D acceleration vector bounded
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.max_steps = 1000
        self.step_count = 0


    def _get_obs(self):
        positions, velocities = self.flock.get_state_arrays()
        my_pos = positions[self.controlled_id]
        my_vel = velocities[self.controlled_id]
        rel_pos = positions - my_pos
        dist = np.linalg.norm(rel_pos, axis=1)
        idx = np.argsort(dist)
        neighbors = []
        for i in idx[1:self.k+1]:
            rp = rel_pos[i]
            rv = velocities[i] - my_vel
            neighbors.append(np.concatenate([rp, rv]))
        # pad if fewer
        while len(neighbors) < self.k:
            neighbors.append(np.zeros(6))
        obs = np.concatenate([np.concatenate(neighbors), my_pos, my_vel])
        return obs.astype(np.float32)


    def step(self, action):
        # scale action into accel
        accel = np.array(action, dtype=np.float32) * self.flock.max_acc
        actions = {self.controlled_id: accel}
        self.flock.step(external_actions=actions)
        obs = self._get_obs()
        reward = self._compute_reward()
        self.step_count += 1
        done = (self.step_count >= self.max_steps)
        info = {}
        return obs, reward, done, False, info


    def reset(self, seed=None, options=None):
        self.flock = Flock(n=self.flock.n, bounds=self.flock.bounds, seed=seed)
        self.step_count = 0
        return self._get_obs(), {}


    def _compute_reward(self):
        # reward: stay close to flock center, avoid collisions, match velocity
        positions, velocities = self.flock.get_state_arrays()
        my_pos = positions[self.controlled_id]
        others = np.delete(positions, self.controlled_id, axis=0)
        center = np.mean(others, axis=0)
        self.dist_to_center = np.linalg.norm(my_pos - center)