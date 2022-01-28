
import numpy as np
import os
import torch
from gym import utils
from gym.envs.mujoco import mujoco_env
import random


class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, mass_scale_set=[0.75, 0.85, 1.0, 1.15, 1.25], damping_scale_set=[0.75, 0.85, 1.0, 1.15, 1.25]):
        self.prev_qpos = None
        mujoco_env.MujocoEnv.__init__(self, '../assets/half_cheetah.xml', 5)

        self.original_mass = np.copy(self.model.body_mass)
        self.original_damping = np.copy(self.model.dof_damping)

        self.mass_scale_set = mass_scale_set
        self.damping_scale_set = damping_scale_set

        utils.EzPickle.__init__(self, mass_scale_set, damping_scale_set)

    def _set_observation_space(self, observation):
        super(HalfCheetahEnv, self)._set_observation_space(observation)
        proc_observation = self.obs_preproc(observation[None])
        self.proc_observation_space_dims = proc_observation.shape[-1]

    def step(self, action):
        self.prev_qpos = np.copy(self.sim.data.qpos.flat)
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()
        
        reward_ctrl = -0.1  * np.square(action).sum()
        reward_run = obs[0]
        reward = reward_run + reward_ctrl

        done = False
        return obs, reward, done, {}

    def _get_obs(self):
        return np.concatenate([
            (self.sim.data.qpos.flat[:1] - self.prev_qpos[:1]) / self.dt,
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
        ])

    def obs_preproc(self, obs):
        if isinstance(obs, np.ndarray):
            return np.concatenate([obs[..., 1:2], np.sin(obs[..., 2:3]), np.cos(obs[..., 2:3]), obs[..., 3:]], axis=-1)
        else:
            return torch.cat((obs[..., 1:2], torch.sin(obs[..., 2:3]), torch.cos(obs[..., 2:3]), obs[..., 3:]),-1)

    def obs_postproc(self, obs, pred):
        if isinstance(obs, np.ndarray):
            return np.concatenate([pred[..., :1], obs[..., 1:] + pred[..., 1:]], axis=-1)
        else:
            return torch.cat((pred[..., :1], obs[..., 1:] + pred[..., 1:]),-1)

    def targ_proc(self, obs, next_obs):
        return np.concatenate([next_obs[..., :1], next_obs[..., 1:] - obs[..., 1:]], axis=-1)

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.normal(loc=0, scale=0.001, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.normal(loc=0, scale=0.001, size=self.model.nv)
        self.set_state(qpos, qvel)
        self.prev_qpos = np.copy(self.sim.data.qpos.flat)

        random_index = self.np_random.randint(len(self.mass_scale_set))
        self.mass_scale = self.mass_scale_set[random_index]

        random_index = self.np_random.randint(len(self.damping_scale_set))
        self.damping_scale = self.damping_scale_set[random_index]

        self.change_env()
        return self._get_obs()
    
    def reset_idx_model(self,param_list,idx):
        # reset an env with aspecific index
        
        qpos = self.init_qpos + self.np_random.normal(loc=0, scale=0.001, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.normal(loc=0, scale=0.001, size=self.model.nv)
        self.set_state(qpos, qvel)
        self.prev_qpos = np.copy(self.sim.data.qpos.flat)
    
        self.mass_scale = param_list[idx][0]
    
        self.damping_scale = param_list[idx][1]
    
        self.change_env()
        return self._get_obs()        
    
    def test_reset_model(self, idx):
        param_list = np.array([[0.85,0.85],[0.85,0.95],[0.85,1.05],[0.85,1.15],
                               [0.95,0.85],[0.95,0.95],[0.95,1.05],[0.95,1.15],
                               [1.05,0.85],[1.05,0.95],[1.05,1.05],[1.05,1.15],
                               [1.15,0.85],[1.15,0.95],[1.15,1.05],[1.15,1.15]])
        
        qpos = self.init_qpos + self.np_random.normal(loc=0, scale=0.001, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.normal(loc=0, scale=0.001, size=self.model.nv)
        self.set_state(qpos, qvel)
        self.prev_qpos = np.copy(self.sim.data.qpos.flat)

        self.mass_scale = param_list[idx][0]

        self.damping_scale = param_list[idx][1]

        self.change_env()
        return self._get_obs()    

    def reward(self, obs, action, next_obs):
        ctrl_cost = 1e-1 * np.sum(np.square(action), axis=-1)
        forward_reward = next_obs[..., 0]
        reward = forward_reward - ctrl_cost
        return reward
    
    def torch_reward_fn(self):
        def _thunk(obs, act, next_obs):
            ctrl_cost = 1e-1  * ((torch.square(act)).sum(axis=-1))
            forward_reward = obs[..., 0]
            reward = forward_reward - ctrl_cost
            return reward
        return _thunk     
    
    def change_env(self):
        mass = np.copy(self.original_mass)
        damping = np.copy(self.original_damping)
        mass *= self.mass_scale
        damping *= self.damping_scale

        self.model.body_mass[:] = mass
        self.model.dof_damping[:] = damping

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.25
        self.viewer.cam.elevation = -55
    
    def get_sim_parameters(self):
        return np.array([self.mass_scale, self.damping_scale])
    
    def num_modifiable_parameters(self):
        return 2

    def log_diagnostics(self, paths, prefix):
        return
    

def metacheetah_cost_torch(obs, act, next_obs):
    ctrl_cost = 1e-1  * ((torch.square(act)).sum(axis=-1))
    forward_reward = next_obs[..., 0]
    reward = forward_reward - ctrl_cost
        
    return -reward     


def metacheetah_reward_torch(obs, act, next_obs):
    ctrl_cost = 1e-1  * ((torch.square(act)).sum(axis=-1))
    forward_reward = next_obs[..., 0]
    reward = forward_reward - ctrl_cost
        
    return reward


def sample_env():

    halfcheetah_env=HalfCheetahEnv(mass_scale_set=[0.80, 0.90, 1.0, 1.10, 1.20], 
                                   damping_scale_set=[0.80, 0.90, 1.0, 1.10, 1.20])
    halfcheetah_env.reset_model()
            
    return halfcheetah_env
    
    
def sample_batch_env(num_envs):

    env_list = [sample_env() for i in range(num_envs)]

    return env_list

    
        
def sample_test_env(idx):

    cheetah_env=HalfCheetahEnv(mass_scale_set=[0.85, 0.95, 1.05, 1.15], 
                                damping_scale_set=[0.85, 0.95, 1.05, 1.15])    
    cheetah_env.test_reset_model(idx)

    return cheetah_env


        
def sample_batch_idx_env(num_envs):
    param_list = np.array([[0.8,0.8],[0.8,0.9],[0.8,1.0],[0.8,1.1],[0.8,1.2],
                           [0.9,0.8],[0.9,0.9],[0.9,1.0],[0.9,1.1],[0.9,1.2],
                           [1.0,0.8],[1.0,0.9],[1.0,1.0],[1.0,1.1],[1.0,1.2],
                           [1.1,0.8],[1.1,0.9],[1.1,1.0],[1.1,1.1],[1.1,1.2],
                           [1.2,0.8],[1.2,0.9],[1.2,1.0],[1.2,1.1],[1.2,1.2]])
    
    env_list = []
    idx_list = random.sample(range(len(param_list)), num_envs)
    for idx in idx_list:
        cheetah_env=HalfCheetahEnv(mass_scale_set=[0.80, 0.90, 1.0, 1.10, 1.20], 
                                   damping_scale_set=[0.80, 0.90, 1.0, 1.10, 1.20])
        cheetah_env.reset_idx_model(param_list, idx)
        env_list.append(cheetah_env)
        
    return env_list 

