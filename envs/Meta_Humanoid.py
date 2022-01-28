import numpy as np
import torch
from gym.envs.mujoco import mujoco_env
from gym import utils
import os
import random
import copy


def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]

class SlimHumanoidEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self, mass_scale_set=[0.80, 0.90, 1.0, 1.10, 1.20], damping_scale_set=[0.80, 0.90, 1.0, 1.10, 1.20]):
        self.prev_pos = None
        mujoco_env.MujocoEnv.__init__(self, '../assets/humanoid.xml', 5)

        self.original_mass = np.copy(self.model.body_mass)
        self.original_damping = np.copy(self.model.dof_damping)

        self.mass_scale_set = mass_scale_set
        self.damping_scale_set = damping_scale_set

        utils.EzPickle.__init__(self, mass_scale_set, damping_scale_set)
    
    def _set_observation_space(self, observation):
        super(SlimHumanoidEnv, self)._set_observation_space(observation)
        proc_observation = self.obs_preproc(observation[None])
        self.proc_observation_space_dims = proc_observation.shape[-1]

    def _get_obs(self):
        data = self.sim.data
        return np.concatenate([data.qpos.flat[2:],
                               data.qvel.flat])

    def obs_preproc(self, obs):
        return obs

    def obs_postproc(self, obs, pred):
        return obs + pred

    def targ_proc(self, obs, next_obs):
        return next_obs - obs

    def step(self, a):
        old_obs = np.copy(self._get_obs())
        self.do_simulation(a, self.frame_skip)
        data = self.sim.data
        lin_vel_cost = 0.25 / 0.015 * old_obs[..., 22]
        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = 0.
        qpos = self.sim.data.qpos
        done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))
        alive_bonus = 5.0 * (1 - float(done))
        done = False
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
        return self._get_obs(), reward, done, dict(reward_linvel=lin_vel_cost, reward_quadctrl=-quad_ctrl_cost, reward_alive=alive_bonus, reward_impact=-quad_impact_cost)

    def reset_model(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv,)
        )
        pos_before = mass_center(self.model, self.sim)
        self.prev_pos = np.copy(pos_before)

        random_index = self.np_random.randint(len(self.mass_scale_set))
        self.mass_scale = self.mass_scale_set[random_index]

        random_index = self.np_random.randint(len(self.damping_scale_set))
        self.damping_scale = self.damping_scale_set[random_index]

        self.change_env()

        return self._get_obs()
    
    def reset_idx_model(self,param_list,idx):
        # reset an env with aspecific index
    
        c = 0.01
        self.set_state(self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
                       self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv,))
        pos_before = mass_center(self.model, self.sim)
        self.prev_pos = np.copy(pos_before)
    
        self.mass_scale = param_list[idx][0]
    
        self.damping_scale = param_list[idx][1]
    
        self.change_env()  
        
        return self._get_obs()
    
    
    def test_reset_model(self,idx):       
        param_list = np.array([[0.85,0.85],[0.85,0.95],[0.85,1.05],[0.85,1.15],
                               [0.95,0.85],[0.95,0.95],[0.95,1.05],[0.95,1.15],
                               [1.05,0.85],[1.05,0.95],[1.05,1.05],[1.05,1.15],
                               [1.15,0.85],[1.15,0.95],[1.15,1.05],[1.15,1.15]])
        
        c = 0.01
        self.set_state(
                self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
                self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv,)
            )
        pos_before = mass_center(self.model, self.sim)
        self.prev_pos = np.copy(pos_before)
    
        self.mass_scale = param_list[idx][0]
    
        self.damping_scale = param_list[idx][1]
    
        self.change_env()
    
        return self._get_obs()        
    

    def reward(self, obs, action, next_obs):
        ctrl = action

        lin_vel_cost = 0.25 / 0.015 * obs[..., 22]
        quad_ctrl_cost = 0.1 * np.sum(np.square(ctrl), axis=-1)
        quad_impact_cost = 0.

        done = (obs[..., 1] < 1.0) | (obs[..., 1] > 2.0)
        alive_bonus = 5.0 * -1 * done

        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus

        return reward
    
    def torch_reward_fn(self):
        def _thunk(obs, act, next_obs):
            ctrl = act

            # lin_vel_cost = 1.25 * obs[..., 0]
            lin_vel_cost = 0.25 / 0.015 * obs[..., 22]
            quad_ctrl_cost = 0.1 * ((torch.square(ctrl)).sum(axis=-1))
            
            quad_impact_cost = 0.

            alive_bonus = 5.0 * torch.tensor(torch.logical_and(torch.gt(obs[..., 1], 1.0),
                               torch.lt(obs[..., 1], 2.0)),dtype=torch.float32)

            reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
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
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] = 2.0
        self.viewer.cam.elevation = -20
    
    def get_sim_parameters(self):
        return np.array([self.mass_scale, self.damping_scale])

    def num_modifiable_parameters(self):
        return 2

    
    

def metahumanoid_cost_torch(obs, act, next_obs):
    ctrl = act

    # lin_vel_cost = 1.25 * obs[..., 0]
    lin_vel_cost = 0.25 / 0.015 * obs[..., 22]
    quad_ctrl_cost = 0.1 * ((torch.square(ctrl)).sum(axis=-1))
    quad_impact_cost = 0.
    alive_bonus = 5.0 * torch.tensor(torch.logical_and(torch.gt(obs[..., 1], 1.0),
                                                       torch.lt(obs[..., 1], 2.0)),dtype=torch.float32)
    reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
            
    return -reward
    
    

def metahumanoid_reward_torch(obs, act, next_obs):
    ctrl = act

    # lin_vel_cost = 1.25 * obs[..., 0]
    lin_vel_cost = 0.25 / 0.015 * obs[..., 22]
    quad_ctrl_cost = 0.1 * ((torch.square(ctrl)).sum(axis=-1))
    quad_impact_cost = 0.
    alive_bonus = 5.0 * torch.tensor(torch.logical_and(torch.gt(obs[..., 1], 1.0),
                                                       torch.lt(obs[..., 1], 2.0)),dtype=torch.float32).cuda()
    reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
            
    return reward



def sample_env():

    humanoid_env=SlimHumanoidEnv(mass_scale_set=[0.80, 0.90, 1.0, 1.10, 1.20], 
                                 damping_scale_set=[0.80, 0.90, 1.0, 1.10, 1.20])
    humanoid_env.reset_model()
    
    return humanoid_env
        

def sample_batch_env(num_envs):

    env_list = [sample_env() for i in range(num_envs)]

    return env_list

    
        
def sample_test_env(idx):

    humanoid_env=SlimHumanoidEnv(mass_scale_set=[0.85, 0.95, 1.05, 1.15], 
                                 damping_scale_set=[0.85, 0.95, 1.05, 1.15])    
    humanoid_env.test_reset_model(idx)

    return humanoid_env


        
def sample_batch_idx_env(num_envs):
    param_list = np.array([[0.8,0.8],[0.8,0.9],[0.8,1.0],[0.8,1.1],[0.8,1.2],
                           [0.9,0.8],[0.9,0.9],[0.9,1.0],[0.9,1.1],[0.9,1.2],
                           [1.0,0.8],[1.0,0.9],[1.0,1.0],[1.0,1.1],[1.0,1.2],
                           [1.1,0.8],[1.1,0.9],[1.1,1.0],[1.1,1.1],[1.1,1.2],
                           [1.2,0.8],[1.2,0.9],[1.2,1.0],[1.2,1.1],[1.2,1.2]])
    
    env_list = []
    idx_list = random.sample(range(len(param_list)), num_envs)
    for idx in idx_list:
        humanoid_env=SlimHumanoidEnv(mass_scale_set=[0.80, 0.90, 1.0, 1.10, 1.20], 
                                     damping_scale_set=[0.80, 0.90, 1.0, 1.10, 1.20])
        humanoid_env.reset_idx_model(param_list, idx)
        env_list.append(humanoid_env)
        
    return env_list







