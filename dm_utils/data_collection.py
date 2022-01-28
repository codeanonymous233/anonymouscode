'''
This file is to gather rollout dataset for meta_dm and meta_pn training. 
'''

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable

from dm_utils.controller import RandomPolicy
from envs.normalized_env import normalize

import numpy as np



def rollout(env, policy, num_steps):
    cur_state = env.reset()
    
    states = [cur_state]
    actions = []
    rewards = []
    
    for _ in range(num_steps):
        cur_state = torch.FloatTensor(cur_state).unsqueeze(0).cuda()
        action = torch.flatten(policy(cur_state))  
        action = action.data.cpu().numpy()
        next_state, reward, *_ = env.step(action)
        
        actions.append(action)
        states.append(next_state)
        rewards.append(reward)
        
        cur_state = next_state
        
    states, actions, rewards = tuple(map(lambda l: np.stack(l, axis=0),
                                         (states, actions, rewards)))
    return states, actions, rewards
    


##########################################################################################################################
    #rollout using last updated policy networks
##########################################################################################################################
    

def pn_ppo_rollout(env, env_name, agent, meta_dm, whether_lvm, check_lvm, x_memo_c, y_memo_c, stats,
                   num_steps, policy_type, init_states = None, render = False):

    meta_dm.eval()
    
    for param in meta_dm.parameters():
        param.requires_grad = False
    
    x_memo_c_norm, y_memo_c_norm = normalize_dataset(stats, x_memo_c, var_order=0), normalize_dataset(stats, y_memo_c, var_order=3)
    context_obs = torch.cat((x_memo_c_norm, y_memo_c_norm), dim=-1).squeeze(0)   
    
    if init_states is None:
        cur_states = torch.FloatTensor(env.reset()).cuda()        
    else:
        cur_states = torch.FloatTensor(init_states)
       
    state_list = [cur_states.data.cpu().numpy()]
    action_list = []
    reward_list = [] 

    if whether_lvm == True :
        if check_lvm == 'GS_DM' :
            mu_c,logvar_c,mu_t,logvar_t,y_pred=meta_dm(x_memo_c_norm,y_memo_c_norm,x_memo_c_norm)
            z_sample=meta_dm.reparameterization(torch.mean(mu_c,dim=1),torch.mean(logvar_c,dim=1))
            
    states = cur_states.unsqueeze(0)      
    if render: env.render()
    
    for t in range(num_steps-1):
        
        if whether_lvm:
            if isinstance(states,np.ndarray):
                states=torch.from_numpy(states).unsqueeze(0).cuda()
            preprocessed_states = dm_input_preporcess(states, env_name)
            normalized_states = normalize_dataset(stats, preprocessed_states, var_order=1)
            
            if policy_type == 'S':
                z_sample_unsq_exp=z_sample.detach().squeeze(0).expand(normalized_states.size(0),-1)
                states_lv=torch.cat((normalized_states,z_sample_unsq_exp),dim=-1)             
                actions, action_log_probs = agent.act(states_lv.float())
            elif policy_type == 'H':
                z_sample_unsq_exp = z_sample.detach().squeeze(0).expand(normalized_states.size(0),-1) 
                actions, action_log_probs, mu_p, logvar_p, z_p = agent.act(normalized_states.float(), z_sample_unsq_exp.float())
            elif policy_type == 'P':
                actions, action_log_probs, mu_p, logvar_p, z_p = agent.act(normalized_states.float(), context_obs.float())
                
        else:
            if isinstance(states,np.ndarray):
                states=torch.from_numpy(states).unsqueeze(0).cuda() 
            preprocessed_states = dm_input_preporcess(states, env_name)
            normalized_states = normalize_dataset(stats, preprocessed_states, var_order=1)            
            actions, action_log_probs = agent.act(normalized_states.float())
        
        actions = torch.flatten(actions).detach().cpu().numpy()      
        next_states, rewards, done, _ = env.step(actions) 
        
        if render: env.render()
        
        states = next_states
        
        state_list.append(states)
        action_list.append(actions)
        reward_list.append(rewards)
    
    state_list, action_list, reward_list = np.array(state_list), np.array(action_list), np.array(reward_list)

    for param in meta_dm.parameters():
        param.requires_grad = True    
        
    return state_list, action_list, reward_list



def multi_env_pn_ppo_rollout(env_list, env_name, agent, meta_dm, whether_lvm, check_lvm, 
                             multi_env_x_memory, multi_env_y_memory, stats_list, num_steps, policy_type):
    
    num_envs = len(env_list)
    multi_env_state_list, multi_env_action_list, multi_env_reward_list = [], [], []
    
    for i in range(num_envs):
        state_array, action_array, reward_array = pn_ppo_rollout(env_list[i], env_name, agent, meta_dm, whether_lvm, check_lvm, 
                                                                 multi_env_x_memory[i:i+1,...], multi_env_y_memory[i:i+1,...], 
                                                                 stats_list[i], num_steps, policy_type)
        multi_env_state_list.append(state_array)
        multi_env_action_list.append(action_array)
        multi_env_reward_list.append(reward_array)
        
    multi_env_state_array, multi_env_action_array, multi_env_reward_array = np.array(multi_env_state_list), \
        np.array(multi_env_action_list), np.array(multi_env_reward_list)
        
    return multi_env_state_array, multi_env_action_array, multi_env_reward_array


##########################################################################################################################
    #convert transitions to specific form of dataset for meta dynamics models training
##########################################################################################################################
    

def convert_trajectory_to_training(states, actions, env_name, whether_preprocess = True):

    if states.ndim == 2:
        assert states.shape[0] == actions.shape[0] + 1
        obs, next_obs = states[:-1], states[1:]
        obs_act = np.concatenate((obs, actions), axis=1) 
        obs_diff = next_obs - obs        
    elif states.ndim ==3: 
        assert states.shape[1] == actions.shape[1] + 1
        obs, next_obs = states[:,:-1,:], states[:,1:,:]
        obs_act = np.concatenate((obs, actions), axis=2) 
        obs_diff = next_obs - obs 
                
    if env_name == 'half_cheetah':
        if whether_preprocess:
            x = np.concatenate([obs_act[..., 1:2], np.sin(obs_act[..., 2:3]), np.cos(obs_act[..., 2:3]), obs_act[..., 3:]], axis=-1) 
            obs_x = np.concatenate([obs[..., 1:2], np.sin(obs[..., 2:3]), np.cos(obs[..., 2:3]), obs[..., 3:]], axis=-1) 
            next_obs_x = np.concatenate([next_obs[..., 1:2], np.sin(next_obs[..., 2:3]), np.cos(next_obs[..., 2:3]), next_obs[..., 3:]], axis=-1) 
            y = np.concatenate([next_obs[..., :1], next_obs_x - obs_x], axis=-1) 
        else:
            x = obs_act
            y = obs_diff
       
    elif env_name == 'humanoid':
        x = obs_act 
        y = obs_diff 
    else:
        x = obs_act
        y = obs_diff        
    
    return x, y


def dm_input_preporcess(obs, env_name):

    if env_name == 'half_cheetah':
        if isinstance(obs, np.ndarray):
            state = np.concatenate([obs[..., 1:2], np.sin(obs[..., 2:3]), np.cos(obs[..., 2:3]), obs[..., 3:]], axis=-1) 
        else:
            state = torch.cat((obs[..., 1:2], torch.sin(obs[..., 2:3]), torch.cos(obs[..., 2:3]), obs[..., 3:]), -1) 

    elif env_name == 'humanoid':
        if isinstance(obs, np.ndarray):
            state = obs 
        else:
            state = obs
    else:
        state = obs
    
    return state    
    
    

def dm_output_postprocess(state, delta, env_name):
    
    if env_name == 'half_cheetah':
        if isinstance(delta, np.ndarray):
            next_obs = np.concatenate([delta[..., :1], state + delta[..., 1:]], axis=-1)
        else:
            next_obs = torch.cat((delta[..., :1], state + delta[..., 1:]), -1) 

    elif env_name == 'humanoid':
        if isinstance(delta, np.ndarray):
            next_obs = state + delta
        else:
            next_obs = state + delta
    else:
        next_obs = state + delta
    
    return next_obs
            
            

##########################################################################################################################
    #Data buffer for converted transitions in training dynamics models
##########################################################################################################################


class DynamicsDataBuffer(data.Dataset):
    def __init__(self, capacity=10000):
        self.data = []
        self.capacity = capacity
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x, y = self.data[idx]
        
        return torch.FloatTensor(x), torch.FloatTensor(y)
    
    def __getallitem__(self):
        x_list, y_list = [], []
        allitem = self.data
        for i in range(len(allitem)):
            x_item, y_item = allitem[i]
            x_list.append(x_item)
            y_list.append(y_item)
        x_arr, y_arr = np.array(x_list), np.array(y_list)
        
        return torch.FloatTensor(x_arr), torch.FloatTensor(y_arr)

    def __repr__(self):
        return f'Dynamics Data Buffer with {len(self.data)} / {self.capacity} elements.\n'

    def push(self, x, y):
        if x.ndim == 1:
            assert y.ndim == 1
            x = x[None, :]
            y = y[None, :]
            
        for i in range(x.shape[0]):
            self.data.append((x[i], y[i]))
            
        if len(self.data) > self.capacity:
            del self.data[:len(self.data) - self.capacity]
            
            

def collect_transitions(env, env_name, rand_policy, num_steps, whether_reward = False):
    states, actions, rewards = rollout(env, rand_policy, num_steps)
    x,y=convert_trajectory_to_training(states, actions, env_name) 
    
    if whether_reward:
        return x, y, rewards
    else:
        return x,y
            
            
            
def multi_env_collect_transitions(env_list, env_name, num_steps):
    x_list, y_list = [], []
    
    rand_policy = RandomPolicy(env_list[0])
    
    for i in range(len(env_list)):
        x,y = collect_transitions(env_list[i], env_name, rand_policy, num_steps)
        x_list.append(x)
        y_list.append(y)
        
    x_array, y_array = np.stack(x_list), np.array(y_list)
    x_array, y_array = np.transpose(x_array,(1,0,2)), np.transpose(y_array,(1,0,2)) 
    
    return x_array, y_array


def multi_env_collect_reward(env_list, env_name, num_steps):
    reward_list = []
    
    rand_policy = RandomPolicy(env_list[0])
    
    for i in range(len(env_list)):
        x,y,reward = collect_transitions(env_list[i], env_name, rand_policy, num_steps, whether_reward = True)
        reward_list.append(reward)
        
    reward_array = np.stack(reward_list)
    
    return reward_array.sum()/len(env_list)
 
    
def multi_env_dm_buffer(env_list, env_name, num_steps, num_traj, dim_s):
    dm_buffer = DynamicsDataBuffer(capacity=num_steps*num_traj)
    
    env_list = [normalize(env) for env in env_list]
    
    for _ in range(num_traj):
        x_array, y_array = multi_env_collect_transitions(env_list, env_name, num_steps)
        dm_buffer.push(x_array, y_array) 
        
    x_all, y_all = dm_buffer.__getallitem__() 
    x_all_permute, y_all_permute = x_all.permute(1, 0, 2).contiguous(), y_all.permute(1,0,2).contiguous() 
    stats_list = multi_env_get_stats(x_all_permute, y_all_permute, dim_s)
    
    return dm_buffer, stats_list


def multi_env_reward(env_list, env_name, num_steps, num_traj):

    env_list = [normalize(env) for env in env_list]
    
    reward_array = np.zeros((num_traj))
    
    for i in range(num_traj):
        reward_array[i] = multi_env_collect_reward(env_list, env_name, num_steps)
        
    return reward_array.mean()
        



##########################################################################################################################
    #normalize and denormalize converted transitions, prepared for training dynamics models
##########################################################################################################################
    

def get_stats(x, y, dim_s):

    if x.dim()==3:
        sa_arr=x.detach().cpu().squeeze(0).numpy()
        s_diff=y.detach().cpu().squeeze(0).numpy()
    elif x.dim()==2:
        sa_arr=x.detach().cpu().numpy()
        s_diff=y.detach().cpu().numpy()            
    
    sa_mean,sa_std=np.mean(sa_arr,axis=0),np.std(sa_arr,axis=0)
    s_mean,s_std=sa_mean[:dim_s],sa_std[:dim_s]
    a_mean,a_std=sa_mean[dim_s:],sa_std[dim_s:]
    
    s_diff_mean,s_diff_std=np.mean(s_diff,axis=0),np.std(s_diff,axis=0)
    
    return (s_mean,s_std,a_mean,a_std,s_diff_mean,s_diff_std)
    

def multi_env_get_stats(x, y, dim_s):
    
    num_envs = x.size(0)
    stats_list = []
    for i in range(num_envs):
        stats = get_stats(x[i:i+1,...], y[i:i+1,...], dim_s)
        stats_list.append(stats)
    
    return stats_list
    
    

def normalize_dataset(stats, x, var_order):
    
    (s_mean,s_std,a_mean,a_std,s_diff_mean,s_diff_std)=stats
    
    if var_order==0:
        sa_mean,sa_std=np.concatenate((s_mean,a_mean)),np.concatenate((s_std,a_std))
        sa_mean_tensor,sa_std_tensor=torch.FloatTensor(sa_mean).cuda(),torch.FloatTensor(sa_std).cuda()
        mean_exp,std_exp=sa_mean_tensor.expand_as(x),sa_std_tensor.expand_as(x)
        
    elif var_order==1:
        s_mean_tensor,s_std_tensor=torch.FloatTensor(s_mean).cuda(),torch.FloatTensor(s_std).cuda()
        mean_exp,std_exp=s_mean_tensor.expand_as(x),s_std_tensor.expand_as(x)
        
    elif var_order==2:
        a_mean_tensor,a_std_tensor=torch.FloatTensor(a_mean).cuda(),torch.FloatTensor(a_std).cuda()
        mean_exp,std_exp=a_mean_tensor.expand_as(x),a_std_tensor.expand_as(x)
        
    elif var_order==3:
        s_diff_mean_tensor,s_diff_std_tensor=torch.FloatTensor(s_diff_mean).cuda(),torch.FloatTensor(s_diff_std).cuda()
        mean_exp,std_exp=s_diff_mean_tensor.expand_as(x),s_diff_std_tensor.expand_as(x)        
    
    normalized_x=(x-mean_exp)/(std_exp+1e-10)
    
    return normalized_x



def multi_env_normalize_dataset(stats_list, x, var_order):
    
    num_envs = len(stats_list)
    normalized_x = torch.zeros_like(x).cuda()
    for i in range(num_envs):
        normalized_x[i:i+1,...] = normalize_dataset(stats_list[i], x[i:i+1,...], var_order) 
    
    return normalized_x 

                 
                                                                    
def denormalize_dataset(stats, normalized_x, var_order):
    
    (s_mean,s_std,a_mean,a_std,s_diff_mean,s_diff_std)=stats
    
    if var_order==0:
        sa_mean,sa_std=np.concatenate((s_mean,a_mean)),np.concatenate((s_std,a_std))
        sa_mean_tensor,sa_std_tensor=torch.FloatTensor(sa_mean).cuda(),torch.FloatTensor(sa_std).cuda()
        mean_exp,std_exp=sa_mean_tensor.expand_as(normalized_x),sa_std_tensor.expand_as(normalized_x)
        
    elif var_order==1:
        s_mean_tensor,s_std_tensor=torch.FloatTensor(s_mean).cuda(),torch.FloatTensor(s_std).cuda()
        mean_exp,std_exp=s_mean_tensor.expand_as(normalized_x),s_std_tensor.expand_as(normalized_x)
        
    elif var_order==2:
        a_mean_tensor,a_std_tensor=torch.FloatTensor(a_mean).cuda(),torch.FloatTensor(a_std).cuda()
        mean_exp,std_exp=a_mean_tensor.expand_as(normalized_x),a_std_tensor.expand_as(normalized_x) 
        
    elif var_order==3:
        s_diff_mean_tensor,s_diff_std_tensor=torch.FloatTensor(s_diff_mean).cuda(),torch.FloatTensor(s_diff_std).cuda()
        mean_exp,std_exp=s_diff_mean_tensor.expand_as(normalized_x),s_diff_std_tensor.expand_as(normalized_x)        
    
    x=normalized_x*(std_exp+1e-10)+mean_exp
    
    return x



def multi_env_denormalize_dataset(stats_list, x, var_order):
    
    num_envs = len(stats_list)
    denormalized_x = torch.zeros_like(x).cuda()
    for i in range(num_envs):
        denormalized_x[i:i+1,...] = denormalize_dataset(stats_list[i], x[i:i+1,...], var_order)
    
    return denormalized_x
            
            
            


    
    
