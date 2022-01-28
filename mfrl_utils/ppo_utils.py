'''
This file is to build replay memory buffers for policy optimization.
'''

import random
import torch
import numpy as np

##########################################################################################################################
    #replay buffer in single task and multi-task policy optimization using PPO
##########################################################################################################################


class ReplayMemory(object):  
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, events):
        for event in zip(*events):
            self.memory.append(event)
            if len(self.memory)>self.capacity:
                del self.memory[0]

    def clear(self):
        self.memory = []

    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: torch.cat(x, 0), samples)



class Multi_Env_ReplayMemory(object):
    
    def __init__(self, capacity, num_envs):
        self.capacity = capacity
        self.num_envs = num_envs
        self.memory = [[] for i in range(num_envs)] 

    def push(self, events, task_idx):
        for event in zip(*events):
            self.memory[task_idx].append(event)
            if len(self.memory[task_idx])>self.capacity:
                del self.memory[task_idx][0]

    def clear(self, task_idx):
        self.memory[task_idx] = []
        
    def multi_env_clear(self):
        for task_idx in range(self.num_envs):
            self.clear(task_idx)

    def sample(self, batch_size, task_idx):
        samples = zip(*random.sample(self.memory[task_idx], batch_size))
        return map(lambda x: torch.cat(x, 0), samples)
    
    def multi_env_sample(self, batch_size):
        multi_env_s_list, multi_env_a_list, multi_env_r_list, multi_env_adv_list, multi_env_logp_list = [], [], [], [], []
        for task_idx in range(self.num_envs):
            batch_states, batch_actions, batch_returns, batch_advantages, batch_logprobs = self.sample(batch_size, task_idx)
            multi_env_s_list.append(batch_states.unsqueeze(0))
            multi_env_a_list.append(batch_actions.unsqueeze(0))
            multi_env_r_list.append(batch_returns.unsqueeze(0))
            multi_env_adv_list.append(batch_advantages.unsqueeze(0))
            multi_env_logp_list.append(batch_logprobs.unsqueeze(0))
        
        multi_env_s_tensor, multi_env_a_tensor, multi_env_r_tensor, multi_env_adv_tensor, multi_env_logp_tensor = \
            torch.cat(multi_env_s_list), torch.cat(multi_env_a_list), torch.cat(multi_env_r_list), \
            torch.cat(multi_env_adv_list), torch.cat(multi_env_logp_list)
        
        return multi_env_s_tensor, multi_env_a_tensor, multi_env_r_tensor, multi_env_adv_tensor, multi_env_logp_tensor
    
    
    

##########################################################################################################################
    #dataset normalization/denormalization and data processing in dynamics models
##########################################################################################################################

    
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


def dm_input_preporcess(obs, env_name, init_state=True):

    if env_name == 'half_cheetah':
        if isinstance(obs, np.ndarray):
            if init_state:
                state = np.concatenate([obs[..., 1:2], np.sin(obs[..., 2:3]), np.cos(obs[..., 2:3]), obs[..., 3:]], axis=-1)
            else:
                state = obs[...,1:]
        else:
            if init_state:
                state = torch.cat((obs[..., 1:2], torch.sin(obs[..., 2:3]), torch.cos(obs[..., 2:3]), obs[..., 3:]), -1)
            else:
                state = obs[...,1:]
  
    elif env_name == 'humanoid':
        if isinstance(obs, np.ndarray):
            state = obs
        else:
            state = obs
    else:
        state = obs
    
    return state  



def dm_output_postprocess(denormalized_state, denormalized_delta, env_name):

    if env_name == 'half_cheetah':
        if isinstance(denormalized_delta, np.ndarray):
            next_obs = np.concatenate([denormalized_delta[..., :1], denormalized_state + denormalized_delta[..., 1:]], axis=-1)
        else:
            next_obs = torch.cat((denormalized_delta[..., :1], denormalized_state + denormalized_delta[..., 1:]), -1)
    elif env_name == 'humanoid':
        next_obs = denormalized_state + denormalized_delta
    else:
        next_obs = denormalized_state + denormalized_delta
    
    return next_obs
    
    
    

def kl_div_prior(z_mu,logvar):

    kld_sum=-0.5*torch.sum(1+logvar-z_mu.pow(2)-logvar.exp())
    if z_mu.dim()==2:
        kld=(torch.mean(kld_sum,dim=0)).sum() 
    elif z_mu.dim()==3:
        kld=(torch.mean(kld_sum,dim=[0,1])).sum() 
    
    return kld
    
