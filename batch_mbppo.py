'''
This file is to compile modules in combining meta dm training and meta model-based policy search.
'''

import torch
import torch.nn as nn
import numpy as np
import gym


from dm_utils.data_collection import *
from dm_utils.meta_dms import *
from dm_utils.dm_training import *
from dm_utils.meta_loss import *

from param_list import *
from utils import *

from mfrl_utils.mb_ppo import multi_env_train_ppo_in_dm, h_multi_env_train_ppo_in_dm, p_multi_env_train_ppo_in_dm, multi_env_eval_ppo_in_env
from mfrl_utils.ppo_model import Model, Hierarchical_Model, Parallel_Model

from dm_utils.data_collection import multi_env_collect_transitions
from dm_utils.dm_training import multi_env_train_meta_dm



torch.manual_seed(2020)

dm_type='GS_DM'
use_lvm=True
policy_type='S'
env_name='half_cheetah'
torch.set_default_dtype(torch.float32)


if env_name=='half_cheetah':
    from envs.Meta_Halfcheetah import sample_batch_idx_env, metacheetah_cost_torch, metacheetah_reward_torch
    # dim_obs=18, dim_action=6, obs=[-inf,inf], action=[-1,+1]
    dim_obs=18
    dim_s=18
    dim_action=6
    act_scale=1.0
    cost_function=metacheetah_cost_torch 
    reward_function=metacheetah_reward_torch

elif env_name=='humanoid':
    from envs.Meta_Humanoid import sample_batch_idx_env, metahumanoid_cost_torch, metahumanoid_reward_torch
    # dim_obs=45, dim_action=17, obs=[-inf,inf], action=[-0.4,+0.4]
    dim_obs=45
    dim_s=45
    dim_action=17
    act_scale=0.4
    cost_function=metahumanoid_cost_torch
    reward_function=metahumanoid_reward_torch

    
    
args_gsdm=GSDM_Param()
args_npdm=NPDM_Param()
args_attnnpdm=AttnNPDM_Param()


if use_lvm:
    if policy_type == 'S':
        agent = Model(num_inputs=dim_s+args_gsdm.dim_lat, num_outputs=dim_action).cuda()       
    elif policy_type == 'H':
        agent = Hierarchical_Model(dim_input=dim_s, dim_output=dim_action, 
                                   dim_lat=args_gsdm.dim_lat, dim_h=args_gsdm.dim_h, num_h=2).cuda()
    elif policy_type == 'P':
        agent = Parallel_Model(dim_input=dim_s, dim_output=dim_action, 
                               dim_obs_x=dim_s+dim_action, dim_obs_y=dim_obs, 
                               dim_lat=args_gsdm.dim_lat, dim_h=args_gsdm.dim_lat, num_h=2).cuda() 
    pn_optim = optim.Adam(agent.parameters(), lr=7e-4)
    pn_optim_scheduler = torch.optim.lr_scheduler.StepLR(pn_optim, step_size=1000, gamma=1.0) 
    
else:
    agent = Model(num_inputs=dim_s, num_outputs=dim_action).cuda()
    pn_optim = optim.Adam(agent.parameters(), lr=7e-4)
    pn_optim_scheduler = torch.optim.lr_scheduler.StepLR(pn_optim, step_size=1000, gamma=1.0) 



if dm_type=='GS_DM':
    meta_dm = GS_DM(args_gsdm).float()
elif dm_type=='NP_DM':
    meta_dm = NP_DM(args_npdm).float()
elif dm_type=='AttnNP_DM':
    meta_dm = AttnNP_DM(args_attnnpdm).float()
    
dm_optim = torch.optim.Adam(meta_dm.parameters(), lr=1e-3)
dm_optim_scheduler = torch.optim.lr_scheduler.StepLR(dm_optim, step_size=100, gamma=0.9)



def main_mbmrl(meta_dm, dm_optim, agent, pn_optim, whether_lvm, check_lvm, env_name, policy_type, num_envs=3, num_traj=1, num_iter=20,
               num_steps=1000, dm_b_size=1000, task_iter=2, writer=1):
    
    dm_tr_loss_list, dm_tr_mse_list, dm_te_mse_list, reward_te_list = np.zeros((num_iter,task_iter)), \
        np.zeros((num_iter,task_iter)), np.zeros((num_iter,task_iter)), np.zeros((num_iter,task_iter))
    
    loss_fun = get_dm_loss(check_lvm)
    
    for i in range(num_iter):
        env_list = sample_batch_idx_env(num_envs)
        
        dm_buffer, stats_list = multi_env_dm_buffer(env_list,env_name,num_steps,num_traj,dim_s)
        
        x_array, y_array = multi_env_collect_transitions(env_list, env_name, num_steps=500)
        multi_env_x_memory, multi_env_y_memory = torch.FloatTensor(x_array).permute(1, 0, 2).contiguous().cuda(), \
            torch.FloatTensor(y_array).permute(1, 0, 2).contiguous().cuda()
        
        for j in range(task_iter):
            
            multi_env_trainloader = torch.utils.data.DataLoader(dm_buffer, batch_size=dm_b_size, shuffle=True)
            
            b_dm_train_loss, b_dm_train_mse  = multi_env_train_meta_dm(meta_dm, stats_list, dm_optim, multi_env_trainloader, 
                                                                       check_lvm, loss_fun, epoch=20, whether_norm=True)
            
            avg_b_dm_train_loss, avg_b_dm_train_mse = np.mean(b_dm_train_loss), np.mean(b_dm_train_mse)
            dm_tr_loss_list[i,j], dm_tr_mse_list[i,j] = avg_b_dm_train_loss, avg_b_dm_train_mse
            
            if policy_type == 'S': 
                agent = multi_env_train_ppo_in_dm(meta_dm, env_list, env_name, agent, pn_optim, whether_lvm, check_lvm, 
                                                  multi_env_x_memory, multi_env_y_memory, stats_list, 
                                                  meta_rewards_torch=reward_function, num_episode=20, num_steps=1000, eval_traj_in_env = False)
            elif policy_type == 'H': 
                agent = h_multi_env_train_ppo_in_dm(meta_dm, env_list, env_name, agent, pn_optim, whether_lvm, check_lvm, 
                                                    multi_env_x_memory, multi_env_y_memory, stats_list, 
                                                    meta_rewards_torch=reward_function, num_episode=20, num_steps=1000, eval_traj_in_env = False)
            elif policy_type == 'P': 
                agent = p_multi_env_train_ppo_in_dm(meta_dm, env_list, env_name, agent, pn_optim, whether_lvm, check_lvm, 
                                                    multi_env_x_memory, multi_env_y_memory, stats_list, 
                                                    meta_rewards_torch=reward_function, num_episode=20, num_steps=1000, eval_traj_in_env = False)                
            
            multi_env_rewards_arr = multi_env_eval_ppo_in_env(env_list, env_name, agent, meta_dm, whether_lvm, check_lvm, 
                                                              multi_env_x_memory, multi_env_y_memory, stats_list, num_steps, 
                                                              policy_type, num_iter = 15) 
            
            avg_rewards=np.mean(multi_env_rewards_arr)
            reward_te_list[i,j] = avg_rewards          
            
            state_array, action_array, reward_array = multi_env_pn_ppo_rollout(env_list, env_name, agent, meta_dm, whether_lvm, check_lvm, 
                                                                               multi_env_x_memory, multi_env_y_memory, stats_list, 
                                                                               num_steps, policy_type)
            
            x, y = convert_trajectory_to_training(state_array, action_array, env_name)
            x_transpose, y_transpose = np.transpose(x,(1,0,2)), np.transpose(y,(1,0,2)) 
            dm_buffer.push(x_transpose, y_transpose)
            
            dm_optim_scheduler.step()
            pn_optim_scheduler.step() 
          
    torch.save(meta_dm.state_dict(), './runs/'+check_lvm+'/'+str(writer)+'/dm_net.pt')
    torch.save(agent.state_dict(), './runs/'+check_lvm+'/'+str(writer)+'/ppo_net.pt')
    np.save('./runs/'+check_lvm+'/'+str(writer)+'/reward_te_list.npy', reward_te_list)
    
    
main_mbmrl(meta_dm, dm_optim, agent, pn_optim, whether_lvm=use_lvm, check_lvm=dm_type, env_name=env_name, 
           policy_type=policy_type, num_envs=1, num_traj=4, num_iter=30, num_steps=1000, dm_b_size=1000, 
           task_iter=4, writer=5)

            
            
        
        
























































    



        
    
