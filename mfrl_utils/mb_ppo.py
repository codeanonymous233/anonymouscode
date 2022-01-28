'''
This file is to enable training of contextual based mode-based meta RL using PPO modules.
'''

import random
import numpy as np
import math

import torch
import torch.nn as nn
from torch.autograd import Variable

from mfrl_utils.ppo_utils import kl_div_prior
from mfrl_utils.ppo_utils import ReplayMemory, Multi_Env_ReplayMemory
from mfrl_utils.ppo_utils import dm_input_preporcess, dm_output_postprocess
from mfrl_utils.ppo_utils import normalize_dataset, denormalize_dataset, multi_env_normalize_dataset, multi_env_denormalize_dataset


def train_ppo_in_dm(meta_dm, env, env_name, agent, optimizer, whether_lvm, check_lvm, 
                    x_memo_c, y_memo_c, stats, meta_rewards_torch, clip = 0.2, 
                    num_episode = 100, num_epoch = 10, num_steps = 1000, gamma = 0.99, 
                    gae_param = 0.95, ent_coeff = 0.01, memo_bsize = 64, 
                    max_grad_norm = 0.5, render = False, collect_memory = True, eval_traj_in_env = False):

    meta_dm.eval()

    for param in meta_dm.parameters():
        param.requires_grad = False 
        
    if collect_memory:
        x_memo_c_norm, y_memo_c_norm = normalize_dataset(stats, x_memo_c, var_order=0), normalize_dataset(stats, y_memo_c, var_order=3)

    if whether_lvm == True :
        if check_lvm == 'GS_DM' :
            mu_t,logvar_t,mu_c,logvar_c,y_pred=meta_dm(x_memo_c_norm,y_memo_c_norm,x_memo_c_norm)
            z_sample=meta_dm.reparameterization(torch.mean(mu_c,dim=1),torch.mean(logvar_c,dim=1)) 
    
    memory = ReplayMemory(num_steps)
    state = env.reset()
    state = Variable(torch.Tensor(state).unsqueeze(0))
    done = True
    episode = -1

    for t in range(num_episode):
        
        episode_length = 0

        while(len(memory.memory)<num_steps):
            states = [] 
            actions = [] 
            rewards = []
            values = []
            returns = []
            advantages = []
            logprobs = []
            av_reward = 0
            cum_reward = 0
            cum_done = 0

            for step in range(num_steps):
                episode_length += 1
                
                if step==0:    
                    preprocessed_state = (dm_input_preporcess(state, env_name)).cuda()
                else:
                    preprocessed_state = (dm_input_preporcess(state, env_name, init_state=False)).cuda()
                state = normalize_dataset(stats, preprocessed_state, var_order=1) 
                #########################################################
                if whether_lvm:
                    z_sample_unsq_exp=z_sample.detach().unsqueeze(0).expand(state.size(0),-1)
                    state_lv=torch.cat((state,z_sample_unsq_exp),dim=-1) 
                    
                    v, action, log_prob = agent.act(state_lv.float(),return_v=True)
                    states.append(state_lv)
                else:
                    v, action, log_prob = agent.act(state,return_v=True)
                    states.append(state)
                    
                actions.append(action)
                logprobs.append(log_prob)
                values.append(v)
                
                normalized_action = normalize_dataset(stats, action, var_order=2) 
                state_action_tensor = (torch.cat((state, normalized_action), dim=-1)).unsqueeze(0) 
            
                if check_lvm == 'NP_DM' :
                    mu_c, logvar_c, mu_t, logvar_t, delta_usq = meta_dm(x_memo_c_norm,y_memo_c_norm,x_memo_c_norm,y_memo_c_norm,state_action_tensor)
                elif check_lvm == 'AttnNP_DM' :
                    mu_c, logvar_c, mu_t, logvar_t, delta_usq=meta_dm(x_memo_c_norm,y_memo_c_norm,x_memo_c_norm,y_memo_c_norm,state_action_tensor)
                elif check_lvm == 'GS_DM' :
                    mu_c, logvar_c, mu_t, logvar_t, delta_usq=meta_dm(x_memo_c_norm,y_memo_c_norm,state_action_tensor)
                else:
                    delta_usq = meta_dm(state_action_tensor)  
                
                denormalized_delta = denormalize_dataset(stats, delta_usq.squeeze(0), var_order=3)
                denormalized_state = denormalize_dataset(stats, state, var_order=1) 
                
                denormalized_next_state = dm_output_postprocess(denormalized_state, denormalized_delta, env_name) 

                reward = (meta_rewards_torch(denormalized_state,action,denormalized_next_state)).item()

                done = (episode_length >= num_steps)
                cum_reward += reward
                reward = max(min(reward, 1), -1)
                rewards.append(reward)
                
                state = denormalized_next_state
                
                if done:
                    episode += 1
                    cum_done += 1
                    av_reward += cum_reward
                    cum_reward = 0
                    episode_length = 0
                    state = env.reset()
                    state = Variable(torch.Tensor(state).unsqueeze(0)).cuda()
                
                if done:
                    break

            R = torch.zeros(1, 1)
            if not done:
                last_state = dm_input_preporcess(state, env_name)
                last_state = normalize_dataset(stats, last_state, var_order=1)                
                if whether_lvm:
                    z_sample_unsq_exp=z_sample.detach().unsqueeze(0).expand(last_state.size(0),-1)
                    state_lv=torch.cat((state,z_sample_unsq_exp),dim=-1) 
                    v, action, log_prob = agent.act(state_lv.float(),return_v=True)
                else:
                    v, action, log_prob = agent.act(last_state,return_v=True)
                
                R = v.data

            R = Variable(R)
            values.append(R)
            A = Variable(torch.zeros(1, 1)).cuda()
            for i in reversed(range(len(rewards))):
                td = rewards[i] + gamma*values[i+1].data[0,0] - values[i].data[0,0]
                A = float(td) + gamma*gae_param*A
                advantages.insert(0, A)
                R = A + values[i]
                returns.insert(0, R)

            memory.push([states, actions, returns, advantages, logprobs])

        for k in range(num_epoch):
            batch_states, batch_actions, batch_returns, batch_advantages, batch_logprobs = memory.sample(memo_bsize)
            batch_actions = Variable(batch_actions.data, requires_grad=False)
            batch_states = Variable(batch_states.data, requires_grad=False)
            batch_returns = Variable(batch_returns.data, requires_grad=False)
            batch_advantages = Variable(batch_advantages.data, requires_grad=False)
            batch_logprobs = Variable(batch_logprobs.data, requires_grad=False)

            mu, sigma_sq, v_pred = agent(batch_states)
            log_std = agent.log_std
            log_probs = -0.5 * ((batch_actions - mu) / (sigma_sq+1e-8)).pow(2) - 0.5 * math.log(2 * math.pi) - log_std
            log_probs = log_probs.sum(-1, keepdim=True)
            dist_entropy = 0.5 + 0.5 * math.log(2 * math.pi) + log_std
            dist_entropy = dist_entropy.sum(-1).mean()

            ratio = torch.exp(log_probs - batch_logprobs)

            surr1 = ratio * batch_advantages.expand_as(ratio) 
            surr2 = ratio.clamp(1-clip, 1+clip) * batch_advantages.expand_as(ratio)
            loss_clip = - torch.mean(torch.min(surr1, surr2))

            loss_value = (v_pred - batch_returns).pow(2).mean()

            loss_ent = - ent_coeff * dist_entropy

            total_loss = (loss_clip + loss_value + loss_ent)
            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm(agent.parameters(), max_grad_norm)
            optimizer.step()

        memory.clear()
        
        if eval_traj_in_env:
            rewards_arr = eval_ppo_in_env(env=env, env_name=env_name, agent=agent, meta_dm=meta_dm, whether_lvm=whether_lvm, 
                                          check_lvm=check_lvm, x_memo_c=x_memo_c, y_memo_c=y_memo_c, stats=stats, 
                                          num_iter=1)
            avg_rewards=np.mean(rewards_arr)

    return agent



def relabel_multi_env_memory_update(meta_dm, env_list, env_name, stats_list, meta_rewards_torch, multi_env_memory, 
                                    agent, num_steps, gamma, gae_param, whether_lvm, check_lvm, norm_multi_env_x_memory, 
                                    norm_multi_env_y_memory, multi_env_z_sample):
    num_envs = len(env_list)
    
    for task_idx in range(num_envs):
    
        state = env_list[task_idx].reset()
        state = Variable(torch.Tensor(state).unsqueeze(0))
        done = True
        
        episode_length = 0 

        while(len(multi_env_memory.memory[task_idx])<num_steps):
            states = []
            actions = []
            rewards = []
            values = []
            returns = []
            advantages = []
            logprobs = []
            av_reward = 0
            cum_reward = 0
            cum_done = 0
    
            for step in range(num_steps):
                episode_length += 1
                
                if step==0:    
                    preprocessed_state = (dm_input_preporcess(state, env_name)).cuda()
                else:
                    preprocessed_state = (dm_input_preporcess(state, env_name, init_state=False)).cuda()
                state = normalize_dataset(stats_list[task_idx], preprocessed_state, var_order=1) 
                
                #########################################################
                if whether_lvm:
                    z_sample_unsq_exp=multi_env_z_sample[task_idx,...].detach().unsqueeze(0).expand(state.size(0),-1)
                    state_lv=torch.cat((state,z_sample_unsq_exp),dim=-1) 
                    
                    v, action, log_prob = agent.act(state_lv.float(),return_v=True)
                    states.append(state_lv)
                else:
                    v, action, log_prob = agent.act(state,return_v=True)
                    states.append(state)
                    
                actions.append(action)
                logprobs.append(log_prob)
                values.append(v)
                
                normalized_action = normalize_dataset(stats_list[task_idx], action, var_order=2) 
                state_action_tensor = (torch.cat((state, normalized_action), dim=-1)).unsqueeze(0)
            
                if check_lvm == 'NP_DM' :
                    mu_c, logvar_c, mu_t, logvar_t, delta_usq = meta_dm(norm_multi_env_x_memory[task_idx:task_idx+1,...],norm_multi_env_y_memory[task_idx:task_idx+1,...],
                                                                        norm_multi_env_x_memory[task_idx:task_idx+1,...],norm_multi_env_y_memory[task_idx:task_idx+1,...],
                                                                        state_action_tensor)
                elif check_lvm == 'AttnNP_DM' :
                    mu_c, logvar_c, mu_t, logvar_t, delta_usq=meta_dm(norm_multi_env_x_memory[task_idx:task_idx+1,...],norm_multi_env_y_memory[task_idx:task_idx+1,...],
                                                                      norm_multi_env_x_memory[task_idx:task_idx+1,...],norm_multi_env_y_memory[task_idx:task_idx+1,...],
                                                                      state_action_tensor)
                elif check_lvm == 'GS_DM' :
                    mu_c, logvar_c, mu_t, logvar_t, delta_usq=meta_dm(norm_multi_env_x_memory[task_idx:task_idx+1,...],norm_multi_env_y_memory[task_idx:task_idx+1,...],
                                                                      state_action_tensor)
                else:
                    delta_usq = meta_dm(state_action_tensor)  
                
                denormalized_delta = denormalize_dataset(stats_list[task_idx], delta_usq.squeeze(0), var_order=3)
                denormalized_state = denormalize_dataset(stats_list[task_idx], state, var_order=1) 
                
                denormalized_next_state = dm_output_postprocess(denormalized_state, denormalized_delta, env_name) 

                reward = (meta_rewards_torch(denormalized_state,action,denormalized_next_state)).item()
   
                done = (episode_length >= num_steps)
                cum_reward += reward
                reward = max(min(reward, 1), -1)
                rewards.append(reward)
                
                state = denormalized_next_state
                
                if done:
                    cum_done += 1
                    av_reward += cum_reward
                    cum_reward = 0
                    episode_length = 0
                    state = env_list[task_idx].reset()
                    state = Variable(torch.Tensor(state).unsqueeze(0)).cuda()
                
                if done:
                    break

            R = torch.zeros(1, 1)
            if not done:
                last_state = dm_input_preporcess(state, env_name)
                last_state = normalize_dataset(stats_list[task_idx], last_state, var_order=1)              
                if whether_lvm:
                    z_sample_unsq_exp=multi_env_z_sample[task_idx,...].detach().squeeze(0).expand(last_state.size(0),-1)
                    state_lv=torch.cat((state,z_sample_unsq_exp),dim=-1) 
                    v, action, log_prob = agent.act(state_lv.float(),return_v=True)
                else:
                    v, action, log_prob = agent.act(last_state,return_v=True)
                
                R = v.data

            R = Variable(R)
            values.append(R)
            A = Variable(torch.zeros(1, 1)).cuda()
            for i in reversed(range(len(rewards))):
                td = rewards[i] + gamma*values[i+1].data[0,0] - values[i].data[0,0]
                A = float(td) + gamma*gae_param*A
                advantages.insert(0, A)
                R = A + values[i]
                returns.insert(0, R)

            multi_env_memory.push([states, actions, returns, advantages, logprobs],task_idx)
        
    return multi_env_memory



def multi_env_train_ppo_in_dm(meta_dm, env_list, env_name, agent, pn_optim, whether_lvm, check_lvm, 
                              multi_env_x_memory, multi_env_y_memory, stats_list, meta_rewards_torch, 
                              clip = 0.2, num_episode = 100, num_epoch = 10, 
                              num_steps = 1000, gamma = 0.99, gae_param = 0.95, ent_coeff = 0.01, memo_bsize = 64, 
                              max_grad_norm = 0.5, collect_memory = True, eval_traj_in_env = True):
    
    meta_dm.eval()

    for param in meta_dm.parameters():
        param.requires_grad = False 
        
    if collect_memory:
        norm_multi_env_x_memory, norm_multi_env_y_memory = multi_env_normalize_dataset(stats_list, multi_env_x_memory, var_order=0), \
            multi_env_normalize_dataset(stats_list, multi_env_y_memory, var_order=3)

    if whether_lvm == True :
        if check_lvm == 'GS_DM' :
            mu_t,logvar_t,mu_c,logvar_c,y_pred=meta_dm(norm_multi_env_x_memory,norm_multi_env_y_memory,norm_multi_env_x_memory)
            multi_env_z_sample=meta_dm.reparameterization(torch.mean(mu_c,dim=1),torch.mean(logvar_c,dim=1))
    else:
        multi_env_z_sample = None
    
    
    num_envs = len(env_list)
    
    multi_env_memory = Multi_Env_ReplayMemory(num_steps, num_envs)
    
    for t in range(num_episode):

        multi_env_memory = relabel_multi_env_memory_update(meta_dm, env_list, env_name, stats_list, meta_rewards_torch, multi_env_memory, 
                                                           agent, num_steps, gamma, gae_param, whether_lvm, check_lvm, norm_multi_env_x_memory, 
                                                           norm_multi_env_y_memory, multi_env_z_sample)        
        
        for k in range(num_epoch):
            batch_states, batch_actions, batch_returns, batch_advantages, batch_logprobs = multi_env_memory.multi_env_sample(memo_bsize)
            batch_actions = Variable(batch_actions.data, requires_grad=False)
            batch_states = Variable(batch_states.data, requires_grad=False)
            batch_returns = Variable(batch_returns.data, requires_grad=False)
            batch_advantages = Variable(batch_advantages.data, requires_grad=False)
            batch_logprobs = Variable(batch_logprobs.data, requires_grad=False)

            mu, sigma_sq, v_pred = agent(batch_states)
            log_std = agent.log_std
            log_probs = -0.5 * ((batch_actions - mu) / sigma_sq).pow(2) - 0.5 * math.log(2 * math.pi) - log_std
            log_probs = log_probs.sum(-1, keepdim=True)
            dist_entropy = 0.5 + 0.5 * math.log(2 * math.pi) + log_std
            dist_entropy = dist_entropy.sum(-1).mean()

            ratio = torch.exp(log_probs - batch_logprobs)

            surr1 = ratio * batch_advantages.expand_as(ratio) 
            surr2 = ratio.clamp(1-clip, 1+clip) * batch_advantages.expand_as(ratio)
            loss_clip = - torch.mean(torch.min(surr1, surr2))

            loss_value = (v_pred - batch_returns).pow(2).mean()

            loss_ent = - ent_coeff * dist_entropy

            total_loss = (loss_clip + loss_value + loss_ent)
            pn_optim.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm(agent.parameters(), max_grad_norm)
            pn_optim.step() 

        multi_env_memory.multi_env_clear()
        
        if eval_traj_in_env:
            multi_env_rewards_arr = multi_env_eval_ppo_in_env(env_list, env_name, agent, meta_dm, whether_lvm, check_lvm, 
                                                              multi_env_x_memory, multi_env_y_memory, stats_list, num_steps, 
                                                              policy_type = 'S', num_iter = 5)
            avg_rewards=np.mean(multi_env_rewards_arr)

    return agent




def h_relabel_multi_env_memory_update(meta_dm, env_list, env_name, stats_list, meta_rewards_torch, multi_env_memory, 
                                      agent, num_steps, gamma, gae_param, whether_lvm, check_lvm, norm_multi_env_x_memory, 
                                      norm_multi_env_y_memory, multi_env_z_sample):
    num_envs = len(env_list)
    
    for task_idx in range(num_envs):
    
        state = env_list[task_idx].reset()
        state = Variable(torch.Tensor(state).unsqueeze(0))
        done = True
        
        episode_length = 0
        
        while(len(multi_env_memory.memory[task_idx])<num_steps):
            states = []
            actions = []
            rewards = []
            values = []
            returns = []
            advantages = []
            logprobs = []
            av_reward = 0
            cum_reward = 0
            cum_done = 0
    
            for step in range(num_steps):
                episode_length += 1
                
                if step==0:    
                    preprocessed_state = (dm_input_preporcess(state, env_name)).cuda()
                else:
                    preprocessed_state = (dm_input_preporcess(state, env_name, init_state=False)).cuda()
                state = normalize_dataset(stats_list[task_idx], preprocessed_state, var_order=1)
                
                #########################################################
                if whether_lvm:
                    z_sample_unsq_exp=multi_env_z_sample[task_idx,...].detach().unsqueeze(0).expand(state.size(0),-1)

                    v, action, log_prob, mu_p, logvar_p, z_p = agent.act(state.float(),z_sample_unsq_exp.float(),return_v=True)
                    states.append(state)
                else:
                    v, action, log_prob = agent.act(state,return_v=True)
                    states.append(state)
                    
                actions.append(action)
                logprobs.append(log_prob)
                values.append(v)
                
                normalized_action = normalize_dataset(stats_list[task_idx], action, var_order=2) 
                state_action_tensor = (torch.cat((state, normalized_action), dim=-1)).unsqueeze(0) 
            
                if check_lvm == 'NP_DM' :
                    mu_c, logvar_c, mu_t, logvar_t, delta_usq = meta_dm(norm_multi_env_x_memory[task_idx:task_idx+1,...],norm_multi_env_y_memory[task_idx:task_idx+1,...],
                                                                        norm_multi_env_x_memory[task_idx:task_idx+1,...],norm_multi_env_y_memory[task_idx:task_idx+1,...],
                                                                        state_action_tensor)
                elif check_lvm == 'AttnNP_DM' :
                    mu_c, logvar_c, mu_t, logvar_t, delta_usq=meta_dm(norm_multi_env_x_memory[task_idx:task_idx+1,...],norm_multi_env_y_memory[task_idx:task_idx+1,...],
                                                                      norm_multi_env_x_memory[task_idx:task_idx+1,...],norm_multi_env_y_memory[task_idx:task_idx+1,...],
                                                                      state_action_tensor)
                elif check_lvm == 'GS_DM' :
                    mu_t, logvar_t, mu_c, logvar_c, delta_usq=meta_dm(norm_multi_env_x_memory[task_idx:task_idx+1,...],norm_multi_env_y_memory[task_idx:task_idx+1,...],
                                                                      state_action_tensor)
                else:
                    delta_usq = meta_dm(state_action_tensor)  
                
                denormalized_delta = denormalize_dataset(stats_list[task_idx], delta_usq.squeeze(0), var_order=3)
                denormalized_state = denormalize_dataset(stats_list[task_idx], state, var_order=1) 
                
                denormalized_next_state = dm_output_postprocess(denormalized_state, denormalized_delta, env_name) 

                reward = (meta_rewards_torch(denormalized_state,action,denormalized_next_state)).item()
   
                done = (episode_length >= num_steps)
                cum_reward += reward
                reward = max(min(reward, 1), -1)
                rewards.append(reward)
                
                state = denormalized_next_state
                
                if done:
                    cum_done += 1
                    av_reward += cum_reward
                    cum_reward = 0
                    episode_length = 0
                    state = env_list[task_idx].reset()
                    state = Variable(torch.Tensor(state).unsqueeze(0)).cuda()
                
                if done:
                    break

            R = torch.zeros(1, 1)
            if not done:
                last_state = dm_input_preporcess(state, env_name)
                last_state = normalize_dataset(stats_list[task_idx], last_state, var_order=1)            
                if whether_lvm:
                    z_sample_unsq_exp=multi_env_z_sample[task_idx,...].detach().unsqueeze(0).expand(last_state.size(0),-1)
                    
                    v, action, log_prob, mu_p, logvar_p, z_p = agent.act(state.float(),z_sample_unsq_exp.float(),return_v=True)
                else:
                    v, action, log_prob = agent.act(last_state,return_v=True)
                
                R = v.data

            R = Variable(R)
            values.append(R)
            A = Variable(torch.zeros(1, 1)).cuda()
            for i in reversed(range(len(rewards))):
                td = rewards[i] + gamma*values[i+1].data[0,0] - values[i].data[0,0]
                A = float(td) + gamma*gae_param*A
                advantages.insert(0, A)
                R = A + values[i]
                returns.insert(0, R)

            multi_env_memory.push([states, actions, returns, advantages, logprobs],task_idx)
        
    return multi_env_memory


def h_multi_env_train_ppo_in_dm(meta_dm, env_list, env_name, agent, pn_optim, whether_lvm, check_lvm, 
                                multi_env_x_memory, multi_env_y_memory, stats_list, meta_rewards_torch, 
                                clip = 0.2, num_episode = 100, num_epoch = 10, 
                                num_steps = 1000, gamma = 0.99, gae_param = 0.95, ent_coeff = 0.01, memo_bsize = 64, 
                                max_grad_norm = 0.5, collect_memory = True, eval_traj_in_env = True):
    
    meta_dm.eval()

    for param in meta_dm.parameters():
        param.requires_grad = False 
        
    if collect_memory:
        norm_multi_env_x_memory, norm_multi_env_y_memory = multi_env_normalize_dataset(stats_list, multi_env_x_memory, var_order=0), \
            multi_env_normalize_dataset(stats_list, multi_env_y_memory, var_order=3)

    if whether_lvm == True :
        if check_lvm == 'GS_DM' :
            mu_t,logvar_t,mu_c,logvar_c,y_pred=meta_dm(norm_multi_env_x_memory,norm_multi_env_y_memory,norm_multi_env_x_memory)
            multi_env_z_sample=meta_dm.reparameterization(torch.mean(mu_c,dim=1),torch.mean(logvar_c,dim=1))
            
    else:
        multi_env_z_sample = None
    
    num_envs = len(env_list)
    
    multi_env_memory = Multi_Env_ReplayMemory(num_steps, num_envs)
    
    for t in range(num_episode):

        multi_env_memory = h_relabel_multi_env_memory_update(meta_dm, env_list, env_name, stats_list, 
                                                             meta_rewards_torch, multi_env_memory, 
                                                             agent, num_steps, gamma, gae_param, 
                                                             whether_lvm, check_lvm, norm_multi_env_x_memory, 
                                                             norm_multi_env_y_memory, multi_env_z_sample)        
        
        for k in range(num_epoch):
            batch_states, batch_actions, batch_returns, batch_advantages, batch_logprobs = multi_env_memory.multi_env_sample(memo_bsize)
            batch_actions = Variable(batch_actions.data, requires_grad=False)
            batch_states = Variable(batch_states.data, requires_grad=False)
            batch_returns = Variable(batch_returns.data, requires_grad=False)
            batch_advantages = Variable(batch_advantages.data, requires_grad=False)
            batch_logprobs = Variable(batch_logprobs.data, requires_grad=False)

            batch_z_d = multi_env_z_sample.detach().unsqueeze(1).expand(-1, batch_states.size(1), -1) 
            mu, sigma_sq, v_pred, mu_p, logvar_p, z_p = agent(batch_states, batch_z_d)
            log_std = agent.log_std
            log_probs = -0.5 * ((batch_actions - mu) / sigma_sq).pow(2) - 0.5 * math.log(2 * math.pi) - log_std
            log_probs = log_probs.sum(-1, keepdim=True)
            dist_entropy = 0.5 + 0.5 * math.log(2 * math.pi) + log_std
            dist_entropy = dist_entropy.sum(-1).mean()

            ratio = torch.exp(log_probs - batch_logprobs)

            surr1 = ratio * batch_advantages.expand_as(ratio) 
            surr2 = ratio.clamp(1-clip, 1+clip) * batch_advantages.expand_as(ratio)
            loss_clip = - torch.mean(torch.min(surr1, surr2))
            
            kld_loss = kl_div_prior(mu_p, logvar_p)
            
            loss_value = (v_pred - batch_returns).pow(2).mean() + 1*kld_loss

            loss_ent = - ent_coeff * dist_entropy

            total_loss = (loss_clip + loss_value + loss_ent)
            pn_optim.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm(agent.parameters(), max_grad_norm)
            pn_optim.step() 

        multi_env_memory.multi_env_clear()
        
        if eval_traj_in_env:
            multi_env_rewards_arr = multi_env_eval_ppo_in_env(env_list, env_name, agent, meta_dm, whether_lvm, check_lvm, 
                                                              multi_env_x_memory, multi_env_y_memory, stats_list, num_steps, 
                                                              policy_type = 'H', num_iter = 5)
            avg_rewards=np.mean(multi_env_rewards_arr)

    return agent


def p_relabel_multi_env_memory_update(meta_dm, env_list, env_name, stats_list, meta_rewards_torch, multi_env_memory, 
                                      agent, num_steps, gamma, gae_param, whether_lvm, check_lvm, norm_multi_env_x_memory, 
                                      norm_multi_env_y_memory, multi_env_z_sample=None):
    num_envs = len(env_list)
    
    multi_env_context_obs = torch.cat((norm_multi_env_x_memory, norm_multi_env_y_memory), dim=-1)
    
    for task_idx in range(num_envs):
    
        state = env_list[task_idx].reset()
        state = Variable(torch.Tensor(state).unsqueeze(0))
        done = True
        
        episode_length = 0 

        while(len(multi_env_memory.memory[task_idx])<num_steps):
            states = []
            actions = []
            rewards = []
            values = []
            returns = []
            advantages = []
            logprobs = []
            av_reward = 0
            cum_reward = 0
            cum_done = 0
    
            for step in range(num_steps):
                episode_length += 1
                
                if step==0:    
                    preprocessed_state = (dm_input_preporcess(state, env_name)).cuda()
                else:
                    preprocessed_state = (dm_input_preporcess(state, env_name, init_state=False)).cuda()
                state = normalize_dataset(stats_list[task_idx], preprocessed_state, var_order=1) 
                
                #########################################################
                if whether_lvm:
                    context_obs=multi_env_context_obs[task_idx,...].detach() 

                    v, action, log_prob, mu_p, logvar_p, z_p = agent.act(state.float(),context_obs.float(),return_v=True)
                    states.append(state)
                else:
                    v, action, log_prob = agent.act(state,return_v=True)
                    states.append(state)
                    
                actions.append(action)
                logprobs.append(log_prob)
                values.append(v)
                
                normalized_action = normalize_dataset(stats_list[task_idx], action, var_order=2) 
                state_action_tensor = (torch.cat((state, normalized_action), dim=-1)).unsqueeze(0) 
            
                if check_lvm == 'NP_DM' :
                    mu_c, logvar_c, mu_t, logvar_t, delta_usq = meta_dm(norm_multi_env_x_memory[task_idx:task_idx+1,...],norm_multi_env_y_memory[task_idx:task_idx+1,...],
                                                                        norm_multi_env_x_memory[task_idx:task_idx+1,...],norm_multi_env_y_memory[task_idx:task_idx+1,...],
                                                                        state_action_tensor)
                elif check_lvm == 'AttnNP_DM' :
                    mu_c, logvar_c, mu_t, logvar_t, delta_usq=meta_dm(norm_multi_env_x_memory[task_idx:task_idx+1,...],norm_multi_env_y_memory[task_idx:task_idx+1,...],
                                                                      norm_multi_env_x_memory[task_idx:task_idx+1,...],norm_multi_env_y_memory[task_idx:task_idx+1,...],
                                                                      state_action_tensor)
                elif check_lvm == 'GS_DM' :
                    mu_c, logvar_c, mu_t, logvar_t, delta_usq=meta_dm(norm_multi_env_x_memory[task_idx:task_idx+1,...],norm_multi_env_y_memory[task_idx:task_idx+1,...],
                                                                      state_action_tensor)
                else:
                    delta_usq = meta_dm(state_action_tensor)  
                
                denormalized_delta = denormalize_dataset(stats_list[task_idx], delta_usq.squeeze(0), var_order=3)
                denormalized_state = denormalize_dataset(stats_list[task_idx], state, var_order=1) 
                
                denormalized_next_state = dm_output_postprocess(denormalized_state, denormalized_delta, env_name)

                reward = (meta_rewards_torch(denormalized_state,action,denormalized_next_state)).item()

                done = (episode_length >= num_steps)
                cum_reward += reward
                reward = max(min(reward, 1), -1)
                rewards.append(reward)
                
                state = denormalized_next_state
                
                if done:
                    cum_done += 1
                    av_reward += cum_reward
                    cum_reward = 0
                    episode_length = 0
                    state = env_list[task_idx].reset()
                    state = Variable(torch.Tensor(state).unsqueeze(0)).cuda()
                
                if done:
                    break

            R = torch.zeros(1, 1)
            if not done:
                last_state = dm_input_preporcess(state, env_name)
                last_state = normalize_dataset(stats_list[task_idx], last_state, var_order=1)               
                if whether_lvm:
                    context_obs=multi_env_context_obs[task_idx,...].detach()
                    
                    v, action, log_prob, mu_p, logvar_p, z_p = agent.act(state.float(),context_obs.float(),return_v=True)
                else:
                    v, action, log_prob = agent.act(last_state,return_v=True)
                
                R = v.data

            R = Variable(R)
            values.append(R)
            A = Variable(torch.zeros(1, 1)).cuda()
            for i in reversed(range(len(rewards))):
                td = rewards[i] + gamma*values[i+1].data[0,0] - values[i].data[0,0]
                A = float(td) + gamma*gae_param*A
                advantages.insert(0, A)
                R = A + values[i]
                returns.insert(0, R)

            multi_env_memory.push([states, actions, returns, advantages, logprobs],task_idx)
        
    return multi_env_memory


def p_multi_env_train_ppo_in_dm(meta_dm, env_list, env_name, agent, pn_optim, whether_lvm, check_lvm, 
                                multi_env_x_memory, multi_env_y_memory, stats_list, meta_rewards_torch, 
                                clip = 0.2, num_episode = 100, num_epoch = 10, 
                                num_steps = 1000, gamma = 0.99, gae_param = 0.95, ent_coeff = 0.01, memo_bsize = 64, 
                                max_grad_norm = 0.5, collect_memory = True, eval_traj_in_env = True):
    
    meta_dm.eval()

    for param in meta_dm.parameters():
        param.requires_grad = False 
        
    if collect_memory:
        norm_multi_env_x_memory, norm_multi_env_y_memory = multi_env_normalize_dataset(stats_list, multi_env_x_memory, var_order=0), \
            multi_env_normalize_dataset(stats_list, multi_env_y_memory, var_order=3)

    if whether_lvm == True :
        if check_lvm == 'GS_DM' :
            mu_t,logvar_t,mu_c,logvar_c,y_pred=meta_dm(norm_multi_env_x_memory,norm_multi_env_y_memory,norm_multi_env_x_memory)
            multi_env_z_sample=meta_dm.reparameterization(torch.mean(mu_c,dim=1),torch.mean(logvar_c,dim=1))
            
    else:
        multi_env_z_sample = None
    
    num_envs = len(env_list)
    
    multi_env_memory = Multi_Env_ReplayMemory(num_steps, num_envs)
    multi_env_context_obs = torch.cat((norm_multi_env_x_memory, norm_multi_env_y_memory), dim=-1) 
    
    for t in range(num_episode):

        multi_env_memory = p_relabel_multi_env_memory_update(meta_dm, env_list, env_name, stats_list, 
                                                             meta_rewards_torch, multi_env_memory, 
                                                             agent, num_steps, gamma, gae_param, 
                                                             whether_lvm, check_lvm, norm_multi_env_x_memory, 
                                                             norm_multi_env_y_memory, multi_env_z_sample=None)        
        
        for k in range(num_epoch):
            batch_states, batch_actions, batch_returns, batch_advantages, batch_logprobs = multi_env_memory.multi_env_sample(memo_bsize)
            batch_actions = Variable(batch_actions.data, requires_grad=False)
            batch_states = Variable(batch_states.data, requires_grad=False)
            batch_returns = Variable(batch_returns.data, requires_grad=False)
            batch_advantages = Variable(batch_advantages.data, requires_grad=False)
            batch_logprobs = Variable(batch_logprobs.data, requires_grad=False)

            mu, sigma_sq, v_pred, mu_p, logvar_p, z_p = agent(batch_states, multi_env_context_obs)
            log_std = agent.log_std
            log_probs = -0.5 * ((batch_actions - mu) / sigma_sq).pow(2) - 0.5 * math.log(2 * math.pi) - log_std
            log_probs = log_probs.sum(-1, keepdim=True)
            dist_entropy = 0.5 + 0.5 * math.log(2 * math.pi) + log_std
            dist_entropy = dist_entropy.sum(-1).mean()

            ratio = torch.exp(log_probs - batch_logprobs)

            surr1 = ratio * batch_advantages.expand_as(ratio)
            surr2 = ratio.clamp(1-clip, 1+clip) * batch_advantages.expand_as(ratio)
            loss_clip = - torch.mean(torch.min(surr1, surr2))
            
            kld_loss = kl_div_prior(mu_p, logvar_p)
            
            loss_value = (v_pred - batch_returns).pow(2).mean() + 1*kld_loss

            loss_ent = - ent_coeff * dist_entropy

            total_loss = (loss_clip + loss_value + loss_ent)
            pn_optim.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm(agent.parameters(), max_grad_norm)
            pn_optim.step() 

        multi_env_memory.multi_env_clear()
        
        if eval_traj_in_env:
            multi_env_rewards_arr = multi_env_eval_ppo_in_env(env_list, env_name, agent, meta_dm, whether_lvm, check_lvm, 
                                                              multi_env_x_memory, multi_env_y_memory, stats_list, num_steps, 
                                                              policy_type = 'P', num_iter = 5)
            avg_rewards=np.mean(multi_env_rewards_arr)

    return agent



def pn_ppo_rollout(env, env_name, agent, meta_dm, whether_lvm, check_lvm, x_memory, y_memory, stats,
                   num_steps, policy_type, init_states = None, render = False):

    meta_dm.eval()
    
    for param in meta_dm.parameters():
        param.requires_grad = False
    
    x_memo_c_norm, y_memo_c_norm = normalize_dataset(stats, x_memory, var_order=0), normalize_dataset(stats, y_memory, var_order=3)
    context_obs = (torch.cat((x_memo_c_norm, y_memo_c_norm), dim=-1)).squeeze(0) 
    
    if init_states is None:
        cur_states = torch.FloatTensor(env.reset()).cuda()        
    else:
        cur_states = torch.FloatTensor(init_states)
       
    state_list = [cur_states.data.cpu().numpy()]
    action_list = []
    reward_list = [] 

    if whether_lvm == True :
        if check_lvm == 'GS_DM' :
            mu_t,logvar_t,mu_c,logvar_c,y_pred=meta_dm(x_memo_c_norm,y_memo_c_norm,x_memo_c_norm)
            z_sample=meta_dm.reparameterization(torch.mean(mu_c,dim=1),torch.mean(logvar_c,dim=1))
            
    states = cur_states.unsqueeze(0)       
    if render: env.render()
    
    for t in range(num_steps):
        
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
        state_array, action_array, reward_array = state_array[None,:], action_array[None,:], reward_array[None,:]
        multi_env_state_list.append(state_array)
        multi_env_action_list.append(action_array)
        multi_env_reward_list.append(reward_array)
        
    multi_env_state_array, multi_env_action_array, multi_env_reward_array = np.array(multi_env_state_list), \
        np.array(multi_env_action_list), np.array(multi_env_reward_list) 
        
    return state_array, action_array, reward_array



##########################################################################################################################
    #evaluate learned policies in real-world environments
##########################################################################################################################


def eval_ppo_in_env(env, env_name, agent, meta_dm, whether_lvm, check_lvm, 
                    x_memo_c, y_memo_c, stats, num_steps, policy_type, num_iter):

    rewards_arr = np.zeros([num_iter])

    for i in range(num_iter):
        states, actions, rewards = pn_ppo_rollout(env, env_name, agent, meta_dm, whether_lvm, check_lvm, 
                                                  x_memo_c, y_memo_c, stats, num_steps, policy_type)
        rewards_arr[i] = np.sum(rewards)
    
    return rewards_arr



def multi_env_eval_ppo_in_env(env_list, env_name, agent, meta_dm, whether_lvm, check_lvm, 
                              multi_env_x_memory, multi_env_y_memory, stats_list, num_steps,
                              policy_type, num_iter):
    num_envs = len(env_list)
    multi_env_rewards_arr = np.zeros([num_envs])
    
    for i in range(num_envs):

        multi_env_rewards_arr = eval_ppo_in_env(env_list[i], env_name, agent, meta_dm, whether_lvm, check_lvm, 
                                                multi_env_x_memory[i:i+1,...], multi_env_y_memory[i:i+1,...], 
                                                stats_list[i], num_steps, policy_type, num_iter)    
    
    return multi_env_rewards_arr