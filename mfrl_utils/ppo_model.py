'''
Mainly refer to https://github.com/alexis-jacq/Pytorch-DPPO.
'''

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



env_name='half_cheetah'

if env_name == 'humanoid':
    act_scale_value=0.4
else:
    act_scale_value = 1.0
    

class Model(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Model, self).__init__()
        h_size_1 = 100
        h_size_2 = 100
        log_std = -2.3 * np.ones(num_outputs, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.FloatTensor(log_std), requires_grad=True)

        self.p_fc1 = nn.Linear(num_inputs, h_size_1)
        self.p_fc2 = nn.Linear(h_size_1, h_size_2)

        self.v_fc1 = nn.Linear(num_inputs, h_size_1*5)
        self.v_fc2 = nn.Linear(h_size_1*5, h_size_2)

        self.mu = nn.Linear(h_size_2, num_outputs)

        self.v = nn.Linear(h_size_2,1)

        for name, p in self.named_parameters():
            # init parameters
            if 'bias' in name:
                p.data.fill_(0)

        # mode
        self.train()


    def forward(self, inputs, act_scale=act_scale_value):
        if inputs.is_cuda==False:
            inputs = inputs.cuda()
            
        # actor
        x = F.relu(self.p_fc1(inputs))
        x = F.tanh(self.p_fc2(x))
        mu = act_scale*F.tanh(self.mu(x))
        
        log_std = torch.clamp(self.log_std, -10.0, -2.3)
        sigma_sq = torch.exp(log_std)

        # critic
        x = F.relu(self.v_fc1(inputs))
        x = F.tanh(self.v_fc2(x))
        v = self.v(x)
        return mu, sigma_sq, v
    
    
    def act(self, state, return_v=False):
        self.eval()
        with torch.no_grad():
            mu, sigma_sq, v = self.forward(state)

        mu_var=(Variable(torch.randn(mu.size()))).cuda()
        action = (mu + sigma_sq*mu_var)

        log_std = self.log_std
        log_prob = -0.5 * ((action - mu) / (sigma_sq+1e-6)).pow(2) - 0.5 * math.log(2 * math.pi) - log_std
        log_prob = log_prob.sum(-1, keepdim=True)
        if return_v:
            return v.detach(), action.detach(), log_prob.detach()
        else:
            return action.detach(), log_prob.detach()
          
        
class Hierarchical_Model(nn.Module):
    def __init__(self, dim_input, dim_output, dim_lat, dim_h, num_h):
        super(Hierarchical_Model, self).__init__()
        h_size_1 = 100
        h_size_2 = 100
        
        log_std = -2.3 * np.ones(dim_output, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.FloatTensor(log_std), requires_grad=True)
        
        self.emb_modules=[]
        self.emb_modules.append(nn.Linear(dim_lat, dim_h))
        for i in range(num_h):
            self.emb_modules.append(nn.ReLU())
            self.emb_modules.append(nn.Linear(dim_h, dim_h))
        self.context_net=nn.Sequential(*self.emb_modules).cuda() 
        self.mu_net=nn.Linear(dim_h, dim_lat).cuda() 
        self.logvar_net=nn.Linear(dim_h, dim_lat).cuda()        
        
        self.p_fc1 = nn.Linear(dim_input+dim_lat, h_size_1)
        self.p_fc2 = nn.Linear(h_size_1, h_size_2)
        
        self.v_fc1 = nn.Linear(dim_input+dim_lat, h_size_1*5)
        self.v_fc2 = nn.Linear(h_size_1*5, h_size_2)

        self.mu = nn.Linear(h_size_2, dim_output)

        self.v = nn.Linear(h_size_2,1)

        for name, p in self.named_parameters():
            # init parameters
            if 'bias' in name:
                p.data.fill_(0)

        # mode
        self.train()

    
    def task_encoder(self, dm_lat):
        h_emb = self.context_net(dm_lat)
        mu_p, logvar_p = self.mu_net(h_emb), self.logvar_net(h_emb)
        
        std=torch.exp(0.5*logvar_p)
        eps=torch.randn_like(std)
        z_p = mu_p + eps*std
        
        if h_emb.dim() == 2:
            return mu_p, logvar_p, z_p
        elif h_emb.dim() == 3: 
            return torch.mean(mu_p,dim=1), torch.mean(logvar_p,dim=1), z_p
        
        
    def forward(self, inputs, dm_lat, act_scale=act_scale_value):
        if inputs.is_cuda==False:
            inputs = inputs.cuda()
        
        mu_p, logvar_p, z_p = self.task_encoder(dm_lat) 
        
        relabeled_inputs = torch.cat((inputs, z_p), dim=-1)
        
        x = F.relu(self.p_fc1(relabeled_inputs))
        x = F.tanh(self.p_fc2(x))
        mu = act_scale*F.tanh(self.mu(x))
        
        log_std = torch.clamp(self.log_std, -10.0, -2.3)
        sigma_sq = torch.exp(log_std)

        x = F.relu(self.v_fc1(relabeled_inputs))
        x = F.tanh(self.v_fc2(x))
        v = self.v(x)
        return mu, sigma_sq, v, mu_p, logvar_p, z_p
    
    
    def act(self, state, dm_lat, return_v=False):
        self.eval()
        
        with torch.no_grad():
            mu, sigma_sq, v, mu_p, logvar_p, z_p = self.forward(state, dm_lat)

        mu_var=(Variable(torch.randn(mu.size()))).cuda()
        action = (mu + sigma_sq*mu_var)

        log_std = self.log_std
        log_prob = -0.5 * ((action - mu) / (sigma_sq+1e-6)).pow(2) - 0.5 * math.log(2 * math.pi) - log_std
        log_prob = log_prob.sum(-1, keepdim=True)
        if return_v:
            return v.detach(), action.detach(), log_prob.detach(), mu_p.detach(), logvar_p.detach(), z_p.detach()
        else:
            return action.detach(), log_prob.detach(), mu_p.detach(), logvar_p.detach(), z_p.detach()



class Parallel_Model(nn.Module):
    def __init__(self, dim_input, dim_output, dim_obs_x, dim_obs_y, dim_lat, dim_h, num_h):
        super(Parallel_Model, self).__init__()
        h_size_1 = 100
        h_size_2 = 100
        
        log_std = -2.3 * np.ones(dim_output, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.FloatTensor(log_std), requires_grad=True)
        
        self.emb_modules=[]
        self.emb_modules.append(nn.Linear(dim_obs_x+dim_obs_y, dim_h))
        for i in range(num_h):
            self.emb_modules.append(nn.ReLU())
            self.emb_modules.append(nn.Linear(dim_h, dim_h))
        self.context_net=nn.Sequential(*self.emb_modules).cuda() 
        self.mu_net=nn.Linear(dim_h, dim_lat).cuda() 
        self.logvar_net=nn.Linear(dim_h, dim_lat).cuda()        
        
        self.p_fc1 = nn.Linear(dim_input+dim_lat, h_size_1)
        self.p_fc2 = nn.Linear(h_size_1, h_size_2)
        
        self.v_fc1 = nn.Linear(dim_input+dim_lat, h_size_1*5)
        self.v_fc2 = nn.Linear(h_size_1*5, h_size_2)

        self.mu = nn.Linear(h_size_2, dim_output)

        self.v = nn.Linear(h_size_2,1)

        for name, p in self.named_parameters():
            if 'bias' in name:
                p.data.fill_(0)

        # mode
        self.train()
    
    
    def task_encoder(self, context_obs):
        h_emb = self.context_net(context_obs)
        if context_obs.dim() == 2: 
            h_emb = torch.mean(h_emb,dim=0)
        elif context_obs.dim() == 3: 
            h_emb = torch.mean(h_emb,dim=1)
        mu_p, logvar_p = self.mu_net(h_emb), self.logvar_net(h_emb)
        
        std=torch.exp(0.5*logvar_p)
        eps=torch.randn_like(std)
        z_p = mu_p + eps*std
        
        return mu_p, logvar_p, z_p
        
        
    def forward(self, inputs, context_obs, act_scale=act_scale_value):
        if inputs.is_cuda==False:
            inputs = inputs.cuda()
        
        mu_p, logvar_p, z_p = self.task_encoder(context_obs)
        
        if inputs.dim()==1: 
            relabeled_inputs = torch.cat((inputs, z_p), dim=-1)
        elif inputs.dim()==2: 
            batch_z_p = z_p.unsqueeze(0).expand(inputs.size(0),-1)
            relabeled_inputs = torch.cat((inputs, batch_z_p), dim=-1)
        elif inputs.dim()==3: 
            batch_z_p = z_p.unsqueeze(1).expand(-1,inputs.size(1),-1)
            relabeled_inputs = torch.cat((inputs, batch_z_p), dim=-1)

        x = F.relu(self.p_fc1(relabeled_inputs))
        x = F.tanh(self.p_fc2(x))
        mu = act_scale*F.tanh(self.mu(x))
        
        log_std = torch.clamp(self.log_std, -10.0, -2.3)
        sigma_sq = torch.exp(log_std)

        x = F.relu(self.v_fc1(relabeled_inputs))
        x = F.tanh(self.v_fc2(x))
        v = self.v(x)
        return mu, sigma_sq, v, mu_p, logvar_p, z_p
    
    
    def act(self, state, context_obs, return_v=False):
        self.eval()
        
        with torch.no_grad():
            mu, sigma_sq, v, mu_p, logvar_p, z_p = self.forward(state, context_obs)

        mu_var=(Variable(torch.randn(mu.size()))).cuda()
        action = (mu + sigma_sq*mu_var)

        log_std = self.log_std
        log_prob = -0.5 * ((action - mu) / (sigma_sq+1e-6)).pow(2) - 0.5 * math.log(2 * math.pi) - log_std
        log_prob = log_prob.sum(-1, keepdim=True)
        if return_v:
            return v.detach(), action.detach(), log_prob.detach(), mu_p.detach(), logvar_p.detach(), z_p.detach()
        else:
            return action.detach(), log_prob.detach(), mu_p.detach(), logvar_p.detach(), z_p.detach()
        
        
        
##########################################################################################################################
        #some utils to preprocess observations
##########################################################################################################################  
        
        
class Shared_grad_buffers():
    def __init__(self, model):
        self.grads = {}
        for name, p in model.named_parameters():
            self.grads[name+'_grad'] = torch.ones(p.size()).share_memory_()

    def add_gradient(self, model):
        for name, p in model.named_parameters():
            self.grads[name+'_grad'] += p.grad.data

    def reset(self):
        for name,grad in self.grads.items():
            self.grads[name].fill_(0)



class Shared_obs_stats():
    def __init__(self, num_inputs):
        self.n = torch.zeros(num_inputs).share_memory_()
        self.mean = torch.zeros(num_inputs).share_memory_()
        self.mean_diff = torch.zeros(num_inputs).share_memory_()
        self.var = torch.zeros(num_inputs).share_memory_()

    def observes(self, obs):

        if obs.is_cuda:
            x=obs.data.squeeze().cpu()
        else:
            x = obs.data.squeeze()
        self.n += 1.
        last_mean = self.mean.clone()
        self.mean += (x-self.mean)/self.n
        self.mean_diff += (x-last_mean)*(x-self.mean)
        self.var = torch.clamp(self.mean_diff/self.n, min=1e-2)

    def normalize(self, inputs):
        if inputs.is_cuda:
            inputs=inputs.cpu()
    
        obs_mean = Variable(self.mean.unsqueeze(0).expand_as(inputs))
        obs_std = Variable(torch.sqrt(self.var).unsqueeze(0).expand_as(inputs))
        return torch.clamp((inputs-obs_mean)/obs_std, -5., 5.)
    
    

