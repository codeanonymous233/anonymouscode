'''
This file is to construct dynamics model for meta reinforcement learning tasks.
'''

import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable
from utils import get_act, Multi_Attn, GC_Net


    
class GS_DM(nn.Module):

    def __init__(self,args):
        super(GS_DM,self).__init__()
        
        self.dim_x=args.dim_x
        self.dim_y=args.dim_y
        
        self.dim_emb_x=args.dim_emb_x 
        self.dim_lat=args.dim_lat 
        self.dim_h_lat=args.dim_h_lat 
        self.num_h_lat=args.num_h_lat 
        
        self.dim_h=args.dim_h 
        self.num_h=args.num_h 
        self.act_type=args.act_type 
        self.amort_y=args.amort_y  
  
        self.gc_net=GC_Net(self.dim_x, self.dim_y, self.dim_emb_x, self.dim_lat, self.dim_h_lat, self.num_h_lat).cuda()
            
        self.mu_net=nn.Sequential(nn.Linear(self.dim_lat, self.dim_lat)).cuda() 
        self.logvar_net=nn.Sequential(nn.Linear(self.dim_lat, self.dim_lat)).cuda() 
        
        self.mu_net_g=nn.Sequential(nn.Linear(self.dim_lat, self.dim_lat)).cuda() 
        self.logvar_net_g=nn.Sequential(nn.Linear(self.dim_lat, self.dim_lat)).cuda()       
        
        self.dec_modules=[]
        self.dec_modules.append(nn.Linear(self.dim_x+self.dim_lat, self.dim_h))
        for i in range(args.num_h):
            self.dec_modules.append(get_act(args.act_type))
            self.dec_modules.append(nn.Linear(self.dim_h, self.dim_h))
        if self.amort_y:
            self.dec_modules.append(get_act(args.act_type))
            self.dec_modules.append(nn.Linear(self.dim_h, 2*self.dim_y)) 
        else:
            self.dec_modules.append(get_act(args.act_type))
            self.dec_modules.append(nn.Linear(self.dim_h, self.dim_y))           
        self.dec_net=nn.Sequential(*self.dec_modules).cuda()
    

    
    def get_context_idx(self,M):
        N = random.randint(1,M)
        idx = random.sample(range(0, M), N)
        idx = torch.tensor(idx).cuda()
        
        return idx
        
        
    def idx_to_data(self,data,sample_dim,idx):
        ind_data= torch.index_select(data, dim=sample_dim, index=idx)
        
        return ind_data
    
    
    def reparameterization(self,mu,logvar):
        std=torch.exp(0.5*logvar)
        eps=torch.randn_like(std)
        
        return mu+eps*std
    
    
    def forward(self,x_memory,y_memory,x_pred):

        c_message_aggregation, c2t_message_aggregation, gc_emb=self.gc_net(x_memory,x_pred,y_memory)
        mu=self.mu_net(gc_emb) 
        logvar=self.logvar_net(gc_emb) 
        mu_g=self.mu_net_g(c2t_message_aggregation)
        logvar_g=self.logvar_net_g(c2t_message_aggregation)
    
        z=self.reparameterization(mu, logvar) 
        
        output=self.dec_net(torch.cat((x_pred,z),dim=-1)) 
        
        if self.amort_y:
            y_mean,y_var=output[:,:,:self.dim_y],F.softplus(output[:,:,self.dim_y:])
            return mu,logvar,mu_g,logvar_g,y_mean,y_var
        else:
            y_pred=output
            return mu,logvar,mu_g,logvar_g,y_pred 
       


class NP_DM(nn.Module):
    
    def __init__(self,args):
        super(NP_DM,self).__init__()
        
        self.dim_x=args.dim_x
        self.dim_y=args.dim_y
        
        self.dim_h_lat=args.dim_h_lat 
        self.num_h_lat=args.num_h_lat 
        self.dim_lat=args.dim_lat 
        
        self.dim_h=args.dim_h 
        self.num_h=args.num_h 
        self.act_type=args.act_type 
        self.amort_y=args.amort_y      

        self.emb_c_modules=[]
        self.emb_c_modules.append(nn.Linear(self.dim_x+self.dim_y,self.dim_h_lat))
        for i in range(self.num_h_lat):
            self.emb_c_modules.append(get_act(self.act_type))
            self.emb_c_modules.append(nn.Linear(self.dim_h_lat,self.dim_h_lat))
        self.emb_c_modules.append(get_act(self.act_type))
        self.context_net=nn.Sequential(*self.emb_c_modules).cuda()
        
        self.mu_net=nn.Linear(self.dim_h_lat, self.dim_lat).cuda() 
        self.logvar_net=nn.Linear(self.dim_h_lat, self.dim_lat).cuda() 
        
        self.dec_modules=[]
        self.dec_modules.append(nn.Linear(self.dim_x+self.dim_lat, self.dim_h))
        for i in range(self.num_h):
            self.dec_modules.append(get_act(self.act_type))
            self.dec_modules.append(nn.Linear(self.dim_h, self.dim_h))
        if self.amort_y:
            self.dec_modules.append(get_act(self.act_type))
            self.dec_modules.append(nn.Linear(self.dim_h, 2*self.dim_y)) 
        else:
            self.dec_modules.append(get_act(self.act_type))
            self.dec_modules.append(nn.Linear(self.dim_h, self.dim_y))          
        self.dec_net=nn.Sequential(*self.dec_modules).cuda() 

    
    
    def get_context_idx(self,M):

        N = random.randint(1,M)
        idx = random.sample(range(0, M), N)
        idx = torch.tensor(idx).cuda()
        
        return idx
        
        
    def idx_to_data(self,data,sample_dim,idx):

        ind_data= torch.index_select(data, dim=sample_dim, index=idx)
        
        return ind_data
        
        
    def emb_aggregator(self,h_context,aggre_dim=1):

        h_aggre=torch.mean(h_context,dim=aggre_dim) 
        
        return h_aggre
    
    
    def reparameterization(self,mu,logvar):

        std=torch.exp(0.5*logvar)
        eps=torch.randn_like(std)
        
        return mu+eps*std

    
    def encoder(self,x_memo_c,y_memo_c,x_memo_t,y_memo_t):

        if self.training:
            memo_c,memo_t=torch.cat((x_memo_c,y_memo_c),dim=-1),torch.cat((x_memo_t,y_memo_t),dim=-1)
            memo_emb_c,memo_emb_t=self.context_net(memo_c),self.context_net(memo_t)
            
            h_c,h_t=self.emb_aggregator(memo_emb_c),self.emb_aggregator(memo_emb_t)
            mu_c,logvar_c=self.mu_net(h_c),self.logvar_net(h_c)
            mu_t,logvar_t=self.mu_net(h_t),self.logvar_net(h_t)
        else:
            memo_c=torch.cat((x_memo_c,y_memo_c),dim=-1)
            memo_emb_c=self.context_net(memo_c)
            
            h_c=self.emb_aggregator(memo_emb_c)
            mu_c,logvar_c=self.mu_net(h_c),self.logvar_net(h_c)
            mu_t,logvar_t=0,0
        
        return mu_c,logvar_c,mu_t,logvar_t
            
        
    def forward(self,x_memo_c,y_memo_c,x_memo_t,y_memo_t,x_pred):

        mu_c,logvar_c,mu_t,logvar_t=self.encoder(x_memo_c, y_memo_c, x_memo_t, y_memo_t)
        z_g=self.reparameterization(mu_c, logvar_c) 
        
        z_g_unsq=z_g.unsqueeze(1).expand(-1,x_pred.size(1),-1)
        output=self.dec_net(torch.cat((x_pred,z_g_unsq),dim=-1)) 
        if self.amort_y:
            y_mean,y_var=output[...,:self.dim_y],F.softplus(output[...,self.dim_y:])
            return mu_c,logvar_c,mu_t,logvar_t,y_mean,y_var
        else:
            y_pred=output
            return mu_c,logvar_c,mu_t,logvar_t,y_pred
            
    

class AttnNP_DM(nn.Module):

    def __init__(self,args):
        super(AttnNP_DM,self).__init__()
        
        self.dim_x=args.dim_x
        self.dim_y=args.dim_y
        
        self.dim_h_lat=args.dim_h_lat
        self.num_h_lat=args.num_h_lat 
        self.dim_lat=args.dim_lat
        self.num_head=args.num_head 
        self.dim_emb_x=args.dim_emb_x         
        
        self.dim_h=args.dim_h 
        self.num_h=args.num_h 
        self.act_type=args.act_type 
        self.amort_y=args.amort_y        

        self.emb_c_modules=[]
        self.emb_c_modules.append(nn.Linear(self.dim_x+self.dim_y,self.dim_h_lat,bias=False))
        for i in range(self.num_h_lat):
            self.emb_c_modules.append(get_act(self.act_type))
            self.emb_c_modules.append(nn.Linear(self.dim_h_lat,self.dim_h_lat,bias=False))

        self.emb_c_modules.append(get_act(self.act_type))
        self.context_net=nn.Sequential(*self.emb_c_modules).cuda()
        
        self.mu_net=nn.Linear(self.dim_h_lat, self.dim_lat).cuda()
        self.logvar_net=nn.Linear(self.dim_h_lat, self.dim_lat).cuda() 
        
        self.dot_attn=Multi_Attn(self.dim_x, self.dim_x+self.dim_y, self.dim_emb_x, 
                                 self.dim_lat, self.num_head) 
        
        self.dec_modules=[]
        self.dec_modules.append(nn.Linear(self.dim_x+self.dim_lat+self.num_head*self.dim_lat, self.dim_h))        
        for i in range(self.num_h):
            self.dec_modules.append(get_act(self.act_type))
            self.dec_modules.append(nn.Linear(self.dim_h, self.dim_h))
        if self.amort_y:
            self.dec_modules.append(get_act(self.act_type))
            self.dec_modules.append(nn.Linear(self.dim_h, 2*self.dim_y)) 
        else:
            self.dec_modules.append(get_act(self.act_type))
            self.dec_modules.append(nn.Linear(self.dim_h, self.dim_y))           
        self.dec_net=nn.Sequential(*self.dec_modules).cuda()
    
    
    
    def get_context_idx(self,M):

        N = random.randint(1,M)
        idx = random.sample(range(0, M), N)
        idx = torch.tensor(idx).cuda()
        
        return idx
        
        
    def idx_to_data(self,data,sample_dim,idx):

        ind_data= torch.index_select(data, dim=sample_dim, index=idx)
        
        return ind_data
        
        
    def emb_aggregator(self,h_context,aggre_dim=1):

        h_aggre=torch.mean(h_context,dim=aggre_dim) 
        
        return h_aggre
    
    
    def reparameterization(self,mu,logvar):

        std=torch.exp(0.5*logvar)
        eps=torch.randn_like(std)
        
        return mu+eps*std

    
    def encoder(self,x_memo_c,y_memo_c,x_memo_t,y_memo_t):

        if self.training:
            memo_c,memo_t=torch.cat((x_memo_c,y_memo_c),dim=-1),torch.cat((x_memo_t,y_memo_t),dim=-1)
            memo_emb_c,memo_emb_t=self.context_net(memo_c),self.context_net(memo_t)
            
            h_c,h_t=self.emb_aggregator(memo_emb_c),self.emb_aggregator(memo_emb_t)
            mu_c,logvar_c=self.mu_net(h_c),self.logvar_net(h_c)
            mu_t,logvar_t=self.mu_net(h_t),self.logvar_net(h_t)
            
        else:
            memo_c=torch.cat((x_memo_c,y_memo_c),dim=-1)
            memo_emb_c=self.context_net(memo_c)
            
            h_c=self.emb_aggregator(memo_emb_c)
            mu_c,logvar_c=self.mu_net(h_c),self.logvar_net(h_c)
            mu_t,logvar_t=0,0            
        
        return mu_c,logvar_c,mu_t,logvar_t
            
        
        
    def forward(self,x_memo_c,y_memo_c,x_memo_t,y_memo_t,x_pred):

        initial_value=torch.cat((x_memo_t,y_memo_t),dim=-1)
        dot_attn_v,dot_attn_weight=self.dot_attn(x_memo_t,x_pred,initial_value)
        
        mu_c,logvar_c,mu_t,logvar_t=self.encoder(x_memo_c, y_memo_c, x_memo_t, y_memo_t)
        z_g=self.reparameterization(mu_c,logvar_c) 
        z_g_unsq=z_g.unsqueeze(1).expand(-1,x_pred.size(1),-1)
        c_merg=torch.cat((dot_attn_v,z_g_unsq),dim=-1)
        
        output=self.dec_net(torch.cat((x_pred,c_merg),dim=-1)) 
        if self.amort_y:
            y_mean,y_var=output[...,:self.dim_y],F.softplus(output[...,self.dim_y:])
            return mu_c,logvar_c,mu_t,y_mean,y_var
        else:
            y_pred=output
            return mu_c,logvar_c,mu_t,logvar_t,y_pred        
    
    

    
    

            

 






