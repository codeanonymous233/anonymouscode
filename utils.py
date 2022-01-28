'''
This file is to record utilities in implementations.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from pathlib import Path
from typing import Union



def get_act(act_type):

    if act_type=='ReLU':
        return nn.ReLU()
    elif act_type=='LeakyReLU':
        return nn.LeakyReLU()
    elif act_type=='ELU':
        return nn.ELU()
    elif act_type=='Sigmoid':
        return nn.Sigmoid()
    else:
        raise ValueError('Invalid argument for act_type')
    
    

    
class Multi_Attn(nn.Module):

    def __init__(self,dim_k_in,dim_v_in,dim_k,dim_v,num_head):
        super(Multi_Attn,self).__init__()
    
        self.dim_k_in=dim_k_in 
        self.dim_v_in=dim_v_in 
        self.dim_k=dim_k 
        self.dim_v=dim_v 
        self.num_head=num_head 

        self.fc_k=nn.Linear(self.dim_k_in, self.num_head*self.dim_k,bias=False).cuda() 
        self.fc_q=nn.Linear(self.dim_k_in, self.num_head*self.dim_k,bias=False).cuda() 
        self.fc_v=nn.Linear(self.dim_v_in, self.num_head*self.dim_v,bias=False).cuda()         
    
    
    def emb_aggregator(self,h_context,aggre_dim):

        h_aggre=torch.mean(h_context,dim=aggre_dim)
        
        return h_aggre
    
    
    def forward(self,key,query,value):

        assert key.dim() == 3
        
        len_k, len_q, len_v = key.size(1), query.size(1), value.size(1)
           
        k=self.fc_k(key).view(key.size(0),len_k,self.num_head,self.dim_k)
        q=self.fc_q(query).view(key.size(0),len_q,self.num_head,self.dim_k)
        v=self.fc_v(value).view(key.size(0),len_v,self.num_head,self.dim_v)
        
        k,q,v=k.transpose(1,2),q.transpose(1,2),v.transpose(1,2)
        
        attn=torch.matmul(q/(self.dim_k)**0.5,k.transpose(2,3))
        attn=F.softmax(attn,dim=-1)
        
        attn_sq=attn.unsqueeze(-1) 
        v_sq_exp=v.unsqueeze(2).expand(-1,-1,len_q,-1,-1) 
        multi_attn_v=v_sq_exp.mul(attn_sq)
        emb_attn_v=self.emb_aggregator(multi_attn_v,aggre_dim=3) 
        attn_v=emb_attn_v.transpose(1,2).contiguous().view(key.size(0),len_q,-1) 
            
        return attn_v,attn
    
        
       
            
class GC_Net(nn.Module):

    def __init__(self,dim_x,dim_y,dim_emb_x,dim_lat,dim_h_lat,num_h_lat,trans_xy:bool=True,requires_grad:bool=False):
        super(GC_Net,self).__init__()

        self.dim_x=dim_x
        self.dim_y=dim_y
        self.dim_emb_x=dim_emb_x
        self.dim_lat=dim_lat
        self.dim_h_lat=dim_h_lat
        self.num_h_lat=num_h_lat
        self.trans_xy=trans_xy
        self.requires_grad=requires_grad 
        
        if self.requires_grad:
            self.beta=Parameter(torch.Tensor(1).uniform_(0,1),requires_grad=self.requires_grad).cuda() 
        else:
            self.beta=Variable(torch.ones(1),requires_grad=self.requires_grad).cuda() 
        
        self.fc_x=nn.Sequential(nn.Linear(self.dim_x, self.dim_emb_x, bias=False)).cuda() 
        
        self.trans_modules=[]
        if self.trans_xy :
            self.trans_modules.append(nn.Linear(self.dim_x+self.dim_y, self.dim_h_lat, bias=False))
        else:
            self.trans_modules.append(nn.Linear(self.dim_y, self.dim_h_lat, bias=False))
            self.trans_modules.append(nn.LayerNorm(self.dim_h_lat))
        self.trans_modules.append(nn.ReLU())
        for i in range(self.num_h_lat):
            self.trans_modules.append(nn.Linear(self.dim_h_lat, self.dim_h_lat))
            self.trans_modules.append(nn.ReLU())
        self.trans_modules.append(nn.Linear(self.dim_h_lat, self.dim_lat))

        self.trans_net=nn.Sequential(*self.trans_modules).cuda() 
            
        
    def emb_aggregator(self,h_context,aggre_dim):

        h_aggre=torch.mean(h_context,dim=aggre_dim)
        
        return h_aggre
    
    
    def forward(self,x_context,x_target,y_context,self_addition=False,whether_l2norm=True):

        assert x_context.dim()== 3
        
        x_emb_c, x_emb_t = self.fc_x(x_context), self.fc_x(x_target)

        x_emb_cn, x_emb_tn = torch.norm(x_emb_c, p=2, dim=-1, keepdim=True).detach(), torch.norm(x_emb_t, p=2, dim=-1, keepdim=True).detach()
        x_emb_c, x_emb_t = x_emb_c.div(x_emb_cn.expand_as(x_emb_c)), x_emb_t.div(x_emb_tn.expand_as(x_emb_t))
        
        b_mask=(zero_diag_mask(x_emb_c)).cuda() 
        
        c_inner_prod=self.beta*torch.matmul(x_emb_c,x_emb_c.transpose(1,2))
        c_n_prod=F.softmax(c_inner_prod,dim=-1)
        
        mask_c_n_prod=torch.mul(c_n_prod,b_mask)
        rand_mat=F.normalize(mask_c_n_prod,p=1,dim=-1) 
        b_id_mat=batch_eye_mat(x_emb_c)
        g_lap_mat=rand_mat+b_id_mat 
        
        t_inner_prod=self.beta*torch.matmul(x_emb_t,x_emb_c.transpose(1,2)) 
        t_n_prod=F.softmax(t_inner_prod,dim=-1) 
        
        if self.trans_xy:
            context_mat=torch.cat((x_context,y_context),dim=-1)
        else:
            context_mat=y_context
        emb_context=self.trans_net(context_mat) 

        if self_addition:
            c_n_prod_unsq=g_lap_mat.unsqueeze(-1) 
        else:
            c_n_prod_unsq=c_n_prod.unsqueeze(-1) 
        emb_c_unsq=emb_context.unsqueeze(1).expand(-1,context_mat.size(1),-1,-1) 
        c_message_pass=emb_c_unsq.mul(c_n_prod_unsq) 

        c_message_aggregation=c_message_pass.sum(2) 
        c2t_message_aggregation=(torch.mean(c_message_aggregation,dim=1)).unsqueeze(1).expand(-1,x_target.size(1),-1) 
        
        c_message_aggregation_unsq=c_message_aggregation.unsqueeze(1).expand(-1,x_target.size(1),-1,-1) 
        t_n_prod_unsq=t_n_prod.unsqueeze(-1) 
        t_message_pass=c_message_aggregation_unsq.mul(t_n_prod_unsq) 
        t_message_aggregation=self.emb_aggregator(t_message_pass,aggre_dim=2) 
        
        if whether_l2norm:
            c2t_message_aggregation = l2_normalization(c2t_message_aggregation)
            t_message_aggregation = l2_normalization(t_message_aggregation)
            
        return c_message_aggregation, c2t_message_aggregation, t_message_aggregation 
        

def batch_eye_mat(x):

    identity_mat=(torch.eye(x.size(1))).reshape((1,x.size(1),x.size(1)))
    b_identity_mat=identity_mat.repeat(x.size(0),1,1).cuda()
    
    return b_identity_mat        


def zero_diag_mask(x):

    identity_mat=(torch.eye(x.size(1))).reshape((1,x.size(1),x.size(1)))
    b_identity_mat=identity_mat.repeat(x.size(0),1,1)
    
    b_one_mat=torch.ones(x.size(0),x.size(1),x.size(1))
    
    b_mask=b_one_mat-b_identity_mat
    
    return b_mask
    


def l2_normalization(x_tensor):
    x_l2_norm = torch.norm(x_tensor, p=2, dim=-1, keepdim=True).detach()
    normalized_x = x_tensor.div(x_l2_norm.expand_as(x_tensor))
    
    return normalized_x

   





        
        
    

    


    



    
    
    

