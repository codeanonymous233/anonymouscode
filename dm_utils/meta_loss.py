'''
This file is about loss functions.
'''

import torch
import torch.nn as nn
import math
from torch.nn import functional as F
from torch.distributions import Normal



def kl_div_gaussians(mu_q, logvar_q, mu_p, logvar_p):

    var_p = torch.exp(logvar_p)
    kl_div = (torch.exp(logvar_q) + (mu_q - mu_p) ** 2) / var_p \
        - 1.0 + logvar_p - logvar_q
    
    if mu_q.dim()==2:
        kl_div = (0.5 * torch.mean(kl_div,dim=0)).sum() 
    elif mu_q.dim()==3:
        kl_div = (0.5 * torch.mean(kl_div,dim=[0,1])).sum() 
    
    return kl_div



def kl_div_prior(z_mu,logvar):

    kld_sum=-0.5*torch.sum(1+logvar-z_mu.pow(2)-logvar.exp())
    if z_mu.dim()==2:
        kld=(torch.mean(kld_sum,dim=0)).sum() 
    elif z_mu.dim()==3:
        kld=(torch.mean(kld_sum,dim=[0,1])).sum() 
    
    return kld



def mse_loss(y_pred,y):
    loss=F.mse_loss(y_pred,y)
    
    return loss


def mse_kl_loss(y_pred, y, mu_q, logvar_q, mu_p, logvar_p, beta=1.):

    b_avg_mse=F.mse_loss(y_pred, y)
    kld=kl_div_gaussians(mu_q, logvar_q, mu_p, logvar_p)
    loss = b_avg_mse + beta * kld
    
    return loss,b_avg_mse,kld


def mse_kl_zeroprior_loss(y_pred, y, z_mu, logvar, beta=1.):

    b_avg_mse=F.mse_loss(y_pred, y)
    kld=kl_div_prior(z_mu, logvar)
    loss = b_avg_mse + beta * kld
    
    return loss, b_avg_mse, kld


def mse_kl_np_loss(y_pred, y, mu_q, logvar_q, mu_p, logvar_p, beta=1.):
    
    b_avg_mse=F.mse_loss(y_pred, y)
    kld=kl_div_gaussians(mu_q, logvar_q, mu_p, logvar_p)
    loss = b_avg_mse + beta * kld
    
    return loss, b_avg_mse, kld



def get_dm_loss(check_lvm):
    
    if check_lvm == 'NP_DM':
        return mse_kl_loss
    elif check_lvm == 'AttnNP_DM':
        return mse_kl_loss
    elif check_lvm == 'GS_DM':
        return mse_kl_loss
    
    
    


