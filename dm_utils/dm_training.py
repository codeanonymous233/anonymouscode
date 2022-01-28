'''
This file is to define the meta_dm training/testing process.
'''

import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

from dm_utils.data_collection import multi_env_normalize_dataset, multi_env_denormalize_dataset, normalize_dataset, denormalize_dataset
from dm_utils.meta_loss import mse_loss, mse_kl_np_loss



##########################################################################################################################
    #Training dynamics models in single task and multi-task settings
##########################################################################################################################


def train_meta_dm(meta_dm, stats, dm_optim, trainloader, 
                  check_lvm, loss_fun, epoch, whether_norm=True):

    meta_dm.train()
    
    list_train_loss, list_train_mse = [], [] 

    for i in range(epoch):
        running_tr_loss=[]
        running_tr_mse=[]
        
        for batch_id, data in enumerate(trainloader): 
            x, y = data 
            x, y = x.unsqueeze(0).cuda(), y.unsqueeze(0).cuda()
            
            if whether_norm:
                x, y = normalize_dataset(stats, x, var_order=0), normalize_dataset(stats, y, var_order=3)
            
            dm_optim.zero_grad()
    
            if check_lvm == 'NP_DM' :
                idx = meta_dm.get_context_idx(x.size(1))
                x_memo_c = meta_dm.idx_to_data(x,sample_dim=1,idx=idx)
                y_memo_c = meta_dm.idx_to_data(y,sample_dim=1,idx=idx)             
                mu_c,logvar_c,mu_t,logvar_t,y_pred = meta_dm(x_memo_c,y_memo_c,x,y,x)
                loss, b_avg_mse, kld = loss_fun(y_pred,y,mu_c,logvar_c,mu_t,logvar_t)
                
            elif check_lvm == 'AttnNP_DM' :
                idx = meta_dm.get_context_idx(x.size(1))
                x_memo_c = meta_dm.idx_to_data(x,sample_dim=1,idx=idx)
                y_memo_c = meta_dm.idx_to_data(y,sample_dim=1,idx=idx)             
                mu_c,logvar_c,mu_t,logvar_t,y_pred = meta_dm(x_memo_c,y_memo_c,x,y,x)
                loss, b_avg_mse, kld=loss_fun(y_pred,y,mu_c,logvar_c,mu_t,logvar_t)    
                
            elif check_lvm == 'GS_DM' :
                idx=meta_dm.get_context_idx(x.size(1))
                x_memo_c=meta_dm.idx_to_data(x,sample_dim=1,idx=idx)
                y_memo_c=meta_dm.idx_to_data(y,sample_dim=1,idx=idx) 
                
                idx_all=np.arange(x.size(1)).tolist()
                idx_t=torch.tensor(list(set(idx_all)-set(idx))).cuda()
                x_t=meta_dm.idx_to_data(x,sample_dim=1,idx=idx_t)
                y_t=meta_dm.idx_to_data(y,sample_dim=1,idx=idx_t)
                
                mu,logvar,mu_c,logvar_c,y_pred=meta_dm(x_memo_c,y_memo_c,x_t)
                loss, b_avg_mse, kld=loss_fun(y_pred,y_t,mu,logvar,mu_c,logvar_c,beta=1.0) # 0.5
                
            else:
                raise NotImplementedError()
    
            loss.backward()
            
            dm_optim.step()
    
            running_tr_loss.append(loss.data) 
            running_tr_mse.append(b_avg_mse.data)
    
        batch_train_loss = torch.mean(torch.tensor(running_tr_loss))
        batch_train_mse = torch.mean(torch.tensor(running_tr_mse))
        list_train_loss.append(batch_train_loss)
        list_train_mse.append(batch_train_mse)

    return np.array(list_train_loss), np.array(list_train_mse)



def multi_env_train_meta_dm(meta_dm, stats_list, dm_optim, multi_env_trainloader, 
                            check_lvm, loss_fun, epoch, whether_norm=True):

    meta_dm.train()
    
    list_train_loss, list_train_mse = [], [] 

    for i in range(epoch):
        running_tr_loss=[]
        running_tr_mse=[]
        
        for batch_id, data in enumerate(multi_env_trainloader): 
            
            x, y = data 
            
            x, y = x.permute(1, 0, 2).contiguous().cuda(), y.permute(1, 0, 2).contiguous().cuda() 
            
            if whether_norm:
                x, y = multi_env_normalize_dataset(stats_list, x, var_order=0), multi_env_normalize_dataset(stats_list, y, var_order=3)
            
            dm_optim.zero_grad()
                
            if check_lvm == 'NP_DM' :
                idx = meta_dm.get_context_idx(x.size(1))
                x_memo_c = meta_dm.idx_to_data(x,sample_dim=1,idx=idx)
                y_memo_c = meta_dm.idx_to_data(y,sample_dim=1,idx=idx)             
                mu_c,logvar_c,mu_t,logvar_t,y_pred = meta_dm(x_memo_c,y_memo_c,x,y,x)
                loss, b_avg_mse, kld = loss_fun(y_pred,y,mu_c,logvar_c,mu_t,logvar_t)
                
            elif check_lvm == 'AttnNP_DM' :
                idx = meta_dm.get_context_idx(x.size(1))
                x_memo_c = meta_dm.idx_to_data(x,sample_dim=1,idx=idx)
                y_memo_c = meta_dm.idx_to_data(y,sample_dim=1,idx=idx)             
                mu_c,logvar_c,mu_t,logvar_t,y_pred = meta_dm(x_memo_c,y_memo_c,x,y,x)
                loss, b_avg_mse, kld=loss_fun(y_pred,y,mu_c,logvar_c,mu_t,logvar_t)    
                
            elif check_lvm == 'GS_DM' :
                idx=meta_dm.get_context_idx(x.size(1))
                x_memo_c=meta_dm.idx_to_data(x,sample_dim=1,idx=idx)
                y_memo_c=meta_dm.idx_to_data(y,sample_dim=1,idx=idx) 
                
                idx_all=np.arange(x.size(1)).tolist()
                idx_t=torch.tensor(list(set(idx_all)-set(idx))).cuda()
                x_t=meta_dm.idx_to_data(x,sample_dim=1,idx=idx_t)
                y_t=meta_dm.idx_to_data(y,sample_dim=1,idx=idx_t)
                
                mu,logvar,mu_c,logvar_c,y_pred=meta_dm(x_memo_c,y_memo_c,x_t)
                loss, b_avg_mse, kld=loss_fun(y_pred,y_t,mu,logvar,mu_c,logvar_c,beta=0.5) #1.0
                
            else:
                raise NotImplementedError()
    
            loss.backward()
            
            dm_optim.step()
    
            running_tr_loss.append(loss.data) 
            running_tr_mse.append(b_avg_mse.data)
    
        batch_train_loss = torch.mean(torch.tensor(running_tr_loss))
        batch_train_mse = torch.mean(torch.tensor(running_tr_mse))
        list_train_loss.append(batch_train_loss)
        list_train_mse.append(batch_train_mse)
        

    return np.array(list_train_loss), np.array(list_train_mse)



##########################################################################################################################
    #Testing dynamics models in single task and multi-task settings
##########################################################################################################################
    
    
def test_meta_dm(meta_dm, x_memo_c, y_memo_c, stats, testloader,
                 check_lvm, loss_fun, dm_optim = None, whether_norm = True):

    meta_dm.eval()
    
    running_te_mse=[]
    
    with torch.no_grad():
        for batch_id, data in enumerate(testloader): 

            x, y = data
            x, y = x.unsqueeze(0).cuda(), y.unsqueeze(0).cuda()
            
            if whether_norm:
                x_norm, y_norm = normalize_dataset(stats, x, var_order=0), normalize_dataset(stats, y, var_order=3)
                x_memo_c_norm, y_memo_c_norm = normalize_dataset(stats, x_memo_c, var_order=0), \
                    normalize_dataset(stats, y_memo_c, var_order=3)
                
            if check_lvm == 'NP_DM' :           
                mu_c,logvar_c,mu_t,logvar_t,y_pred=meta_dm(x_memo_c_norm,y_memo_c_norm,x_norm,y_norm,x_norm)
                if whether_norm:
                    y_pred = denormalize_dataset(stats, y_pred, var_order=3)
                    y = denormalize_dataset(stats, y_norm, var_order=3)                
                b_avg_mse=loss_fun(y_pred,y)
                
            elif check_lvm == 'AttnNP_DM' :          
                mu_c,logvar_c,mu_t,logvar_t,y_pred=meta_dm(x_memo_c_norm,y_memo_c_norm,x_norm,y_norm,x_norm)
                if whether_norm:
                    y_pred = denormalize_dataset(stats, y_pred, var_order=3)
                    y = denormalize_dataset(stats, y_norm, var_order=3)                
                b_avg_mse=loss_fun(y_pred,y) 
                
            elif check_lvm == 'GS_DM' :
                mu,logvar,mu_c,logvar_c,y_pred=meta_dm(x_memo_c_norm,y_memo_c_norm,x_norm)
                if whether_norm:
                    y_pred = denormalize_dataset(stats, y_pred, var_order=3)
                    y = denormalize_dataset(stats, y_norm, var_order=3)                
                b_avg_mse=loss_fun(y_pred,y)
                
            else:
                raise NotImplementedError()
    
            running_te_mse.append(b_avg_mse.data)

    batch_test_mse = torch.mean(torch.tensor(running_te_mse))

    return np.array(batch_test_mse)




def multi_env_test_meta_dm(meta_dm, x_memo_c, y_memo_c, stats_list, testloader,
                           check_lvm, loss_fun, dm_optim = None, whether_norm = True):
    meta_dm.eval()
    
    running_te_mse=[]
    
    with torch.no_grad():
        for batch_id, data in enumerate(testloader): 

            x, y = data
            x, y = x.permute(1, 0, 2).cuda(), y.permute(1, 0, 2).cuda()
            
            if whether_norm:
                x_norm, y_norm = multi_env_normalize_dataset(stats_list, x, var_order=0), multi_env_normalize_dataset(stats_list, y, var_order=3)
                x_memo_c_norm, y_memo_c_norm = multi_env_normalize_dataset(stats_list, x_memo_c, var_order=0), \
                    multi_env_normalize_dataset(stats_list, y_memo_c, var_order=3)
                
            if check_lvm == 'NP_DM' :          
                mu_c,logvar_c,mu_t,logvar_t,y_pred=meta_dm(x_memo_c_norm,y_memo_c_norm,x_norm,y_norm,x_norm)
                if whether_norm:
                    y_pred = multi_env_denormalize_dataset(stats_list, y_pred, var_order=3)
                    y = multi_env_denormalize_dataset(stats_list, y_norm, var_order=3)                
                b_avg_mse=loss_fun(y_pred,y)
                
            elif check_lvm == 'AttnNP_DM' :          
                mu_c,logvar_c,mu_t,logvar_t,y_pred=meta_dm(x_memo_c_norm,y_memo_c_norm,x_norm,y_norm,x_norm)
                if whether_norm:
                    y_pred = multi_env_denormalize_dataset(stats_list, y_pred, var_order=3)
                    y = multi_env_denormalize_dataset(stats_list, y_norm, var_order=3)                
                b_avg_mse=loss_fun(y_pred,y) 
                
            elif check_lvm == 'GS_DM' :
                mu,logvar,mu_t,logvar_t,y_pred=meta_dm(x_memo_c_norm,y_memo_c_norm,x_norm)
                if whether_norm:
                    y_pred = multi_env_denormalize_dataset(stats_list, y_pred, var_order=3)
                    y = multi_env_denormalize_dataset(stats_list, y_norm, var_order=3)                
                b_avg_mse=loss_fun(y_pred,y)
                
            else:
                raise NotImplementedError()
    
            running_te_mse.append(b_avg_mse.data)  

    batch_test_mse = torch.mean(torch.tensor(running_te_mse))

    return np.array(batch_test_mse)




