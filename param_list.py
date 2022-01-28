import argparse


env_name='half_cheetah'

if env_name=='half_cheetah':
    # dim_obs=18, dim_action=6, obs=[-inf,inf], action=[-1,+1]
    dim_x=24
    dim_y=19
    dim_lat=8 # for gssm 8 self-addition = False
    num_h=5

    

#######################Meta Dynamics Models' Parameters#######################

def GSDM_Param():
    parser_gsdm = argparse.ArgumentParser(description='Parameters for GSDM')
    
    parser_gsdm.add_argument('--dim_x', type=int, default=dim_x, metavar='N',
                             help='dimension of state_action pair as input for dm')
    parser_gsdm.add_argument('--dim_y', type=int, default=dim_y, metavar='N',
                             help='dimension of state difference as output for dm')
    parser_gsdm.add_argument('--dim_emb_x', type=int, default=32, metavar='N',
                             help='dimension of embedding of x for attention module in GS_Net')
    parser_gsdm.add_argument('--dim_lat', type=int, default=dim_lat, metavar='N',
                             help='dimension of z, the latent variable for dm and pn')
    parser_gsdm.add_argument('--dim_h_lat', type=int, default=32, metavar='N',
                             help='dimension of hidden units in transforming latent variable')
    parser_gsdm.add_argument('--num_h_lat', type=int, default=2, metavar='N',
                             help='number of transfoming layers for latent variable')
    parser_gsdm.add_argument('--dim_h', type=int, default=200, metavar='N',
                             help='dimension of hidden layers of decoder')
    parser_gsdm.add_argument('--num_h', type=int, default=num_h, metavar='N',
                             help='number of hidden layers of decoder')
    parser_gsdm.add_argument('--act_type', type=str, default='ReLU', metavar='N',
                             help='activation unit type')
    parser_gsdm.add_argument('--amort_y', type=bool, default=False, metavar='N',
                             help='whether to amortize output distribution')
    
    args_gsdm = parser_gsdm.parse_args()
    
    return args_gsdm



def NPDM_Param():
    parser_npdm = argparse.ArgumentParser(description='Parameters for NPDM')
    
    parser_npdm.add_argument('--dim_x', type=int, default=dim_x, metavar='N',
                             help='dimension of input of dm')
    parser_npdm.add_argument('--dim_y', type=int, default=dim_y, metavar='N',
                             help='dimension of state difference as output for dm')
    parser_npdm.add_argument('--dim_lat', type=int, default=dim_lat, metavar='N',
                             help='dimension of z, the latent variable for dm and pn')
    parser_npdm.add_argument('--dim_h_lat', type=int, default=32, metavar='N',
                             help='dimension of hidden units in transforming latent variable')
    parser_npdm.add_argument('--num_h_lat', type=int, default=2, metavar='N',
                             help='number of transfoming layers for latent variable')
    parser_npdm.add_argument('--dim_h', type=int, default=200, metavar='N',
                             help='dimension of hidden layers of decoder')
    parser_npdm.add_argument('--num_h', type=int, default=num_h, metavar='N',
                             help='number of hidden layers of decoder')
    parser_npdm.add_argument('--act_type', type=str, default='ReLU', metavar='N',
                             help='activation unit type')
    parser_npdm.add_argument('--amort_y', type=bool, default=False, metavar='N',
                             help='whether to amortize output distribution')
    
    args_npdm = parser_npdm.parse_args()
    
    return args_npdm



def AttnNPDM_Param():
    parser_attnnpdm = argparse.ArgumentParser(description='Parameters for AttnNPDM')
    
    parser_attnnpdm.add_argument('--dim_x', type=int, default=dim_x, metavar='N',
                                 help='dimension of input of dm')
    parser_attnnpdm.add_argument('--dim_y', type=int, default=dim_y, metavar='N',
                                 help='dimension of state difference as output for dm')
    parser_attnnpdm.add_argument('--dim_emb_x', type=int, default=32, metavar='N',
                                 help='dimension of embedding of x')
    parser_attnnpdm.add_argument('--dim_lat', type=int, default=dim_lat, metavar='N',
                                 help='dimension of z, the latent variable for dm and pn')
    parser_attnnpdm.add_argument('--dim_h_lat', type=int, default=32, metavar='N',
                                 help='dimension of hidden units in transforming latent variable')
    parser_attnnpdm.add_argument('--num_h_lat', type=int, default=2, metavar='N',
                                 help='number of transfoming layers for latent variable')
    parser_attnnpdm.add_argument('--num_head', type=int, default=1, metavar='N',
                                 help='number of heads in attention networks')
    parser_attnnpdm.add_argument('--dim_h', type=int, default=200, metavar='N',
                                 help='dimension of hidden layers of decoder')
    parser_attnnpdm.add_argument('--num_h', type=int, default=num_h, metavar='N',
                                 help='number of hidden layers of decoder')
    parser_attnnpdm.add_argument('--act_type', type=str, default='ReLU', metavar='N',
                                 help='activation unit type')
    parser_attnnpdm.add_argument('--amort_y', type=bool, default=False, metavar='N',
                                 help='whether to amortize output distribution')
    
    args_attnnpdm = parser_attnnpdm.parse_args()
    
    return args_attnnpdm
