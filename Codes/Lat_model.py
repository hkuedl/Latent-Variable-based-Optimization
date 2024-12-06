#%%
import numpy as np
import math
import pandas as pd
import os
import random
import copy
import torch

def set_seed(seed):
    # seed init
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # torch seed init.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False # train speed is slower after enabling this opts.

    # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

    # avoiding nondeterministic algorithms (see https://pytorch.org/docs/stable/notes/randomness.html)
    torch.use_deterministic_algorithms(True)

set_seed(20)

import torch.nn as nn
import torch.optim as optim
device = torch.device("cuda:3")
from sklearn.preprocessing import MinMaxScaler
import cvxpy as cp
import MZ_Model_Func
import matplotlib.pyplot as plt

def to_np(x):
    return x.cpu().detach().numpy()

nn_zone = 90
T_Fre = 4
Train_s = T_Fre*24*(31+28+31+30+31)
Train_e = Train_s + T_Fre*24*(30+31)
Train_s2 = Train_e
Train_e2 = Train_s2 + T_Fre*24*31
Train_s3 = Train_e
Train_e3 = Train_s3 + T_Fre*24*31

P_dele = [8,17,26,35,44,53,62,71,80,89]
#P_dele = [0]
SS_Z,S_Z_tr,_,S_Z_te,Z_tr,_,Z_te \
    = MZ_Model_Func.data_input(T_Fre,Train_s,Train_e,Train_s2,Train_e2,Train_s3,Train_e3,nn_zone,P_dele,'old')

S_X_tr,S_X_te,X_tr,X_te = S_Z_tr[:,2*nn_zone-len(P_dele):],S_Z_te[:,2*nn_zone-len(P_dele):],Z_tr[:,2*nn_zone-len(P_dele):],Z_te[:,2*nn_zone-len(P_dele):]
S_P_tr,S_P_te,P_tr,P_te = S_Z_tr[:,nn_zone:2*nn_zone-len(P_dele)],S_Z_te[:,nn_zone:2*nn_zone-len(P_dele)],Z_tr[:,nn_zone:2*nn_zone-len(P_dele)],Z_te[:,nn_zone:2*nn_zone-len(P_dele)]
S_Y_tr,S_Y_te,Y_tr,Y_te = S_Z_tr[:,:nn_zone],S_Z_te[:,:nn_zone],Z_tr[:,:nn_zone],Z_te[:,:nn_zone]
SS_X,SS_P,SS_Y = MinMaxScaler().fit(X_tr), MinMaxScaler().fit(P_tr), MinMaxScaler().fit(Y_tr)

predict_period = T_Fre*24
M_train_data,M_train_label = MZ_Model_Func.model_data(S_X_tr,S_P_tr,S_Y_tr, predict_period)
M_test_data,M_test_label = MZ_Model_Func.model_data(S_X_te,S_P_te,S_Y_te, predict_period)

M_nn_input,M_nn_hidden,M_nn_output = M_train_data.shape[-1],[128,64,64],nn_zone
M_model0 = MZ_Model_Func.model_FNN(M_nn_input, M_nn_hidden, M_nn_output,0).to(device)
m_state_dict = torch.load('Results/'+str(nn_zone)+'_whole_NN.pt')
M_model0.load_state_dict(m_state_dict)

M_test_pred = to_np(M_model0(M_test_data[:-1,:,:]))
S_Y_te_new  = M_test_pred.reshape((M_test_pred.shape[0]*M_test_pred.shape[1], M_test_pred.shape[2]), order='F')
Y_te_new  = SS_Y.inverse_transform(S_Y_te_new)
M_train_pred = to_np(M_model0(M_train_data[:-1,:,:]))
S_Y_tr_new = M_train_pred.reshape((M_train_pred.shape[0]*M_train_pred.shape[1], M_train_pred.shape[2]), order='F')
Y_tr_new = SS_Y.inverse_transform(S_Y_tr_new)
Z_tr_new,Z_te_new = np.hstack((Y_tr_new,Z_tr[:,nn_zone:])),np.hstack((Y_te_new,Z_te[:,nn_zone:]))
S_Z_tr_new,S_Z_te_new = np.hstack((S_Y_tr_new,S_Z_tr[:,nn_zone:])),np.hstack((S_Y_te_new,S_Z_te[:,nn_zone:]))

#%
def Jointacc(M_model,S_Y_te_ts,S_P_te_ts,S_X_te_ts,SS_Y,SS_P,SS_X):
    [_,_,_,de_x0_te,de_x1_te,de_x2_te,_,u_y_de_te] = M_model(S_Y_te_ts,S_P_te_ts,S_X_te_ts)
    M_x0_te_acc = MZ_Model_Func.ACC(SS_Y, to_np(de_x0_te),to_np(S_Y_te_ts))
    M_x1_te_acc = MZ_Model_Func.ACC(SS_P, to_np(de_x1_te),to_np(S_P_te_ts))
    M_x2_te_acc = MZ_Model_Func.ACC(SS_X, to_np(de_x2_te),to_np(S_X_te_ts))
    M_mm_te_acc = MZ_Model_Func.ACC(SS_Y, to_np(u_y_de_te),to_np(S_Y_te_ts))
    M_x0_te_acc_avg = [sum(M_x0_te_acc[i])/len(M_x0_te_acc[i]) for i in range(3)]
    M_x1_te_acc_avg = [sum(M_x1_te_acc[i])/len(M_x1_te_acc[i]) for i in range(3)]
    M_x2_te_acc_avg = [sum(M_x2_te_acc[i])/len(M_x2_te_acc[i]) for i in range(3)]
    M_mm_te_acc_avg = [sum(M_mm_te_acc[i])/len(M_mm_te_acc[i]) for i in range(3)]
    M_mm_te_acc_std = [np.std(np.array(M_mm_te_acc[i])) for i in range(3)]
    return M_x0_te_acc_avg,M_x1_te_acc_avg,M_x2_te_acc_avg,M_mm_te_acc_avg,M_mm_te_acc_std
predict_period_m = T_Fre*24
N_sam_m = int(S_Y_tr_new.shape[0]/predict_period_m)

ae_input_num,ae_hidden_units,ae_output_num,mm_hidden_units = \
    [S_Y_tr_new.shape[1],S_P_tr.shape[1],S_X_tr.shape[1]], [[64,32,16],[64,32,16],[128,64,32]], [3,4,6], [32,16,16]

m_state_dict_Y,m_state_dict_P,m_state_dict_X = torch.load('Results/90_full_AE_ind_Y.pt'),torch.load('Results/90_full_AE_ind_P.pt'),torch.load('Results/90_full_AE_ind_X.pt')
m_state_dict_model = torch.load('Results/'+str(nn_zone)+'_full_Model_ind_Y.pt')
m_state_dict_model_lin = torch.load('Results/'+str(nn_zone)+'_full_Model_ind_Y_lin.pt')
m_state_dict_model_hui = [m_state_dict_model,m_state_dict_model_lin]

S_Y_ts = torch.tensor(S_Y_tr_new, dtype=torch.float).to(device)
#S_Y_ts = torch.tensor(S_Y_tr, dtype=torch.float).to(device)
S_P_ts = torch.tensor(S_P_tr, dtype=torch.float).to(device)
S_X_ts = torch.tensor(S_X_tr, dtype=torch.float).to(device)
S_Y_te_ts = torch.tensor(S_Y_te_new, dtype=torch.float).to(device)
#S_Y_te_ts = torch.tensor(S_Y_te, dtype=torch.float).to(device)
S_P_te_ts = torch.tensor(S_P_te, dtype=torch.float).to(device)
S_X_te_ts = torch.tensor(S_X_te, dtype=torch.float).to(device)

Acc_AE = np.zeros((2,3,18))
seeds = [10,20,30,40,50,60]
epochs = [2000,1000]  #[2000,3000,4000]

Joint_model_linear = 1
aa_ini,mm_ini = 'True','True'   #'False'
for i_seed in range(1,2):
    for i_epoch in range(0,1):
        set_seed(seeds[i_seed])
        M_epoch = epochs[i_epoch]
        M_epoch_freq = int(M_epoch/10)
        M_model = MZ_Model_Func.model_joint(ae_input_num,ae_hidden_units,ae_output_num,mm_hidden_units,Joint_model_linear,aa_ini,mm_ini,[m_state_dict_Y,m_state_dict_P,m_state_dict_X],m_state_dict_model_hui[Joint_model_linear]).to(device)
        #m_state_dict_ = torch.load('Results/'+str(nn_zone)+'_full_Model_joint_lin.pt')
        #M_model.load_state_dict(m_state_dict_)
        
        M_batch_size = N_sam_m
        M_batch_num = math.ceil(N_sam_m/M_batch_size)
        M_optimizer = optim.Adam(M_model.parameters(),lr = 1e-3)
        M_lossfn = nn.MSELoss()
        for m in range(M_epoch):
            for num in range(M_batch_num):
                M_optimizer.zero_grad()
                en_x0,en_x1,en_x2,de_x0,de_x1,de_x2,u_y,u_y_de = M_model(S_Y_ts,S_P_ts,S_X_ts)
                en_x0.to(device)
                en_x1.to(device)
                en_x2.to(device)
                de_x0.to(device)
                de_x1.to(device)
                de_x2.to(device)
                M_loss_ae = M_lossfn(de_x0,S_Y_ts) + M_lossfn(de_x1,S_P_ts) + M_lossfn(de_x2,S_X_ts)
                M_loss_mm = M_lossfn(u_y_de,S_Y_ts)
                M_loss = M_loss_mm + 0.2*M_loss_ae
                M_loss.backward()
                M_optimizer.step()
                
            if m % M_epoch_freq == 0:
                with torch.no_grad():
                    print('Train Epoch: {} \tLoss_ae: {:.6f}\tLoss_mm: {:.6f}'.format(m, M_loss_ae.item(), M_loss_mm.item()))

        _,_,_,M_mm_tr_acc_avg,M_mm_tr_acc_std = Jointacc(M_model,S_Y_ts,S_P_ts,S_X_ts,SS_Y,SS_P,SS_X)
        _,_,_,M_mm_te_acc_avg,M_mm_te_acc_std = Jointacc(M_model,S_Y_te_ts,S_P_te_ts,S_X_te_ts,SS_Y,SS_P,SS_X)

        print([M_mm_tr_acc_avg[0],M_mm_tr_acc_std[0]])
        print([M_mm_te_acc_avg[0],M_mm_te_acc_std[0]])
        print([M_mm_tr_acc_avg[1],M_mm_tr_acc_std[1]])
        print([M_mm_te_acc_avg[1],M_mm_te_acc_std[1]])
        print([M_mm_tr_acc_avg[2],M_mm_tr_acc_std[2]])
        print([M_mm_te_acc_avg[2],M_mm_te_acc_std[2]])
        Acc_AE[0,:,i_seed*3+i_epoch] = M_mm_tr_acc_avg[:]
        Acc_AE[1,:,i_seed*3+i_epoch] = M_mm_te_acc_avg[:]
