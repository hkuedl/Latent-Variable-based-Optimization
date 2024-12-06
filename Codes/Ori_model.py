#%%
import numpy as np
import math
import pandas as pd
import os
import random
import copy
import torch

def set_seed(seed):
    # seed init.
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
from sklearn.decomposition import PCA

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

M_linear = 0
M_model0 = MZ_Model_Func.model_FNN(M_nn_input, M_nn_hidden, M_nn_output,M_linear).to(device)
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

predict_period = T_Fre*24
M_train_data,M_train_label = MZ_Model_Func.model_data(S_X_tr,S_P_tr,S_Y_tr_new, predict_period)
M_test_data,M_test_label = MZ_Model_Func.model_data(S_X_te,S_P_te,S_Y_te_new, predict_period)

Acc_NN = np.zeros((2,3,18))
Acc_Lin = np.zeros((2,3,18))
seeds = [10,20,30,40,50,70]  #60]
epochs = [4000,5000,6000]
M_nn_hidden_nnn = [32,16,16]
for M_linear in range(1,2):  #2):
    for i_seed in range(1,2):  #(6):
        for i_epoch in range(1,2):  #3):
            print([M_linear,i_seed,i_epoch])
            set_seed(seeds[i_seed])
            M_epoch = 15000 #epochs[i_epoch]
            M_epoch_freq = int(M_epoch/10)
            M_model = MZ_Model_Func.model_FNN(M_nn_input, M_nn_hidden, M_nn_output,M_linear).to(device)
            #m_state_dict = torch.load('Results/'+str(nn_zone)+'_whole_Lin.pt')
            #M_model.load_state_dict(m_state_dict)
            
            M_batch_size = M_train_data.shape[1]
            M_batch_num = math.ceil(M_train_data.shape[1]/M_batch_size)
            M_optimizer = optim.Adam(M_model.parameters(),lr = 1e-3)
            M_lossfn = nn.MSELoss()
            for m in range(M_epoch):
                M_batch_list = list(range(M_train_data.shape[1]))
                for num in range(M_batch_num):
                    M_batch_size_i = min(M_batch_size,len(M_batch_list))
                    M_batch_list_i = random.sample(M_batch_list,M_batch_size_i)
                    M_batch_list = [x for x in M_batch_list if x not in M_batch_list_i]
                    M_optimizer.zero_grad()
                    M_batch_data = M_train_data[:-1,M_batch_list_i,:]
                    M_batch_label = M_train_label[:,M_batch_list_i]
                    M_batch_pred = M_model(M_batch_data)
                    M_batch_pred.to(device)
                    M_loss = M_lossfn(M_batch_pred,M_batch_label)
                    M_loss.backward()
                    M_optimizer.step()
                if m % M_epoch_freq == 0:
                    with torch.no_grad():
                        print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(m, (num+1) * M_batch_size, M_train_data.shape[1], M_loss.item()))
            M_test_pred = to_np(M_model(M_test_data[:-1,:,:]))
            M_test_pred_1 = M_test_pred.reshape((M_test_pred.shape[0]*M_test_pred.shape[1], M_test_pred.shape[2]), order='F')
            M_te_rmse,M_te_mae,M_te_r2 = MZ_Model_Func.ACC(SS_Y,M_test_pred_1,S_Y_te_new)
            M_te_rmse_avg,M_te_mae_avg,M_te_r2_avg = sum(M_te_rmse)/len(M_te_rmse),sum(M_te_mae)/len(M_te_mae),sum(M_te_r2)/len(M_te_r2)
            print([M_te_rmse_avg,np.std(np.array(M_te_rmse))])
            print([M_te_mae_avg,np.std(np.array(M_te_mae))])
            print([M_te_r2_avg,np.std(np.array(M_te_r2))])
            
            M_train_pred = to_np(M_model(M_train_data[:-1,:,:]))
            M_train_pred_1 = M_train_pred.reshape((M_train_pred.shape[0]*M_train_pred.shape[1], M_train_pred.shape[2]), order='F')
            M_tr_rmse,M_tr_mae,M_tr_r2 = MZ_Model_Func.ACC(SS_Y,M_train_pred_1,S_Y_tr_new)
            M_tr_rmse_avg,M_tr_mae_avg,M_tr_r2_avg = sum(M_tr_rmse)/len(M_tr_rmse),sum(M_tr_mae)/len(M_tr_mae),sum(M_tr_r2)/len(M_tr_r2)
            print([M_tr_rmse_avg,np.std(np.array(M_tr_rmse))])
            print([M_tr_mae_avg,np.std(np.array(M_tr_mae))])
            print([M_tr_r2_avg,np.std(np.array(M_tr_r2))])            
            
            
            if M_linear == 0:
                Acc_NN[0,:,i_seed*3+i_epoch] = [M_tr_rmse_avg,M_tr_mae_avg,M_tr_r2_avg]
                Acc_NN[1,:,i_seed*3+i_epoch] = [M_te_rmse_avg,M_te_mae_avg,M_te_r2_avg]
            elif M_linear == 1:
                Acc_Lin[0,:,i_seed*3+i_epoch] = [M_tr_rmse_avg,M_tr_mae_avg,M_tr_r2_avg]
                Acc_Lin[1,:,i_seed*3+i_epoch] = [M_te_rmse_avg,M_te_mae_avg,M_te_r2_avg]
            # np.save('MZ_results/Acc_NN_1.npy',Acc_NN)
            # np.save('MZ_results/Acc_Lin_1.npy',Acc_Lin)

#torch.save(M_model.state_dict(), 'Results/'+str(nn_zone)+'_whole_Lin.pt')
