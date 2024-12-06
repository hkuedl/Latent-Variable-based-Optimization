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
#P_dele = [0]
SS_Z,S_Z_tr,_,S_Z_te,Z_tr,_,Z_te \
    = MZ_Model_Func.data_input(T_Fre,Train_s,Train_e,Train_s2,Train_e2,Train_s3,Train_e3,nn_zone,P_dele)

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

M_linear = 0
M_model = MZ_Model_Func.model_FNN(M_nn_input, M_nn_hidden, M_nn_output,M_linear).to(device)


# #%%  读取模型参数
if M_linear == 0:
    m_state_dict = torch.load('Results/'+str(nn_zone)+'_whole_NN.pt')
elif M_linear == 1:
    m_state_dict = torch.load('Results/'+str(nn_zone)+'_whole_Lin.pt')
M_model.load_state_dict(m_state_dict)

model_para = []
for name, params in M_model.state_dict().items():
    model_para.append(to_np(params))

#% 优化
import time
predict_period = T_Fre*24
M_train_data,M_train_label = MZ_Model_Func.model_data(S_X_tr,S_P_tr,S_Y_tr_new, predict_period)
M_test_data,M_test_label = MZ_Model_Func.model_data(S_X_te,S_P_te,S_Y_te_new, predict_period)

c_0_price = np.loadtxt("/mnt/ExtraDisk/cxy/Building_MPC/Data_price/data_2023.csv",delimiter=",",skiprows=1,usecols=range(1, 16))[::int(12/T_Fre),:][Train_s:Train_e2,9]
c_period_train,c_period_test = int((Train_e-Train_s)/predict_period),int((Train_e2-Train_s2)/predict_period)

c_price = np.zeros((c_period_train+c_period_test,predict_period))
for i in range(c_period_train+c_period_test):
    for j in range(predict_period):
        c_price[i,j] = c_0_price[i*predict_period+j]/1000
c_upper_p = 15*np.ones((predict_period, nn_zone-len(P_dele)))
c_lower_p = 0*np.ones((predict_period, nn_zone-len(P_dele)))
c_PI = 3.60 

#%% 

HZ_NN_P = np.zeros((31,96,80))
HZ_NN_Tem = np.zeros((31,96,90))
HZ_NN_Tem_diff = np.zeros((31,96,90))
HZ_NN_obj = np.zeros((31,9))

for i_day in range(0,1):
    c_period_i = c_period_train + i_day
    opt_period = T_Fre*24
    c_0_tem_comfort = np.zeros((opt_period,))
    c_0_tem_comfort[:24],c_0_tem_comfort[24:48],c_0_tem_comfort[48:72],c_0_tem_comfort[72:] \
        = 0.2,0.3,0.5,0.2
    c_upper_tem,c_lower_tem = np.zeros((opt_period,S_Y_tr.shape[1])),np.zeros((opt_period,S_Y_tr.shape[1]))
    c_0_tem_ref = np.zeros((opt_period,S_Y_tr.shape[1]))
    for j in range(c_upper_tem.shape[1]):
        if c_period_i < c_period_train:
            tem_base = np.round(Y_tr_new[:,j][c_period_i*opt_period:(c_period_i+1)*opt_period],5)
        else:
            c_period_i_up = c_period_i - c_period_train
            tem_base = np.round(Y_te_new[:,j][c_period_i_up*opt_period:(c_period_i_up+1)*opt_period],5)
        c_upper_tem[:,j] = tem_base+c_0_tem_comfort
        c_lower_tem[:,j] = tem_base-c_0_tem_comfort
        c_0_tem_ref[:,j] = tem_base

    def relu(x):
        if x <= 0:
            return 0
        else:
            return x
    def Grad_Obj_cal(N_zone, SS_Y,SS_Z,M_model_deter,J_tr_en,c_upper_tem,c_lower_tem,c_upper_p,c_lower_p,c_price,opt_period,c_period_i):
        J_tr_en_re = J_tr_en.reshape(J_tr_en.shape[0],1,J_tr_en.shape[1])
        w_pow = 1
        pho_tem = 1e-3
        c_PI = 3.6
        time1 = time.time()    
        J_tr_en_out = to_np(M_model_deter(J_tr_en_re[:-1,:,:]))
        J_tr_en_out = J_tr_en_out.reshape((J_tr_en_out.shape[0]*J_tr_en_out.shape[1], J_tr_en_out.shape[2]), order='F')
        J_tr_de_s_hy = SS_Y.inverse_transform(J_tr_en_out)
        J_tr_en_hy = SS_Z.inverse_transform(to_np(J_tr_en))
        J_t_obj1 = sum(c_price[c_period_i,t]*(1/T_Fre)*(1/c_PI)*sum(J_tr_en_hy[t,N_zone:2*N_zone-len(P_dele)]) for t in range(opt_period-1))
        J_t_obj2 = pho_tem*(1/T_Fre)*sum(sum(relu(J_tr_de_s_hy[t,i]-c_upper_tem[t,i])**2+relu(c_lower_tem[t,i]-J_tr_de_s_hy[t,i])**2 for i in range(N_zone)) for t in range(opt_period))
        
        J_t_pen_tem = 0
        J_t_pen_pow = w_pow*sum(relu(J_tr_en_hy[t,N_zone+i]-c_PI*c_upper_p[t,i])**2 + relu(c_PI*c_lower_p[t,i]-J_tr_en_hy[t,N_zone+i])**2 for t in range(opt_period-1) for i in range(N_zone-len(P_dele)))
        J_t_sum = 1*(J_t_obj1 + J_t_obj2 + J_t_pen_tem + J_t_pen_pow)
        time2 = time.time()
        
        Sen_sens_grd = np.zeros((opt_period,J_tr_en.shape[-1]))
        for fea in range(N_zone-len(P_dele)):
            for t in range(opt_period-1):
                if abs(0.01*J_tr_en[t,N_zone+fea]) > 0:
                    delta_e = abs(0.01*J_tr_en[t,N_zone+fea])
                else:
                    delta_e = 0.001
                J_tr_en_new = J_tr_en.clone().detach()
                J_tr_en_new[t,N_zone+fea] = J_tr_en[t,N_zone+fea] + delta_e
                J_tr_en_new_re = J_tr_en_new.reshape(J_tr_en.shape[0],1,J_tr_en.shape[1])
                J_tr_en_out_new = to_np(M_model_deter(J_tr_en_new_re[:-1,:,:]))
                J_tr_en_out_new = J_tr_en_out_new.reshape((J_tr_en_out_new.shape[0]*J_tr_en_out_new.shape[1], J_tr_en_out_new.shape[2]), order='F')
                J_tr_en_new_hy = SS_Z.inverse_transform(to_np(J_tr_en_new))
                J_tr_en_out_new_hy = SS_Y.inverse_transform(J_tr_en_out_new)
                J_i_obj1 = (1/T_Fre)*(1/c_PI)*sum(c_price[c_period_i,t]*sum(J_tr_en_new_hy[t,N_zone:2*N_zone-len(P_dele)]) for t in range(opt_period-1))
                J_i_obj2 = pho_tem*(1/T_Fre)*sum(sum(relu(J_tr_en_out_new_hy[t,i]-c_upper_tem[t,i])**2+relu(c_lower_tem[t,i]-J_tr_en_out_new_hy[t,i])**2 for i in range(N_zone)) for t in range(opt_period))
                J_i_pen_tem = 0
                J_i_pen_pow = w_pow*sum(relu(J_tr_en_new_hy[t,N_zone+i]-c_PI*c_upper_p[t,i])**2 + relu(c_PI*c_lower_p[t,i]-J_tr_en_new_hy[t,N_zone+i])**2 for t in range(opt_period-1) for i in range(N_zone-len(P_dele)))
                Sen_cost_sum = J_i_obj1+J_i_obj2 + J_i_pen_tem + J_i_pen_pow
                Sen_sens_grd[t,N_zone+fea] = (Sen_cost_sum-J_t_sum)/(1*delta_e)                    
        time3 = time.time()
        return [J_t_obj1,J_t_obj2, J_t_pen_pow, J_t_sum], torch.tensor(Sen_sens_grd, dtype=torch.float).to(device), time2-time1, time3-time2

    if c_period_i < c_period_train:
        Z_tr_ini = Z_tr_new[opt_period*c_period_i:opt_period*(c_period_i+1),:]
    else:
        opt_day_ind_up = c_period_i - c_period_train
        Z_tr_ini = Z_te_new[opt_period*opt_day_ind_up:opt_period*(opt_day_ind_up+1),:]
    Z_tr_ini[0,:nn_zone] = 0.5*(c_upper_tem[0,:]+c_lower_tem[0,:])
    ini_P = np.load('Results/HZ_AE_P.npy')
    Z_tr_ini[:,nn_zone:2*nn_zone-len(P_dele)] = ini_P[i_day,:,:]
    S_Z_tr_ini = SS_Z.transform(Z_tr_ini)
    S_Z_tr_ini_ts = torch.tensor(S_Z_tr_ini, dtype=torch.float).to(device)

    Iters = 50
    Iter_obj = np.zeros((Iters,4))
    Iter_obj = pd.DataFrame(Iter_obj,columns = ['Obj.pow','Obj.tem','Power_pen','Sum_Obj'])
    Iter_sens_grd = torch.zeros((Iters,opt_period,S_Z_tr_ini_ts.shape[-1])).to(device)
    Iter_J_tr_en = torch.zeros((Iters,opt_period,S_Z_tr_ini_ts.shape[-1])).to(device)

    Iter_lr = 1e-4
    Iter_cond = 1e-3
    Iter_count = 0
    Iter_incre = 0
    Time_obj = []
    Time_grd = []
    for iter in range(0,Iters):
        if iter == 0:
            Iter_J_tr_en[iter,:,:] = S_Z_tr_ini_ts.clone().detach()
        else:
            Iter_J_tr_en[iter,:,:] = Iter_J_tr_en[iter-1,:,:] - Iter_lr*Iter_sens_grd[iter-1,:,:]
        
        Iter_obj.iloc[iter,:],Iter_sens_grd[iter,:,:],time1,time2 = Grad_Obj_cal(nn_zone, SS_Y,SS_Z,M_model,Iter_J_tr_en[iter,:,:],c_upper_tem,c_lower_tem,c_upper_p,c_lower_p,c_price,opt_period,c_period_i)
        Time_obj.append(time1)
        Time_grd.append(time2)

        if iter > 0 and Iter_obj.iloc[iter,-1] > Iter_obj.iloc[iter-1,-1]:
            print('!!!!')
            Iter_incre += 1
        else:
            Iter_incre = 0
        if Iter_incre >= 2:
            print('Reduce step_size!')
            Iter_lr = 0.5*Iter_lr
        
        if iter > 0 and abs(Iter_obj.iloc[iter,-1] - Iter_obj.iloc[iter-1,-1]) <= Iter_cond:
            Iter_count += 1
        else:
            Iter_count = 0
        if Iter_count >= 3:
            break
        if iter % 1 == 0:
            print('Train Iteration: {} \t Obj_sum: {:.6f}'.format(iter,Iter_obj.iloc[iter,-1]))
    
    i_opt = iter
    SS_Z_min,SS_Z_max = [SS_Z.data_min_[i] for i in range(Z_tr.shape[-1])],[SS_Z.data_max_[i] for i in range(Z_tr.shape[-1])]
    YY_en = Iter_J_tr_en[i_opt,:,:]
    for ii in range(96):
        for jj in range(80):
            if YY_en[ii,90+jj] >= (3.6*15-SS_Z_min[90+jj])/(SS_Z_max[90+jj]-SS_Z_min[90+jj]):
                YY_en[ii,90+jj] = (3.6*15-SS_Z_min[90+jj])/(SS_Z_max[90+jj]-SS_Z_min[90+jj])
            elif YY_en[ii,90+jj] <= (0-SS_Z_min[90+jj])/(SS_Z_max[90+jj]-SS_Z_min[90+jj]):
                YY_en[ii,90+jj] = (0-SS_Z_min[90+jj])/(SS_Z_max[90+jj]-SS_Z_min[90+jj])
    J_tr_en_re = YY_en.reshape(YY_en.shape[0],1,YY_en.shape[1])
    J_tr_en_out = to_np(M_model(J_tr_en_re[:-1,:,:]))
    J_tr_en_out = J_tr_en_out.reshape((J_tr_en_out.shape[0]*J_tr_en_out.shape[1], J_tr_en_out.shape[2]), order='F')
    J_tr_de_s_hy = SS_Y.inverse_transform(J_tr_en_out) 
    J_tr_en_hy = SS_Z.inverse_transform(to_np(YY_en))
    pho_tem = 1e-3
    c_PI = 3.6
    J_t_obj1 = sum(c_price[c_period_i,t]*(1/T_Fre)*(1/c_PI)*sum(J_tr_en_hy[t,nn_zone:2*nn_zone-len(P_dele)]) for t in range(opt_period-1))
    J_t_obj2 = pho_tem*(1/T_Fre)*sum(sum(relu(J_tr_de_s_hy[t,i]-c_upper_tem[t,i])**2+relu(c_lower_tem[t,i] - J_tr_de_s_hy[t,i])**2 for i in range(nn_zone)) for t in range(opt_period))

    print(J_t_obj1)
    print(J_t_obj2)
    print(J_t_obj1+J_t_obj2)
    
    HZ_NN_P[i_day,:,:] = J_tr_en_hy[:,nn_zone:2*nn_zone-len(P_dele)]
    HZ_NN_Tem[i_day,:,:] = J_tr_de_s_hy[:,:]
    HZ_NN_obj[i_day,:] = Iter_obj.iloc[i_opt,0],Iter_obj.iloc[i_opt,1],  Iter_obj.iloc[i_opt,0]+Iter_obj.iloc[i_opt,1],\
        J_t_obj1,J_t_obj2,  J_t_obj1+J_t_obj2,  iter,sum(Time_obj)/len(Time_obj), sum(Time_grd)/len(Time_grd)

np.save('MZ_results/HZ_NNnum_P.npy',HZ_NN_P)
np.save('MZ_results/HZ_NNnum_Tem.npy',HZ_NN_Tem)
np.save('MZ_results/HZ_NNnum_obj.npy',HZ_NN_obj)
