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

M_linear = 1
M_model = MZ_Model_Func.model_FNN(M_nn_input, M_nn_hidden, M_nn_output,M_linear).to(device)


if M_linear == 0:
    m_state_dict = torch.load('Results/'+str(nn_zone)+'_whole_NN.pt')
elif M_linear == 1:
    m_state_dict = torch.load('Results/'+str(nn_zone)+'_whole_Lin.pt')
M_model.load_state_dict(m_state_dict)

model_para = []
for name, params in M_model.state_dict().items():
    model_para.append(to_np(params))

import time

predict_period = T_Fre*24
M_train_data,M_train_label = MZ_Model_Func.model_data(S_X_tr,S_P_tr,S_Y_tr_new, predict_period)
M_test_data,M_test_label = MZ_Model_Func.model_data(S_X_te,S_P_te,S_Y_te_new, predict_period)

c_0_price = np.loadtxt("data_2023.csv",delimiter=",",skiprows=1,usecols=range(1, 16))[::int(12/T_Fre),:][Train_s:Train_e2,9]
c_period_train,c_period_test = int((Train_e-Train_s)/predict_period),int((Train_e2-Train_s2)/predict_period)
c_period_hour = int(predict_period/T_Fre) 
c_time = predict_period + 1

c_price = np.zeros((c_period_train+c_period_test,predict_period))
for i in range(c_period_train+c_period_test):
    for j in range(predict_period):
        c_price[i,j] = c_0_price[i*predict_period+j]/1000

c_upper_p = 15*np.ones((predict_period, nn_zone-len(P_dele)))  
c_lower_p = 0*np.ones((predict_period, nn_zone-len(P_dele)))
c_PI = 3.60
c_upper_q = SS_P.transform(c_PI*c_upper_p)  
c_lower_q = SS_P.transform(c_PI*c_lower_p)
Cr_obj = np.zeros((c_period_train+c_period_test,3))
Cr_q = np.zeros((c_period_train+c_period_test,predict_period,nn_zone-len(P_dele)))
Cr_tem = np.zeros((c_period_train+c_period_test,predict_period+1,nn_zone))
Cr_tem_u = np.zeros((c_period_train+c_period_test,predict_period+1,nn_zone))

Cr_q_hy = np.zeros((Train_e2-Train_s,nn_zone-len(P_dele)))
Cr_tem_hy = np.zeros(((predict_period + 1)*(c_period_train + c_period_test),nn_zone))

Noise_Lin_obj = np.zeros((31,10,6))
HZ_Lin_P = np.zeros((31,96,80))
HZ_Lin_Tem = np.zeros((31,96,90))
HZ_Lin_Tem_diff = np.zeros((31,96,90))
HZ_Lin_obj = np.zeros((31,9))

for i_day in range(0,31):
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


    c_period_hour = int(predict_period/T_Fre)
    c_time = predict_period + 1
    M_big = 100
    
    if c_period_i < c_period_train:
        c_par_input = to_np(M_train_data[:,c_period_i,:])
    else:
        c_par_input = to_np(M_test_data[:,c_period_i-c_period_train,:])
    for i_noise in range(1,11):
        if i_noise > 0 and i_day not in [0,10,20]:
            continue
        else:
            noise = np.random.normal(0, (0.1*i_noise)**2, (96,180))
            noise = np.clip(noise, -0.9, 0.9)
            c_par_input[:,171:] = c_par_input[:,171:]*(1+noise)
            
            cons_model = []
            c_v_tem,c_v_q = cp.Variable((c_time, nn_zone)),cp.Variable((predict_period, nn_zone-len(P_dele)))
            c_v_tem_u = cp.Variable((c_time, nn_zone), nonneg=True)
            c_v_hid,c_v_hid_noact,c_v_hid_aux = [],[],[]
            for hid in range(len(M_nn_hidden)):
                c_v_hid.append(cp.Variable((predict_period, M_nn_hidden[hid])))
                c_v_hid_noact.append(cp.Variable((predict_period, M_nn_hidden[hid])))
                c_v_hid_aux.append(cp.Variable((predict_period, M_nn_hidden[hid]), boolean=True))

            for hid_t in range(predict_period):
                for hid in range(len(M_nn_hidden)):
                    if hid == 0:
                        cons_model += [c_v_hid_noact[hid][hid_t,:] == model_para[2*hid][:,:nn_zone]@c_v_tem[hid_t,:].T + model_para[2*hid][:,nn_zone:2*nn_zone-len(P_dele)]@c_v_q[hid_t,:].T \
                                    + model_para[2*hid][:,2*nn_zone-len(P_dele):]@c_par_input[hid_t,2*nn_zone-len(P_dele):].T + model_para[2*hid+1]]
                    else:
                        cons_model += [c_v_hid_noact[hid][hid_t,:] == model_para[2*hid][:,:]@c_v_hid[hid-1][hid_t,:].T + model_para[2*hid+1]]
                    if M_linear == 0:
                        cons_model += [c_v_hid[hid][hid_t,:] >= c_v_hid_noact[hid][hid_t,:]]
                        cons_model += [c_v_hid[hid][hid_t,:] >= 0]
                        cons_model += [c_v_hid[hid][hid_t,:] <= c_v_hid_noact[hid][hid_t,:] + (1-c_v_hid_aux[hid][hid_t,:])*M_big]
                        cons_model += [c_v_hid[hid][hid_t,:] <= c_v_hid_aux[hid][hid_t,:]*M_big]
                    elif M_linear == 1:
                        cons_model += [c_v_hid[hid][hid_t,:] == c_v_hid_noact[hid][hid_t,:]]
                    
                cons_model += [c_v_tem[hid_t+1,:] == model_para[-2][:,:]@c_v_hid[-1][hid_t,:].T + model_para[-1]]
                
            pho_tem = 2e-3
            c_PI = 3.60
            c_obj1 = sum(c_price[c_period_i,t]*(1/c_PI)*(1/T_Fre)*sum((SS_P.data_max_[i]-SS_P.data_min_[i])*c_v_q[t,i]+SS_P.data_min_[i] for i in range(nn_zone-len(P_dele))) for t in range(predict_period))
            c_obj2 = sum(pho_tem*(1/T_Fre)*(c_v_tem_u[t,i]**2) for i in range(nn_zone) for t in range(c_time))
            c_obj = c_obj1 + c_obj2
            cons_tem1 = [c_v_tem[t,i] <= (c_upper_tem[t,i]-SS_Y.data_min_[i])/(SS_Y.data_max_[i]-SS_Y.data_min_[i]) + c_v_tem_u[t,i]/(SS_Y.data_max_[i]-SS_Y.data_min_[i]) for i in range(nn_zone) for t in range(c_time-1)]
            cons_tem2 = [c_v_tem[t,i] >= (c_lower_tem[t,i]-SS_Y.data_min_[i])/(SS_Y.data_max_[i]-SS_Y.data_min_[i]) - c_v_tem_u[t,i]/(SS_Y.data_max_[i]-SS_Y.data_min_[i]) for i in range(nn_zone) for t in range(c_time-1)]
            cons_q = [c_v_q <= c_upper_q, c_v_q >= c_lower_q]
            cons_ini1 = [c_v_tem[0,i] == (0.5*(c_upper_tem[0,i]+c_lower_tem[0,i])-SS_Y.data_min_[i])/(SS_Y.data_max_[i]-SS_Y.data_min_[i]) for i in range(nn_zone)]
            cons = cons_tem1 + cons_tem2 + cons_q + cons_model + cons_ini1
            
            time_start = time.time()
            c_prob = cp.Problem(cp.Minimize(c_obj), cons)
            c_prob.solve(solver = cp.GUROBI,verbose=False)
            time_end = time.time()
            Cr_obj[c_period_i,:] = [c_prob.value,c_obj1.value,c_obj2.value]
            Cr_q[c_period_i,:,:] = c_v_q.value
            Cr_tem[c_period_i,:,:] = c_v_tem.value
            Cr_tem_u[c_period_i,:,:] = c_v_tem_u.value
            hid1 = c_v_hid[0].value
            Cr_q_hy[predict_period*c_period_i:predict_period*(c_period_i+1),:] = SS_P.inverse_transform(c_v_q.value)
            Cr_tem_hy[(predict_period+1)*c_period_i:(predict_period+1)*(c_period_i+1)] = SS_Y.inverse_transform(c_v_tem.value)

            tt_q = c_v_q.value
            tt_tem = c_v_tem.value
            tt_tem_hy = SS_Y.inverse_transform(tt_tem[:-1,:])
            tt_q_hy = SS_P.inverse_transform(tt_q)
            if c_period_i < c_period_train:
                tt_input = np.hstack((tt_tem[:-1,:],tt_q,to_np(M_train_data[:,c_period_i,2*nn_zone-len(P_dele):])))
            else:
                tt_input = np.hstack((tt_tem[:-1,:],tt_q,to_np(M_test_data[:,c_period_i-c_period_train,2*nn_zone-len(P_dele):])))
            tt_input_1 = tt_input[:-1,:].reshape(tt_input.shape[0]-1,1,tt_input.shape[1])
            tt_input_1 = torch.tensor(tt_input_1).to(device)
            t_linear = 0
            t_nn_input,t_nn_hidden,t_nn_output = M_train_data.shape[-1],[128,64,64],nn_zone
            t_model_NN = MZ_Model_Func.model_FNN(t_nn_input, t_nn_hidden, t_nn_output,t_linear).to(device)
            m_state_dict = torch.load('Results/'+str(nn_zone)+'_whole_NN.pt')
            t_model_NN.load_state_dict(m_state_dict)
            tt_output = t_model_NN(tt_input_1)
            tt_output_hy = SS_Y.inverse_transform(to_np(tt_output[:,0,:]))

            tt_output_err = tt_output_hy-tt_tem_hy

            def relu(x):
                if x > 0:
                    return x
                else:
                    return 0
            c_tem_ref = 0.5*(c_upper_tem+c_lower_tem)
            tt_true_cost1 = sum(c_price[c_period_i,t]*(1/c_PI)*(1/T_Fre)*sum(tt_q_hy[t,:]) for t in range(predict_period))
            tt_true_cost2 = sum(pho_tem*(1/T_Fre)*(relu(tt_output_hy[t,i] - c_upper_tem[t,i])**2+relu(c_lower_tem[t,i] - tt_output_hy[t,i])**2) for i in range(nn_zone) for t in range(c_time-1))
            print(tt_true_cost1)
            print(tt_true_cost2)
            if i_noise == 0:
                HZ_Lin_P[i_day,:,:] = tt_q_hy[:,:]
                HZ_Lin_Tem[i_day,:,:] = tt_output_hy[:,:]
                HZ_Lin_Tem_diff[i_day,:,:] = tt_output_hy[:,:] - tt_tem_hy[:,:]
                HZ_Lin_obj[i_day,:] = Cr_obj[c_period_i,1],Cr_obj[c_period_i,2],Cr_obj[c_period_i,0],\
                    tt_true_cost1,tt_true_cost2,  tt_true_cost1+tt_true_cost2,  0,0,time_end-time_start
            else:
                Noise_Lin_obj[i_day,i_noise-1,:] = Cr_obj[c_period_i,1],Cr_obj[c_period_i,2],Cr_obj[c_period_i,0],\
                    tt_true_cost1,tt_true_cost2,  tt_true_cost1+tt_true_cost2
