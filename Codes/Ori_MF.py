#%%

import numpy as np
import pandas as pd
import os
import random
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
if M_linear == 0:
    m_state_dict = torch.load('Results/'+str(nn_zone)+'_whole_NN.pt')
elif M_linear == 1:
    m_state_dict = torch.load('Results/'+str(nn_zone)+'_whole_Lin.pt')
M_model.load_state_dict(m_state_dict)

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
Cr_time = np.zeros((c_period_train+c_period_test,1))
Cr_q = np.zeros((c_period_train+c_period_test,predict_period,nn_zone-len(P_dele)))
Cr_tem = np.zeros((c_period_train+c_period_test,predict_period+1,nn_zone))
Cr_tem_u = np.zeros((c_period_train+c_period_test,predict_period+1,nn_zone))

Obj_sum = np.zeros((31,4))

Noise_NNzero_obj = np.zeros((31,10,6))
HZ_NNzero_P = np.zeros((31,96,80))
HZ_NNzero_Tem = np.zeros((31,96,90))
HZ_NNzero_Tem_diff = np.zeros((31,96,90))
HZ_NNzero_obj = np.zeros((31,9))

for i_day in range(0,31):  #31):
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

    opt_period = T_Fre*24
    if c_period_i < c_period_train:
        Z_tr_ini = Z_tr_new[opt_period*c_period_i:opt_period*(c_period_i+1),:]
    else:
        opt_day_ind_up = c_period_i - c_period_train
        Z_tr_ini = Z_te_new[opt_period*opt_day_ind_up:opt_period*(opt_day_ind_up+1),:]
    Z_tr_ini[0,:nn_zone] = 0.5*(c_upper_tem[0,:]+c_lower_tem[0,:])
    S_Z_tr_ini_noise = SS_Z.transform(Z_tr_ini)
    S_Z_tr_ini = SS_Z.transform(Z_tr_ini)
    
    for i_noise in range(0,11):  #11
        if i_noise > 0 and i_day not in [0,10,20]:
            continue
        else:
            noise = np.random.normal(0, (0.1*i_noise)**2, (96,180))
            noise = np.clip(noise, -0.9, 0.9)
            S_Z_tr_ini_noise[:,171:] = S_Z_tr_ini_noise[:,171:]*(1+noise)
            w_pow = 10
            c_PI,pho_pow,pho_tem = 3.6, 1, 2e-3
            ten_Y_max,ten_Y_min,ten_P_max,ten_P_min = torch.tensor([SS_Y.data_max_[i] for i in range(90)]),torch.tensor([SS_Y.data_min_[i] for i in range(90)]),torch.tensor([SS_P.data_max_[i] for i in range(80)]),torch.tensor([SS_P.data_min_[i] for i in range(80)])
            P_price = np.zeros((96,80))
            for i in range(95):
                P_price[i,:] = pho_pow*(1/T_Fre)*(1/c_PI)*c_price[c_period_i,i]
            P_price[95,:] = 0
            P_price = torch.tensor(P_price).to(device)
            wei_Y = torch.sqrt(torch.tensor(pho_tem/T_Fre)).to(device)
            wei_P = torch.sqrt(torch.tensor(w_pow)).to(device)
            Upp_Y,Low_Y = wei_Y*torch.tensor(c_upper_tem[:96,:90]).to(device),wei_Y*torch.tensor(c_lower_tem[:96,:90]).to(device)
            Upp_P,Low_P = wei_P*torch.tensor(c_PI*c_upper_p[:96,:80]).to(device),wei_P*torch.tensor(c_PI*c_lower_p[:96,:80]).to(device)
            
            M_obj = MZ_Model_Func.Ori_NN_obj_layer(M_nn_input, M_nn_hidden, M_nn_output,M_linear,m_state_dict,ten_Y_max,ten_Y_min,ten_P_max,ten_P_min,wei_Y,wei_P)

            for param in M_obj.parameters():
                param.requires_grad = False

            S_Z_tr_ini_ts = torch.tensor(S_Z_tr_ini_noise, dtype=torch.float).to(device)
            S_Z_tr_ini_ts[:,90:170] = 0.01
            Iters = 2000
            Iter_obj = np.zeros((Iters,4))
            Iter_obj = pd.DataFrame(Iter_obj,columns = ['Obj.pow','Obj.tem','Power_pen','Sum_Obj'])
            Iter_obj_zero = np.zeros((Iters,1))
            Iter_sens_grd = torch.zeros((Iters,opt_period,nn_zone-len(P_dele))).to(device)
            Iter_J_tr_en = torch.zeros((Iters,opt_period,S_Z_tr_ini_ts.shape[-1])).to(device)
            for i in range(Iters):
                Iter_J_tr_en[i,:,:] = S_Z_tr_ini_ts.clone().detach()

            Iter_lr = 1e-3
            Iter_cond = 1e-3
            Iter_count = 0
            Iter_incre = 0
            Time_obj = []
            Time_grd = []
            u_direction = np.random.uniform(0, 0.5, (96, 80, Iters))
            u_r, u_d = 0.1, 1.0
            two_points = 'yes'
            a_momentum = 0.2
            for iter in range(0,Iters):
                u_k = u_direction[:,:,iter]
                if iter == 0:
                    Iter_J_tr_en[iter,:,:] = S_Z_tr_ini_ts.clone().detach()
                elif iter == 1:
                    Iter_J_tr_en[iter,:,nn_zone:2*nn_zone-len(P_dele)] = Iter_J_tr_en[iter-1,:,nn_zone:2*nn_zone-len(P_dele)] - Iter_lr*Iter_sens_grd[iter-1,:,:]
                elif iter >= 2:
                    Iter_J_tr_en[iter,:,nn_zone:2*nn_zone-len(P_dele)] = Iter_J_tr_en[iter-1,:,nn_zone:2*nn_zone-len(P_dele)] - Iter_lr*Iter_sens_grd[iter-1,:,:] \
                                    + a_momentum*(Iter_J_tr_en[iter-1,:,nn_zone:2*nn_zone-len(P_dele)] - Iter_J_tr_en[iter-2,:,nn_zone:2*nn_zone-len(P_dele)])
                
                J_tr_en = to_np(Iter_J_tr_en[iter,:,:])
                J_tr_en = torch.tensor(J_tr_en,dtype=torch.float).to(device)
                J_tr_en.requires_grad_(True)
                
                time1 = time.time()
                Obj_Y_upp,Obj_Y_low,Obj_P_1,Obj_P_2upp,Obj_P_2low = \
                    M_obj(J_tr_en,P_price,Upp_Y,Low_Y,Upp_P,Low_P)
                J_t_sum1 = torch.sum(Obj_P_1)
                J_t_sum2 = torch.sum(Obj_Y_upp**2) + torch.sum(Obj_Y_low**2)
                J_t_sum_pen = torch.sum(Obj_P_2upp[:-1,:]**2) + torch.sum(Obj_P_2low[:-1,:]**2)
                J_t_sum = J_t_sum1+J_t_sum2+J_t_sum_pen
                time2 = time.time()
                J_tr_en_u = to_np(Iter_J_tr_en[iter,:,:])
                J_tr_en_u[:,90:170] += u_r*u_k
                J_tr_en_u = torch.tensor(J_tr_en_u,dtype=torch.float).to(device)
                J_tr_en_u.requires_grad_(True)
                Obj_Y_upp,Obj_Y_low,Obj_P_1,Obj_P_2upp,Obj_P_2low = \
                    M_obj(J_tr_en_u,P_price,Upp_Y,Low_Y,Upp_P,Low_P)
                J_t_sum1 = torch.sum(Obj_P_1)
                J_t_sum2 = torch.sum(Obj_Y_upp**2) + torch.sum(Obj_Y_low**2)
                J_t_sum_pen = torch.sum(Obj_P_2upp[:-1,:]**2) + torch.sum(Obj_P_2low[:-1,:]**2)
                J_t_sum_u = J_t_sum1+J_t_sum2+J_t_sum_pen
                if two_points == 'yes':
                    Iter_sens_grd[iter,:,:] = u_d*(1/u_r)*(J_t_sum_u-J_t_sum)*torch.tensor(u_k).to(device)
                time3 = time.time()
                Iter_obj.iloc[iter,:] = [to_np(J_t_sum1),to_np(J_t_sum2),to_np(J_t_sum_pen), to_np(J_t_sum)]
                Iter_obj_zero[iter,:] = [to_np(J_t_sum_u)]
                
                Time_obj.append(time2-time1)
                Time_grd.append(time3-time2)
                if iter > 0 and Iter_obj.iloc[iter,-1] > Iter_obj.iloc[iter-1,-1]:
                    print('!!!!')
                    Iter_incre += 1
                else:
                    Iter_incre = 0
                if Iter_incre >= 2:
                    print('Reduce step_size!')
                
                if iter > 0 and abs(Iter_obj.iloc[iter,-1] - Iter_obj.iloc[iter-1,-1]) <= Iter_cond:
                    Iter_count += 1
                else:
                    Iter_count = 0
                if Iter_count >= 5:
                    break
                if iter % 1 == 0:
                    print('Train Iteration: {} \t Obj_sum: {:.6f}'.format(iter,Iter_obj.iloc[iter,-1]))
            
            print("Time:", sum(Time_obj)/len(Time_obj), sum(Time_grd)/len(Time_grd))

            plt.plot(Iter_obj.iloc[:iter,-1])
            plt.show()

            i_opt = iter
            SS_Z_min,SS_Z_max = [SS_Z.data_min_[i] for i in range(Z_tr.shape[-1])],[SS_Z.data_max_[i] for i in range(Z_tr.shape[-1])]

            YY_en = Iter_J_tr_en[i_opt,:,:]
            
            Ori = YY_en.reshape(YY_en.shape[0],1,YY_en.shape[1])
            Ori_out = to_np(M_model(Ori[:-1,:,:]))
            Ori_out = Ori_out.reshape((Ori_out.shape[0]*Ori_out.shape[1], Ori_out.shape[2]), order='F')

            Ori_out = SS_Y.inverse_transform(Ori_out)
            for ii in range(96):
                for jj in range(80):
                    if YY_en[ii,90+jj] >= (3.6*15-SS_Z_min[90+jj])/(SS_Z_max[90+jj]-SS_Z_min[90+jj]):
                        YY_en[ii,90+jj] = (3.6*15-SS_Z_min[90+jj])/(SS_Z_max[90+jj]-SS_Z_min[90+jj])
                    elif YY_en[ii,90+jj] <= (0-SS_Z_min[90+jj])/(SS_Z_max[90+jj]-SS_Z_min[90+jj]):
                        YY_en[ii,90+jj] = (0-SS_Z_min[90+jj])/(SS_Z_max[90+jj]-SS_Z_min[90+jj])

            J_tr_en_re = YY_en.reshape(YY_en.shape[0],1,YY_en.shape[1])
            J_tr_en_re_ts = torch.tensor(S_Z_tr_ini, dtype=torch.float).to(device)
            J_tr_en_re[:,0,:170] = YY_en[:,:170]
            J_tr_en_re[:,0,170:] = J_tr_en_re_ts[:,170:]
            J_tr_en_out = to_np(M_model(J_tr_en_re[:-1,:,:]))
            J_tr_en_out = J_tr_en_out.reshape((J_tr_en_out.shape[0]*J_tr_en_out.shape[1], J_tr_en_out.shape[2]), order='F')

            J_tr_de_s_hy = SS_Y.inverse_transform(J_tr_en_out)
            J_tr_en_hy = SS_Z.inverse_transform(to_np(YY_en))
            J_t_obj1 = sum(c_price[c_period_i,t]*(1/T_Fre)*(1/c_PI)*sum(J_tr_en_hy[t,nn_zone:2*nn_zone-len(P_dele)]) for t in range(opt_period-1))
            J_t_obj2 = pho_tem*(1/T_Fre)*sum(sum(relu(J_tr_de_s_hy[t,i]-c_upper_tem[t,i])**2+relu(c_lower_tem[t,i] - J_tr_de_s_hy[t,i])**2 for i in range(nn_zone)) for t in range(opt_period))

            print(J_t_obj1)
            print(J_t_obj2)
            print(Iter_obj.iloc[i_opt,0])
            print(Iter_obj.iloc[i_opt,1])
            if i_noise == 0:
                HZ_NNzero_P[i_day,:,:] = J_tr_en_hy[:,nn_zone:2*nn_zone-len(P_dele)]
                HZ_NNzero_Tem[i_day,:,:] = J_tr_de_s_hy[:,:]
                HZ_NNzero_Tem_diff[i_day,:,:] = J_tr_de_s_hy[:,:] - Ori_out[:,:]
                HZ_NNzero_obj[i_day,:] = Iter_obj.iloc[i_opt,0],Iter_obj.iloc[i_opt,1],  Iter_obj.iloc[i_opt,0]+Iter_obj.iloc[i_opt,1],\
                    J_t_obj1,J_t_obj2,  J_t_obj1+J_t_obj2,  iter,sum(Time_obj)/len(Time_obj), sum(Time_grd)/len(Time_grd)
            else:
                Noise_NNzero_obj[i_day,i_noise-1,:] = Iter_obj.iloc[i_opt,0],Iter_obj.iloc[i_opt,1],  Iter_obj.iloc[i_opt,0]+Iter_obj.iloc[i_opt,1],\
                    J_t_obj1,J_t_obj2,  J_t_obj1+J_t_obj2
            if i_noise == 0 and i_day == 20:
                np.save('MZ_results/HZ_NNzero_loss.npy',Iter_obj.iloc[:i_opt+1,-1].to_numpy())

np.save('MZ_results/HZ_NNzero_P.npy',HZ_NNzero_P)
np.save('MZ_results/HZ_NNzero_Tem.npy',HZ_NNzero_Tem)
np.save('MZ_results/HZ_NNzero_Tem_diff.npy',HZ_NNzero_Tem_diff)
np.save('MZ_results/HZ_NNzero_obj.npy',HZ_NNzero_obj)
np.save('MZ_results/Noise_NNzero_obj.npy',Noise_NNzero_obj)
