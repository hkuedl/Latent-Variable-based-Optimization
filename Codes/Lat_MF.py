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
M_model = MZ_Model_Func.model_FNN(M_nn_input, M_nn_hidden, M_nn_output,M_linear).to(device)
m_state_dict = torch.load('Results/'+str(nn_zone)+'_whole_NN.pt')
M_model.load_state_dict(m_state_dict)

M_test_pred = to_np(M_model(M_test_data[:-1,:,:]))
S_Y_te_new  = M_test_pred.reshape((M_test_pred.shape[0]*M_test_pred.shape[1], M_test_pred.shape[2]), order='F')
Y_te_new  = SS_Y.inverse_transform(S_Y_te_new)
M_train_pred = to_np(M_model(M_train_data[:-1,:,:]))
S_Y_tr_new = M_train_pred.reshape((M_train_pred.shape[0]*M_train_pred.shape[1], M_train_pred.shape[2]), order='F')
Y_tr_new = SS_Y.inverse_transform(S_Y_tr_new)
Z_tr_new,Z_te_new = np.hstack((Y_tr_new,Z_tr[:,nn_zone:])),np.hstack((Y_te_new,Z_te[:,nn_zone:]))
S_Z_tr_new,S_Z_te_new = np.hstack((S_Y_tr_new,S_Z_tr[:,nn_zone:])),np.hstack((S_Y_te_new,S_Z_te[:,nn_zone:]))

def AE_testing(J_model_sta_1,S_Y_tr,S_Y_te,SS_Y):
    S_Y_tr_1 = torch.tensor(S_Y_tr, dtype=torch.float).to(device)
    J_tr_en_sta,J_tr_de_sta = J_model_sta_1(S_Y_tr_1)
    J_tr_de_sta = to_np(J_tr_de_sta)
    ERR_rmse_tr,ERR_mae_tr,ERR_r2_tr = MZ_Model_Func.ACC(SS_Y,J_tr_de_sta,S_Y_tr)
    
    S_Y_te_1 = torch.tensor(S_Y_te, dtype=torch.float).to(device)
    J_te_en_sta,J_te_de_sta = J_model_sta_1(S_Y_te_1)
    J_te_de_sta = to_np(J_te_de_sta)
    ERR_rmse_te,ERR_mae_te,ERR_r2_te = MZ_Model_Func.ACC(SS_Y,J_te_de_sta,S_Y_te)
    return J_tr_en_sta,J_te_en_sta,[ERR_rmse_tr,ERR_mae_tr,ERR_r2_tr],\
            [ERR_rmse_te,ERR_mae_te,ERR_r2_te]

def relu(x):
    if x <= 0:
        return 0
    else:
        return x

def model_output(model,en_x0,en_x1,en_x2):
    N_sample = int((en_x0.shape[0])/(24*4))
    u_y = torch.zeros(en_x0.shape[0], en_x0.shape[1], dtype=torch.float).to(device)
    for i in range(N_sample):
        u_y[i*96,:] = en_x0[i*96,:]
    for j in range(96-1):
        input_new = torch.zeros(N_sample, en_x0.shape[1]+en_x1.shape[1]+en_x2.shape[1]).to(device)
        for tt in range(N_sample):
            input_new[tt,:en_x0.shape[1]] = u_y[tt*96+j,:]
            input_new[tt,en_x0.shape[1]:(en_x0.shape[1]+en_x1.shape[1])] = en_x1[tt*96+j,:]
            input_new[tt,(en_x0.shape[1]+en_x1.shape[1]):(en_x0.shape[1]+en_x1.shape[1]+en_x2.shape[1])] = en_x2[tt*96+j,:]
        u_out = model.mm(input_new)
        for tt in range(N_sample):
            u_y[tt*96+j+1,:] = u_out[tt,:]
    u_y_de = model.encoder0.decoder(u_y)
    return u_y_de
def u_obj(J_tr_en_P,M_model_deter,J_tr_de_Y,M_model,S_Z_tr_ini,SS_Y,SS_P):
    J_tr_de_P = M_model_deter.encoder1.decoder(torch.tensor(J_tr_en_P).to(device)).to(device)
    Black_de = np.hstack((to_np(J_tr_de_Y),to_np(J_tr_de_P),S_Z_tr_ini[:,170:]))
    Black_de = Black_de.reshape(Black_de.shape[0],1,Black_de.shape[1])
    J_tr_en_out = to_np(M_model(torch.tensor(Black_de[:-1,:,:]).to(device)))
    J_tr_en_out = J_tr_en_out.reshape((J_tr_en_out.shape[0]*J_tr_en_out.shape[1], J_tr_en_out.shape[2]), order='F')
    J_tr_de_s_hy = SS_Y.inverse_transform(J_tr_en_out)
    J_tr_de_p_hy = SS_P.inverse_transform(to_np(J_tr_de_P))
    return J_tr_de_s_hy,J_tr_de_p_hy

#%%  看敏感度
import time
opt_period = 24*T_Fre
c_0_price = np.loadtxt("/mnt/ExtraDisk/cxy/Building_MPC/Data_price/data_2023.csv",delimiter=",",skiprows=1,usecols=range(1, 16))[::int(12/T_Fre),:][Train_s:Train_e2,9]

c_period_train,c_period_test = int((Train_e-Train_s)/opt_period),int((Train_e2-Train_s2)/opt_period)

Noise_AE_obj = np.zeros((31,10,6))
HZ_AE_P = np.zeros((31,96,80))
HZ_AE_Tem = np.zeros((31,96,90))
HZ_AE_Tem_diff = np.zeros((31,96,90))
HZ_AE_obj = np.zeros((31,9))

for i_day in range(0,1):
    c_period_i = c_period_train + i_day

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

    c_price = np.zeros((c_period_train+c_period_test,opt_period))
    for i in range(c_period_train+c_period_test):
        for j in range(opt_period):
            c_price[i,j] = c_0_price[i*opt_period+j]/1000

    c_upper_p = 15*np.ones((opt_period,S_P_tr.shape[1]))
    c_lower_p = 0*np.ones((opt_period,S_P_tr.shape[1]))

    opt_day_ind = c_period_i
    opt_start = T_Fre*0
    opt_period = T_Fre*24
    # load model
    ae_input_num,ae_hidden_units,ae_output_num,mm_hidden_units = \
        [S_Y_tr_new.shape[1],S_P_tr.shape[1],S_X_tr.shape[1]], [[64,32,16],[64,32,16],[128,64,32]], [3,4,6], [32,16,16]
    m_state_dict_Y,m_state_dict_P,m_state_dict_X = torch.load('Results/90_full_AE_ind_Y.pt'),torch.load('Results/90_full_AE_ind_P.pt'),torch.load('Results/90_full_AE_ind_X.pt')
    m_state_dict_model = torch.load('Results/'+str(nn_zone)+'_full_Model_ind_Y.pt')
    M_model_deter = MZ_Model_Func.model_joint(ae_input_num,ae_hidden_units,ae_output_num,mm_hidden_units,0,'False','False',[m_state_dict_Y,m_state_dict_P,m_state_dict_X],m_state_dict_model).to(device)
    state_dict_joint = torch.load('Results/'+str(nn_zone)+'_full_Model_joint.pt')
    M_model_deter.load_state_dict(state_dict_joint)
    if opt_day_ind < c_period_train:
        S_Z_tr_ini = S_Z_tr_new[opt_day_ind*96+opt_start:opt_day_ind*96+opt_start+opt_period,:]
    else:
        opt_day_ind_up = opt_day_ind - c_period_train
        S_Z_tr_ini = S_Z_te_new[opt_day_ind_up*96+opt_start:opt_day_ind_up*96+opt_start+opt_period,:]
        S_Z_tr_ini_noise = S_Z_te_new[opt_day_ind_up*96+opt_start:opt_day_ind_up*96+opt_start+opt_period,:]
        
    for i_noise in range(0,1):  #11
        if i_noise > 0 and i_day not in [0,10,20]:
            continue
        else:
            noise = np.random.normal(0, (0.1*i_noise)**2, (96,180))
            noise = np.clip(noise, -0.9, 0.9)
            S_Z_tr_ini_noise[:,171:] = S_Z_tr_ini_noise[:,171:]*(1+noise)
            S_Z_tr_ini_ts = torch.tensor(S_Z_tr_ini_noise, dtype=torch.float).to(device)
            [J_tr_en_Y_ini,J_tr_en_P_ini,J_tr_en_X_ini,_,_,_,_,_] = M_model_deter(S_Z_tr_ini_ts[:,:nn_zone],S_Z_tr_ini_ts[:,nn_zone:2*nn_zone-len(P_dele)],S_Z_tr_ini_ts[:,2*nn_zone-len(P_dele):])
            J_tr_en_P_ini[:,:] = 0.01   #Set a better initial point
            J_tr_en_ini = torch.cat((J_tr_en_Y_ini,J_tr_en_P_ini,J_tr_en_X_ini),dim = 1)
            
            w_pow = 1
            c_PI,pho_pow,pho_tem = 3.6, 1, 2e-3
            ten_Y_max,ten_Y_min,ten_P_max,ten_P_min = torch.tensor([SS_Y.data_max_[i] for i in range(90)]),torch.tensor([SS_Y.data_min_[i] for i in range(90)]),torch.tensor([SS_P.data_max_[i] for i in range(80)]),torch.tensor([SS_P.data_min_[i] for i in range(80)])
            P_price = np.zeros((96,80))  #,dtype=torch.float32)
            for i in range(95):
                P_price[i,:] = pho_pow*(1/T_Fre)*(1/c_PI)*c_price[opt_day_ind,opt_start+i]
            P_price[95,:] = 0
            P_price = torch.tensor(P_price).to(device)
            wei_Y = torch.sqrt(torch.tensor(pho_tem/T_Fre)).to(device)
            wei_P = torch.sqrt(torch.tensor(w_pow)).to(device)
            Upp_Y,Low_Y = wei_Y*torch.tensor(c_upper_tem[opt_start:opt_start+96,:90]).to(device),wei_Y*torch.tensor(c_lower_tem[opt_start:opt_start+96,:90]).to(device)
            Upp_P,Low_P = wei_P*torch.tensor(c_PI*c_upper_p[opt_start:opt_start+96,:80]).to(device),wei_P*torch.tensor(c_PI*c_lower_p[opt_start:opt_start+96,:80]).to(device)
            M_obj = MZ_Model_Func.NN_obj_layer(ae_input_num,ae_hidden_units,ae_output_num,mm_hidden_units,[m_state_dict_Y,m_state_dict_P,m_state_dict_X],m_state_dict_model,state_dict_joint,ten_Y_max,ten_Y_min,ten_P_max,ten_P_min,wei_Y,wei_P)
            for param in M_obj.parameters():
                param.requires_grad = False

            Iters3 = 500
            J_AE = sum(ae_output_num)
            Iter_obj = np.zeros((Iters3,4))
            Iter_obj = pd.DataFrame(Iter_obj,columns = ['Obj.pow','Obj.tem','Power_pen','Sum_Obj'])
            Iter_obj_zero = np.zeros((Iters3,4))  #用来存放扰动后的目标值的numpy
            Iter_sens_grd = torch.zeros((Iters3,opt_period,ae_output_num[1])).to(device)
            Iter_J_tr_en = torch.zeros((Iters3,opt_period,J_AE)).to(device)
            for i in range(Iters3):
                Iter_J_tr_en[i,:,:] = J_tr_en_ini.clone().detach()

            Iter_lr = 3e-3
            Iter_cond = 1e-3
            Iter_count = 0
            Iter_incre = 0
            Time_obj,Time_grd = [],[]
            u_direction = np.random.uniform(0, 1, (96, 4, Iters3))
            #u_direction = np.random.randn(96, 4, Iters3)
            u_r, u_d = 0.1, 1.0
            two_points = 'yes'
            a_momentum = 0.2
            for iter in range(0, Iters3):
                u_k = u_direction[:,:,iter]
                if iter == 0:
                    Iter_J_tr_en[iter,:,:] = J_tr_en_ini.clone().detach()   #Iter_J_tr_en_dis[0,:,:].clone().detach()
                elif iter == 1:
                    Iter_J_tr_en[iter,:,3:7] = Iter_J_tr_en[iter-1,:,3:7] - Iter_lr*Iter_sens_grd[iter-1,:,:]
                elif iter >= 2:
                    Iter_J_tr_en[iter,:,3:7] = Iter_J_tr_en[iter-1,:,3:7] - Iter_lr*Iter_sens_grd[iter-1,:,:] \
                                + a_momentum*(Iter_J_tr_en[iter-1,:,3:7] - Iter_J_tr_en[iter-2,:,3:7])

                J_tr_en = to_np(Iter_J_tr_en[iter,:,:])
                J_tr_en_Y_ini,J_tr_en_P_ini,J_tr_en_X_ini = torch.tensor(J_tr_en[:,:3],dtype=torch.float64).to(device),torch.tensor(J_tr_en[:,3:3+4],dtype=torch.float).to(device),torch.tensor(J_tr_en[:,3+4:],dtype=torch.float64).to(device)
                J_tr_en_Y_ini.requires_grad_(True)
                J_tr_en_P_ini.requires_grad_(True)
                J_tr_en_X_ini.requires_grad_(True)
                
                Obj_Y_upp,Obj_Y_low,Obj_P_1,Obj_P_2upp,Obj_P_2low,J_tr_de_P,J_tr_de_Y = \
                    M_obj(J_tr_en_Y_ini,J_tr_en_P_ini,J_tr_en_X_ini,P_price,Upp_Y,Low_Y,Upp_P,Low_P)
                
                time1 = time.time()
                Black_in = to_np(J_tr_en_P_ini)
                J_tr_de_s_hy,J_tr_de_p_hy = u_obj(Black_in,M_model_deter,J_tr_de_Y,M_model,S_Z_tr_ini,SS_Y,SS_P)
                J_obj_y = pho_tem*(1/T_Fre)*sum(sum(relu(J_tr_de_s_hy[t,i]-c_upper_tem[t,i])**2+relu(c_lower_tem[t,i]-J_tr_de_s_hy[t,i])**2 for i in range(nn_zone)) for t in range(opt_period))
                J_obj_p1 = sum(c_price[c_period_i,t]*(1/T_Fre)*(1/c_PI)*sum(J_tr_de_p_hy[t,:]) for t in range(opt_period-1))
                J_obj_p2 = w_pow*sum(relu(J_tr_de_p_hy[t,i]-c_PI*c_upper_p[t,i])**2 + relu(c_PI*c_lower_p[t,i]-J_tr_de_p_hy[t,i])**2 for t in range(opt_period-1) for i in range(nn_zone-len(P_dele)))
                J_obj = J_obj_y+J_obj_p1+J_obj_p2
                time2 = time.time()
                Black_in += u_r*u_k 
                J_tr_de_s_hy_u,J_tr_de_p_hy_u = u_obj(Black_in,M_model_deter,J_tr_de_Y,M_model,S_Z_tr_ini,SS_Y,SS_P)
                J_black_y = pho_tem*(1/T_Fre)*sum(sum(relu(J_tr_de_s_hy_u[t,i]-c_upper_tem[t,i])**2+relu(c_lower_tem[t,i]-J_tr_de_s_hy_u[t,i])**2 for i in range(nn_zone)) for t in range(opt_period))
                J_black_p1 = sum(c_price[c_period_i,t]*(1/T_Fre)*(1/c_PI)*sum(J_tr_de_p_hy_u[t,:]) for t in range(opt_period-1))
                J_black_p2 = w_pow*sum(relu(J_tr_de_p_hy_u[t,i]-c_PI*c_upper_p[t,i])**2 + relu(c_PI*c_lower_p[t,i]-J_tr_de_p_hy_u[t,i])**2 for t in range(opt_period-1) for i in range(nn_zone-len(P_dele)))
                J_black = J_black_y+J_black_p1+J_black_p2
                if two_points == 'yes': 
                    Iter_sens_grd[iter,:,:] = torch.tensor(u_d*(1/u_r)*(J_black-J_obj)*u_k).to(device)
                else:
                    if iter == 0:
                        Iter_sens_grd[iter,:,:] = torch.tensor(u_d*(1/u_r)*(J_black)*u_k).to(device)
                    else:
                        Iter_sens_grd[iter,:,:] = torch.tensor(u_d*(1/u_r)*(J_black-Iter_obj_zero[iter-1,-1])*u_k).to(device)
                
                time3 = time.time()
                
                Iter_obj.iloc[iter,:] = [J_obj_p1,J_obj_y,J_obj_p2, J_obj]
                Iter_obj_zero[iter,:] = [J_black_p1,J_black_y,J_black_p2, J_black]
                Time_obj.append(time2-time1)
                Time_grd.append(time3-time2)

                if iter > 0 and Iter_obj.iloc[iter,-1] > Iter_obj.iloc[iter-1,-1]:
                    print('!!!!')
                    Iter_incre += 1
                else:
                    Iter_incre = 0
                if Iter_incre >= 2:
                    print('Reduce step_size!')
                    #Iter_lr = 0.5*Iter_lr

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

            final_i = iter
            Final_en = Iter_J_tr_en[final_i,:,:]

            SS_Z_min,SS_Z_max = [SS_Z.data_min_[i] for i in range(Z_tr.shape[-1])],[SS_Z.data_max_[i] for i in range(Z_tr.shape[-1])]
            Final_de_Y = model_output(M_model_deter,Final_en[:,:ae_output_num[0]],Final_en[:,ae_output_num[0]:sum(ae_output_num[0:2])],Final_en[:,sum(ae_output_num[0:2]):])
            Final_de_P = M_model_deter.encoder1.decoder(Final_en[:,ae_output_num[0]:sum(ae_output_num[0:2])])
            Final_de_X = M_model_deter.encoder2.decoder(Final_en[:,sum(ae_output_num[0:2]):])
            Final_de_Y,Final_de_P,Final_de_X = to_np(Final_de_Y),to_np(Final_de_P),to_np(Final_de_X)
            Final_de_Y_hy = SS_Y.inverse_transform(Final_de_Y)
            Final_de_P_hy = SS_P.inverse_transform(Final_de_P)
            Final_de_P_hy_jiu = copy.deepcopy(Final_de_P_hy)
            for ii in range(96-1):
                for jj in range(80):
                    if Final_de_P_hy_jiu[ii,jj] >= 3.6*15:
                        Final_de_P_hy_jiu[ii,jj] = 3.6*15
                    elif Final_de_P_hy_jiu[ii,jj] <= 0:
                        Final_de_P_hy_jiu[ii,jj] = 0

            Final_de_P_jiu = SS_P.transform(Final_de_P_hy_jiu)

            Final_de_X_hy = SS_X.inverse_transform(Final_de_X)
            if opt_day_ind < c_period_train:
                aa_true = Z_tr_new[opt_day_ind*96+opt_start:opt_day_ind*96+opt_start+opt_period,2*nn_zone-len(P_dele):]
            else:
                opt_day_ind_up = opt_day_ind-c_period_train
                aa_true = Z_te_new[opt_day_ind_up*96+opt_start:opt_day_ind_up*96+opt_start+opt_period,2*nn_zone-len(P_dele):]

            
            Final_de = np.hstack((Final_de_Y,Final_de_P_jiu,S_Z_tr_ini[:,2*nn_zone-len(P_dele):]))  #Use real noises !
            Final_de_1 = Final_de[:-1,:].reshape(Final_de.shape[0]-1,1,Final_de.shape[1])
            
            Final_de_1[0,0,:nn_zone] = S_Z_tr_ini[0,:nn_zone]  #set the correct initial points
            Final_de_1 = torch.tensor(Final_de_1).to(device)   #must use real true value
            M_linear = 0
            M_nn_input,M_nn_hidden,M_nn_output = M_train_data.shape[-1],[128,64,64],nn_zone
            M_model_NN = MZ_Model_Func.model_FNN(M_nn_input, M_nn_hidden, M_nn_output,M_linear).to(device)
            m_state_dict = torch.load('Results/'+str(nn_zone)+'_whole_NN.pt')
            M_model_NN.load_state_dict(m_state_dict)
            Final_true = M_model_NN(Final_de_1)
            Final_true_hy = SS_Y.inverse_transform(to_np(Final_true[:,0,:]))
            #Final_true_err = Final_true_hy - Final_de_Y_hy

            Final_true_cost1 = pho_pow*(1/T_Fre)*(1/c_PI)*sum(c_price[opt_day_ind,opt_start+t]*sum(Final_de_P_hy_jiu[t,:]) for t in range(opt_period-1))
            Final_true_cost2_true = pho_tem*(1/T_Fre)*sum(sum(relu(Final_true_hy[t,i]-c_upper_tem[opt_start+t,i])**2 + relu(c_lower_tem[opt_start+t,i] - Final_true_hy[t,i])**2 for i in range(c_upper_tem.shape[1]) ) for t in range(opt_period))
            
            print(Iter_obj.iloc[final_i,0:2])
            print(Final_true_cost1,Final_true_cost2_true)

            if i_noise == 0:
                HZ_AE_P[i_day,:,:] = Final_de_P_hy_jiu[:,:]
                HZ_AE_Tem[i_day,:,:] = Final_true_hy[:,:]
                HZ_AE_Tem_diff[i_day,:,:] = Final_true_hy[:,:]-J_tr_de_s_hy[:,:]
                HZ_AE_obj[i_day,:] = Iter_obj.iloc[final_i,0],Iter_obj.iloc[final_i,1],  Iter_obj.iloc[final_i,0]+Iter_obj.iloc[final_i,1],\
                    Final_true_cost1,Final_true_cost2_true,  Final_true_cost1+Final_true_cost2_true,  iter,sum(Time_obj)/len(Time_obj), sum(Time_grd)/len(Time_grd)
            else:
                Noise_AE_obj[i_day,i_noise-1,:] = Iter_obj.iloc[final_i,0],Iter_obj.iloc[final_i,1],  Iter_obj.iloc[final_i,0]+Iter_obj.iloc[final_i,1],\
                    Final_true_cost1,Final_true_cost2_true,  Final_true_cost1+Final_true_cost2_true
#             if i_noise == 0 and i_day == 20:
#                 np.save('MZ_results/HZ_AEzero_loss.npy',Iter_obj.iloc[:final_i+1,-1].to_numpy())


# np.save('MZ_results/HZ_AEzero_P.npy',HZ_AE_P)
# np.save('MZ_results/HZ_AEzero_Tem.npy',HZ_AE_Tem)
# np.save('MZ_results/HZ_AEzero_Tem_diff.npy',HZ_AE_Tem_diff)
# np.save('MZ_results/HZ_AEzero_obj.npy',HZ_AE_obj)
# np.save('MZ_results/Noise_AEzero_obj.npy',Noise_AE_obj)
