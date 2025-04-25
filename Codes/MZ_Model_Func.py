#%%
import numpy as np
import math
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import random
import copy
import torch
import torch.nn as nn
import torch.optim as optim
device = torch.device("cuda:3")
from sklearn.metrics import r2_score


class Autoencoder(nn.Module):
    def __init__(self,input_num,hidden_units,output_num):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_num, hidden_units[0]),
            nn.ReLU(),
            nn.Linear(hidden_units[0], hidden_units[1]),
            nn.ReLU(),
            nn.Linear(hidden_units[1], hidden_units[2]),
            nn.ReLU(),
            nn.Linear(hidden_units[2], output_num),).to(device)
        self.decoder = nn.Sequential(
            nn.Linear(output_num, hidden_units[2]),
            nn.ReLU(),
            nn.Linear(hidden_units[2], hidden_units[1]),
            nn.ReLU(),
            nn.Linear(hidden_units[1], hidden_units[0]),
            nn.ReLU(),
            nn.Linear(hidden_units[0], input_num),).to(device)
        
    def forward(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded,decoded

def getdata(batch_step_bei, T_Fre, S_input_tr, FF, input):
    batch_size = int((S_input_tr[0].shape[0])/(24*T_Fre))
    batch_days = [i for i in range(batch_size)]
    batch_num = 24*T_Fre
    batch_time = torch.arange(0., batch_num, FF).to(device)

    input_num_0 = S_input_tr[0].shape[1] 
    T_Y_tr = np.zeros((input_num_0,batch_size,len(batch_time)))
    for i0 in range(input_num_0):
        for i in range(batch_size):
            for j in range(len(batch_time)):
                T_Y_tr[i0,i,j] = S_input_tr[0][i*(batch_num)+FF*j,i0]
    T_Y_tr = torch.tensor(T_Y_tr).to(device)

    if input == 1:
        input_num = S_input_tr[1].shape[1]
        T_X_tr = np.zeros((input_num,batch_size,len(batch_time)))
        for i in range(input_num):
            for j in range(batch_size):
                for k in range(len(batch_time)):
                    T_X_tr[i,j,k] = S_input_tr[1][j*(batch_num)+FF*k,i]
        T_X_tr = torch.tensor(T_X_tr).to(device)
    elif input == 0: 
        input_num = S_input_tr[0].shape[1]
        T_X_tr = np.zeros((input_num,batch_size,len(batch_time)))
        for i in range(input_num):
            for j in range(batch_size):
                for k in range(len(batch_time)):
                    T_X_tr[i,j,k] = S_input_tr[0][j*(batch_num)+FF*k,i]
        T_X_tr = torch.tensor(T_X_tr).to(device)           

    batch_y0 = torch.zeros((len(batch_days),1,input_num_0)).to(device)  #(21,1,1)
    for j in range(input_num_0):
        batch_y0[:,0,j] = T_Y_tr[j,batch_days,0]
    NN_step = (len(batch_time)-1)*batch_step_bei + 1
    batch_y11 = torch.zeros((NN_step,len(batch_days),1,T_X_tr.shape[0])).to(device)   #(191,21,1,5)
    if batch_step_bei == 1:
        for i in range(len(batch_days)):
            batch_y11[:,i,0,:] = torch.transpose(T_X_tr[:,i,:],0,1)
    batch_y = torch.zeros((len(batch_time),len(batch_days),1,input_num_0)).to(device)  #(96,21,1,1)
    for i in range(len(batch_days)):
        for j in range(input_num_0):
            batch_y[:,i,0,j] = T_Y_tr[j,batch_days[i],:]
    return T_Y_tr, batch_y0, batch_y11, batch_y

def data_input(T_Fre,Train_s,Train_e,Train_s2,Train_e2,Train_s3,Train_e3,N_zone,P_dele,pattern):
    T_len = 8760*T_Fre
    if N_zone == 90:
        data_in = np.loadtxt("90zone_15min.csv",delimiter=",",skiprows=1,usecols=range(1,5*N_zone+2))
        data_in_new = np.loadtxt("90zone_15min_newpattern.csv",delimiter=",",skiprows=1,usecols=range(1,5*N_zone+2))
    P0,T_in0 = np.zeros((T_len,N_zone)),np.zeros((T_len,N_zone))
    T_rad0 = data_in[:,1+N_zone*1:1+N_zone*2]/1000
    for ii in range(N_zone):
        P0[:,ii] = data_in[:,3+N_zone*2+3*ii]/1000
        T_in0[:,ii] = data_in[:,1+N_zone*2+3*ii]
    T_o0 = data_in[:,0].reshape(-1,1)
    T_occ0 = data_in[:,1:1+N_zone*1].copy()/1000


    P0_new,T_in0_new = np.zeros((T_len,N_zone)),np.zeros((T_len,N_zone))
    T_rad0_new = data_in_new[:,1+N_zone*1:1+N_zone*2]/1000
    for ii in range(N_zone):
        P0_new[:,ii] = data_in_new[:,3+N_zone*2+3*ii]/1000
        T_in0_new[:,ii] = data_in_new[:,1+N_zone*2+3*ii]
    T_o0_new = data_in_new[:,0].reshape(-1,1)
    T_occ0_new = data_in_new[:,1:1+N_zone*1].copy()/1000

    if len(P_dele) > 0:
        P0 = np.delete(P0, P_dele, axis=1)  
        P0_new = np.delete(P0_new, P_dele, axis=1)
    X_tr = np.hstack((T_in0[Train_s:Train_e,:],P0[Train_s:Train_e,:],T_o0[Train_s:Train_e,0:1],T_rad0[Train_s:Train_e,:],T_occ0[Train_s:Train_e,:]))
    X_va = np.hstack((T_in0[Train_s2:Train_e2,:],P0[Train_s2:Train_e2,:],T_o0[Train_s2:Train_e2,0:1],T_rad0[Train_s2:Train_e2,:],T_occ0[Train_s2:Train_e2,:]))
    if pattern == 'old':
        X_te = np.hstack((T_in0[Train_s3:Train_e3,:],P0[Train_s3:Train_e3,:],T_o0[Train_s3:Train_e3,0:1],T_rad0[Train_s3:Train_e3,:],T_occ0[Train_s3:Train_e3,:]))
    else:
        X_te = np.hstack((T_in0_new[Train_s3:Train_e3,:],P0_new[Train_s3:Train_e3,:],T_o0_new[Train_s3:Train_e3,0:1],T_rad0_new[Train_s3:Train_e3,:],T_occ0_new[Train_s3:Train_e3,:]))
    
    SS_X = MinMaxScaler().fit(X_tr)
    
    S_X_tr = SS_X.transform(X_tr)
    S_X_va = SS_X.transform(X_va)
    S_X_te = SS_X.transform(X_te)
    return SS_X,S_X_tr,S_X_va,S_X_te,X_tr,X_va,X_te

def ACC(SS_X,J_tr_de,S_X_tr):
    T_len = S_X_tr.shape[0]
    J_tr_de_huanyuan = SS_X.inverse_transform(J_tr_de)
    S_X_tr_huanyuan = SS_X.inverse_transform(S_X_tr)
    Err_tr = np.abs(J_tr_de_huanyuan - S_X_tr_huanyuan)
    ERR_rmse,ERR_mae,ERR_r2 = [],[],[]
    for fea in range(S_X_tr.shape[1]):
        Err_tr1 = math.sqrt(sum(Err_tr[i,fea]**2/(T_len) for i in range(T_len)))  #RMSE
        Err_tr2 = sum(Err_tr[i,fea] for i in range(T_len))/(T_len)  #MAE
        Err_tr3 = r2_score(S_X_tr_huanyuan[:,fea], J_tr_de_huanyuan[:,fea])  #R2
        ERR_rmse.append(Err_tr1)
        ERR_mae.append(Err_tr2)
        ERR_r2.append(Err_tr3)
    return ERR_rmse,ERR_mae,ERR_r2


def model_data(S_X_tr,S_P_tr,S_Y_tr, predict_period):
    N_sample = int((S_X_tr.shape[0])/predict_period)
    model_data = np.zeros((predict_period, N_sample, S_X_tr.shape[-1]+S_P_tr.shape[-1]+S_Y_tr.shape[-1]))
    model_label = np.zeros((predict_period,N_sample,S_Y_tr.shape[-1]))
    for i in range(N_sample):
        i_start = i*predict_period
        model_data[:,i,:S_Y_tr.shape[-1]] = S_Y_tr[i_start:i_start+predict_period,:]
        model_data[:,i,S_Y_tr.shape[-1]:S_Y_tr.shape[-1]+S_P_tr.shape[-1]] = S_P_tr[i_start:i_start+predict_period,:]
        model_data[:,i,S_Y_tr.shape[-1]+S_P_tr.shape[-1]:] = S_X_tr[i_start:i_start+predict_period,:]
        model_label[:,i,:] = S_Y_tr[i_start:i_start+predict_period,:]
    model_data = torch.tensor(model_data).to(device)
    model_label = torch.tensor(model_label).to(device)
    return model_data,model_label


class model_FNN(nn.Module):
    def __init__(self,input_num,hidden_units,output_num,linear):
        super(model_FNN, self).__init__()
        self.input_num,self.hidden_units,self.output_num = input_num,hidden_units,output_num
        self.layer_num = len(hidden_units)
        layers = []
        layers.append(nn.Linear(self.input_num, self.hidden_units[0]))
        if linear == 0:
            layers.append(nn.ReLU())
        for ii in range(self.layer_num-1):
            layers.append(nn.Linear(self.hidden_units[ii], self.hidden_units[ii+1]))
            if linear == 0:
                layers.append(nn.ReLU())
        layers.append(nn.Linear(self.hidden_units[-1], self.output_num))
        self.net = nn.Sequential(*layers)
    
    def forward(self,u):
        u_y = torch.zeros(u.shape[0]+1, u.shape[1], self.output_num, dtype=torch.float64).to(device)
        u_y[0,:,:] = u[0,:,:self.output_num]
        for j in range(u_y.shape[0]-1):
            input_new = torch.zeros(u.shape[1],u.shape[2]).to(device)
            if j == 0:
                input_new[:,:] = u[j,:,:]
            else:
                input_new[:,self.output_num:] = u[j,:,self.output_num:]
                input_new[:,:self.output_num] = u_y[j,:,:]
            u_out = self.net(input_new)
            u_y[j+1:j+2,:,:] = u_out
        return u_y

def to_np(x):
    return x.cpu().detach().numpy()


class mm_NN(nn.Module):
    def __init__(self,input_num,hidden_units,output_num,linear):
        super(mm_NN, self).__init__()
        self.input_num,self.hidden_units,self.output_num = input_num,hidden_units,output_num
        self.layer_num = len(hidden_units)
        layers = []
        layers.append(nn.Linear(self.input_num, self.hidden_units[0]))
        if linear == 0:
            layers.append(nn.ReLU())
        for ii in range(self.layer_num-1):
            layers.append(nn.Linear(self.hidden_units[ii], self.hidden_units[ii+1]))
            if linear == 0:
                layers.append(nn.ReLU())
        layers.append(nn.Linear(self.hidden_units[-1], self.output_num))
        self.net = nn.Sequential(*layers)
    
    def forward(self,input_new):
        u_out = self.net(input_new)
        return u_out
class model_joint(nn.Module):
    def __init__(self,ae_input_num,ae_hidden_units,ae_output_num,mm_hidden_units,linear,ae_ini,mm_ini,ae_paras,mm_paras):
        super(model_joint, self).__init__()
        self.ae_input_num,self.ae_hidden_units,self.ae_output_num = ae_input_num,ae_hidden_units,ae_output_num
        self.mm_input_num,self.mm_hidden_units,self.mm_output_num = sum(ae_output_num),mm_hidden_units,ae_output_num[0]
        self.mm_layer_num = len(mm_hidden_units)

        self.encoder0 = Autoencoder(ae_input_num[0],ae_hidden_units[0],ae_output_num[0]).to(device)
        self.encoder1 = Autoencoder(ae_input_num[1],ae_hidden_units[1],ae_output_num[1]).to(device)
        self.encoder2 = Autoencoder(ae_input_num[2],ae_hidden_units[2],ae_output_num[2]).to(device)

        self.mm = mm_NN(self.mm_input_num, self.mm_hidden_units, self.mm_output_num,linear).to(device)

        if ae_ini == 'True':
            self.encoder0.load_state_dict(ae_paras[0])
            self.encoder1.load_state_dict(ae_paras[1])
            self.encoder2.load_state_dict(ae_paras[2])
        if mm_ini == 'True':
            self.mm.load_state_dict(mm_paras)


    def forward(self,x0,x1,x2):   #Must follow original time-series order
        en_x0,de_x0 = self.encoder0(x0)
        en_x1,de_x1 = self.encoder1(x1)
        en_x2,de_x2 = self.encoder2(x2)

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
            u_out = self.mm(input_new)
            for tt in range(N_sample):
                u_y[tt*96+j+1,:] = u_out[tt,:]
        u_y_de = self.encoder0.decoder(u_y)
        return en_x0,en_x1,en_x2,de_x0,de_x1,de_x2,u_y,u_y_de



class NN_obj_layer(nn.Module):
    def __init__(self,ae_input_num,ae_hidden_units,ae_output_num,mm_hidden_units,ae_paras,mm_paras,joint_paras,SS_Y_max,SS_Y_min,SS_P_max,SS_P_min,wei_Y,wei_P,linear):
        super(NN_obj_layer, self).__init__()
        self.mm_input_num,self.mm_hidden_units,self.mm_output_num = sum(ae_output_num),mm_hidden_units,ae_output_num[0]
        self.M_model_deter = model_joint(ae_input_num,ae_hidden_units,ae_output_num,mm_hidden_units,linear,'False','False',ae_paras,mm_paras).to(device)
        self.M_model_deter.load_state_dict(joint_paras)
        self.relu = nn.ReLU()
        
        self.linear_Y = nn.Linear(90, 90).to(device)
        self.weight_Y = torch.zeros((90,90), dtype=torch.float32).to(device)
        self.bias_Y = torch.zeros((96,90), dtype=torch.float32).to(device)
        for i in range(90):
            self.weight_Y[i,i] = SS_Y_max[i]-SS_Y_min[i]
            self.bias_Y[:,i] = SS_Y_min[i]
        self.linear_Y.weight = nn.Parameter(self.weight_Y)
        self.linear_P = nn.Linear(90, 80).to(device)
        self.weight_P = torch.zeros((80,80), dtype=torch.float).to(device)
        self.bias_P = torch.zeros((96,80), dtype=torch.float).to(device)
        for i in range(80):
            self.weight_P[i,i] = SS_P_max[i]-SS_P_min[i]
            self.bias_P[:,i] = SS_P_min[i]
        self.linear_P.weight = nn.Parameter(self.weight_P)
        
        self.obj_Y_upp,self.obj_Y_low = nn.Linear(90, 90).to(device),nn.Linear(90, 90).to(device)
        self.obj_weight_Y_upp,self.obj_weight_Y_low = torch.zeros((90,90), dtype=torch.float32).to(device),torch.zeros((90,90), dtype=torch.float32).to(device)
        for i in range(90):
            self.obj_weight_Y_upp[i,i] = wei_Y
            self.obj_weight_Y_low[i,i] = -1*wei_Y            
        self.obj_Y_upp.weight = nn.Parameter(self.obj_weight_Y_upp)#,nn.Parameter(obj_bias_Y_upp)
        self.obj_Y_low.weight = nn.Parameter(self.obj_weight_Y_low)#,nn.Parameter(obj_bias_Y_low)
        
        self.obj_P_upp,self.obj_P_low = nn.Linear(80, 80).to(device),nn.Linear(80, 80).to(device)
        self.obj_weight_P_upp,self.obj_weight_P_low = torch.zeros((80,80), dtype=torch.float32).to(device),torch.zeros((80,80), dtype=torch.float32).to(device)
        for i in range(80):
            self.obj_weight_P_upp[i,i] = wei_P
            self.obj_weight_P_low[i,i] = -1*wei_P
        self.obj_P_upp.weight = nn.Parameter(self.obj_weight_P_upp)#,nn.Parameter(obj_bias_P_upp)
        self.obj_P_low.weight = nn.Parameter(self.obj_weight_P_low)#,nn.Parameter(obj_bias_P_low)

        self.sum_P = nn.Linear(80,1, dtype=torch.float32).to(device)
        self.sum_P_weight = torch.zeros((80, 80), dtype=torch.float32).to(device)
        for i in range(80):
            self.sum_P_weight[i,i] = 1.0
        self.sum_P.weight = nn.Parameter(torch.ones((80, 1), dtype=torch.float32).to(device))
        
        self.sum_Y = nn.Linear(90,1, dtype=torch.float32).to(device)
        self.sum_Y_weight = torch.zeros((90, 90), dtype=torch.float32).to(device)
        for i in range(90):
            self.sum_Y_weight[i,i] = 1.0
        self.sum_Y.weight = nn.Parameter(torch.ones((90, 1), dtype=torch.float32).to(device))

    def forward(self,J_tr_en_Y_ini,J_tr_en_P_ini,J_tr_en_X_ini,P_price,Upp_Y,Low_Y,Upp_P,Low_P):   #Must follow original time-series order
        N_sample = int((J_tr_en_Y_ini.shape[0])/(24*4))
        u_y = torch.zeros(J_tr_en_Y_ini.shape[0], J_tr_en_Y_ini.shape[1], dtype=torch.float).to(device)
        for i in range(N_sample):
            u_y[i*96,:] = J_tr_en_Y_ini[i*96,:]
        for j in range(96-1):
            input_new = torch.zeros(N_sample, J_tr_en_Y_ini.shape[1]+J_tr_en_P_ini.shape[1]+J_tr_en_X_ini.shape[1]).to(device)
            for tt in range(N_sample):
                input_new[tt,:J_tr_en_Y_ini.shape[1]] = u_y[tt*96+j,:]
                input_new[tt,J_tr_en_Y_ini.shape[1]:(J_tr_en_Y_ini.shape[1]+J_tr_en_P_ini.shape[1])] = J_tr_en_P_ini[tt*96+j,:]
                input_new[tt,(J_tr_en_Y_ini.shape[1]+J_tr_en_P_ini.shape[1]):(J_tr_en_Y_ini.shape[1]+J_tr_en_P_ini.shape[1]+J_tr_en_X_ini.shape[1])] = J_tr_en_X_ini[tt*96+j,:]
            u_out = self.M_model_deter.mm(input_new)
            for tt in range(N_sample):
                u_y[tt*96+j+1,:] = u_out[tt,:]
        J_tr_de_Y = self.M_model_deter.encoder0.decoder(u_y).to(device)
        J_tr_de_P = self.M_model_deter.encoder1.decoder(J_tr_en_P_ini).to(device)

        J_tr_de_Y_hy = torch.matmul(J_tr_de_Y, self.weight_Y)  #self.linear_Y(J_tr_de_Y).to(device)
        J_tr_de_Y_hy += self.bias_Y
        J_tr_de_P_hy = torch.matmul(J_tr_de_P, self.weight_P)  #self.linear_P(J_tr_de_P).to(device)
        J_tr_de_P_hy += self.bias_P

        Obj_Y_upp = torch.matmul(J_tr_de_Y_hy,self.obj_weight_Y_upp)  #self.obj_Y_upp(J_tr_de_Y_hy).to(device)  #self.obj_weight_Y_upp
        Obj_Y_upp -= Upp_Y
        Obj_Y_upp = self.relu(Obj_Y_upp)
        Obj_Y_upp = torch.matmul(Obj_Y_upp,self.sum_Y_weight)   #self.sum_Y(Obj_Y_upp)
        Obj_Y_low = torch.matmul(J_tr_de_Y_hy,self.obj_weight_Y_low)   #self.obj_Y_low(J_tr_de_Y_hy).to(device)
        Obj_Y_low += Low_Y
        Obj_Y_low = self.relu(Obj_Y_low)
        Obj_Y_low = torch.matmul(Obj_Y_low,self.sum_Y_weight)   #self.sum_Y(Obj_Y_low)
        
        Obj_P_1 = J_tr_de_P_hy * P_price
        
        Obj_P_2upp = torch.matmul(J_tr_de_P_hy,self.obj_weight_P_upp)  # self.obj_P_upp(J_tr_de_P_hy).to(device)
        Obj_P_2upp -= Upp_P
        Obj_P_2upp = self.relu(Obj_P_2upp)
        Obj_P_2upp = torch.matmul(Obj_P_2upp,self.sum_P_weight)  ##self.sum_P(Obj_P_2upp)
        Obj_P_2low = torch.matmul(J_tr_de_P_hy,self.obj_weight_P_low)  #self.obj_P_low(J_tr_de_P_hy).to(device)
        Obj_P_2low += Low_P
        Obj_P_2low = self.relu(Obj_P_2low)
        Obj_P_2low = torch.matmul(Obj_P_2low,self.sum_P_weight)  ##self.sum_P(Obj_P_2low)
        
        return Obj_Y_upp,Obj_Y_low,Obj_P_1,Obj_P_2upp,Obj_P_2low,J_tr_de_P,J_tr_de_Y


class Ori_NN_obj_layer(nn.Module):
    def __init__(self,M_nn_input, M_nn_hidden, M_nn_output,M_linear,m_state_dict,SS_Y_max,SS_Y_min,SS_P_max,SS_P_min,wei_Y,wei_P):
        super(Ori_NN_obj_layer, self).__init__()
        self.M_model_deter = model_FNN(M_nn_input, M_nn_hidden, M_nn_output,M_linear).to(device)
        self.M_model_deter.load_state_dict(m_state_dict)
        self.relu = nn.ReLU()
        
        self.linear_Y = nn.Linear(90, 90).to(device)
        self.weight_Y = torch.zeros((90,90), dtype=torch.float32).to(device)
        self.bias_Y = torch.zeros((96,90), dtype=torch.float32).to(device)
        for i in range(90):
            self.weight_Y[i,i] = SS_Y_max[i]-SS_Y_min[i]
            self.bias_Y[:,i] = SS_Y_min[i]
        self.linear_Y.weight = nn.Parameter(self.weight_Y)
        self.linear_P = nn.Linear(90, 80).to(device)
        self.weight_P = torch.zeros((80,80), dtype=torch.float).to(device)
        self.bias_P = torch.zeros((96,80), dtype=torch.float).to(device)
        for i in range(80):
            self.weight_P[i,i] = SS_P_max[i]-SS_P_min[i]
            self.bias_P[:,i] = SS_P_min[i]
        self.linear_P.weight = nn.Parameter(self.weight_P)
        
        
        self.obj_Y_upp,self.obj_Y_low = nn.Linear(90, 90).to(device),nn.Linear(90, 90).to(device)
        self.obj_weight_Y_upp,self.obj_weight_Y_low = torch.zeros((90,90), dtype=torch.float32).to(device),torch.zeros((90,90), dtype=torch.float32).to(device)
        for i in range(90):
            self.obj_weight_Y_upp[i,i] = wei_Y
            self.obj_weight_Y_low[i,i] = -1*wei_Y            
        self.obj_Y_upp.weight = nn.Parameter(self.obj_weight_Y_upp)#,nn.Parameter(obj_bias_Y_upp)
        self.obj_Y_low.weight = nn.Parameter(self.obj_weight_Y_low)#,nn.Parameter(obj_bias_Y_low)
        
        self.obj_P_upp,self.obj_P_low = nn.Linear(80, 80).to(device),nn.Linear(80, 80).to(device)
        self.obj_weight_P_upp,self.obj_weight_P_low = torch.zeros((80,80), dtype=torch.float32).to(device),torch.zeros((80,80), dtype=torch.float32).to(device)
        for i in range(80):
            self.obj_weight_P_upp[i,i] = wei_P
            self.obj_weight_P_low[i,i] = -1*wei_P
        self.obj_P_upp.weight = nn.Parameter(self.obj_weight_P_upp)#,nn.Parameter(obj_bias_P_upp)
        self.obj_P_low.weight = nn.Parameter(self.obj_weight_P_low)#,nn.Parameter(obj_bias_P_low)

        self.sum_P = nn.Linear(80,1, dtype=torch.float32).to(device)
        self.sum_P_weight = torch.zeros((80, 80), dtype=torch.float32).to(device)
        for i in range(80):
            self.sum_P_weight[i,i] = 1.0
        self.sum_P.weight = nn.Parameter(torch.ones((80, 1), dtype=torch.float32).to(device))
        
        self.sum_Y = nn.Linear(90,1, dtype=torch.float32).to(device)
        self.sum_Y_weight = torch.zeros((90, 90), dtype=torch.float32).to(device)
        for i in range(90):
            self.sum_Y_weight[i,i] = 1.0
        self.sum_Y.weight = nn.Parameter(torch.ones((90, 1), dtype=torch.float32).to(device))

    def forward(self,J_tr_en,P_price,Upp_Y,Low_Y,Upp_P,Low_P):
        J_tr_en_in = J_tr_en.reshape(J_tr_en.shape[0],1,J_tr_en.shape[1])
        J_tr_en_out = self.M_model_deter(J_tr_en_in[:-1,:,:])
        
        J_tr_de_Y = J_tr_en_out[:,0,:]
        J_tr_de_P = J_tr_en[:,90:170]
        J_tr_de_Y = J_tr_de_Y.type(torch.float32)
        J_tr_de_P = J_tr_de_P.type(torch.float32)        

        J_tr_de_Y_hy = torch.matmul(J_tr_de_Y, self.weight_Y)  #self.linear_Y(J_tr_de_Y).to(device)
        J_tr_de_Y_hy += self.bias_Y
        J_tr_de_P_hy = torch.matmul(J_tr_de_P, self.weight_P)  #self.linear_P(J_tr_de_P).to(device)
        J_tr_de_P_hy += self.bias_P

        Obj_Y_upp = torch.matmul(J_tr_de_Y_hy,self.obj_weight_Y_upp)  #self.obj_Y_upp(J_tr_de_Y_hy).to(device)  #self.obj_weight_Y_upp
        Obj_Y_upp -= Upp_Y
        Obj_Y_upp = self.relu(Obj_Y_upp)
        Obj_Y_upp = torch.matmul(Obj_Y_upp,self.sum_Y_weight)   #self.sum_Y(Obj_Y_upp)
        Obj_Y_low = torch.matmul(J_tr_de_Y_hy,self.obj_weight_Y_low)   #self.obj_Y_low(J_tr_de_Y_hy).to(device)
        Obj_Y_low += Low_Y
        Obj_Y_low = self.relu(Obj_Y_low)
        Obj_Y_low = torch.matmul(Obj_Y_low,self.sum_Y_weight)   #self.sum_Y(Obj_Y_low)
        
        Obj_P_1 = J_tr_de_P_hy * P_price
        
        Obj_P_2upp = torch.matmul(J_tr_de_P_hy,self.obj_weight_P_upp)  # self.obj_P_upp(J_tr_de_P_hy).to(device)
        Obj_P_2upp -= Upp_P
        Obj_P_2upp = self.relu(Obj_P_2upp)
        Obj_P_2upp = torch.matmul(Obj_P_2upp,self.sum_P_weight)  ##self.sum_P(Obj_P_2upp)
        Obj_P_2low = torch.matmul(J_tr_de_P_hy,self.obj_weight_P_low)  #self.obj_P_low(J_tr_de_P_hy).to(device)
        Obj_P_2low += Low_P
        Obj_P_2low = self.relu(Obj_P_2low)
        Obj_P_2low = torch.matmul(Obj_P_2low,self.sum_P_weight)  ##self.sum_P(Obj_P_2low)
        return Obj_Y_upp,Obj_Y_low,Obj_P_1,Obj_P_2upp,Obj_P_2low
