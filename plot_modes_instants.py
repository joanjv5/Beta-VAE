

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import sys
import matplotlib.pyplot as plt
from dataset import Load_h5_file, pre_proces
import numpy as np
from torchsummary import summary
from tools import reshape_var, h5_load, plotSnapshot
from classes_2 import VariationalAutoencoder, Decoder, Encoder, masc2

channels = 16
latent_dimension = 5
batch_size = 32
vae = VariationalAutoencoder(channels, latent_dimension)
meshDict, time, varDict = h5_load('CYLINDER.h5')

vae.load_state_dict(torch.load('beta_vae2'))
vae.eval()

dataset = Load_h5_file('CYLINDER.h5')
len_train = int(0.7*len(dataset))+1
print(len_train)
len_test = int(0.1*len(dataset))
print(len_test)
len_vali = int(0.2*len(dataset))
print(len_vali)


train_dataset , test_dataset, validation_dataset = torch.utils.data.random_split(dataset,(len_train,len_test, len_vali))


validation_dataloader = DataLoader(validation_dataset,batch_size=batch_size, shuffle=True)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,)

test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,)


_, mit_value, _  = pre_proces(varDict,time)
mit_tensor = mit_value*np.ones([89351,1])
mit_tensor = torch.tensor(mit_tensor)

with torch.no_grad():
    snap3 = np.float32([0,0,0,0,1])
    snap3 = np.tile(snap3,[32,1])
    instant = iter(train_dataloader)
    batch = next(instant)
    snap2 = batch[0,0,:,:]
    recon_instant = vae(batch)
    recon_instant = np.asanyarray(recon_instant[0])
    recon_instant = masc2(torch.tensor(recon_instant))
    recon_instant = recon_instant[0,:,:]
    recon_instant =  torch.reshape(recon_instant,[89351,1]) + mit_tensor
    snap2 = masc2(snap2)
    snap2 = torch.reshape(snap2, [89351,1]) +mit_tensor
    snap3 = torch.from_numpy(snap3)
    snap3 = vae.decoder(snap3)   
    snap3 = snap3[1,:,:,:]
    snap3 = masc2(snap3)
    snap3 = snap3.resize_([89351,1]) 
    
    
#snap 3 plots one mode, snap2 plots the original instant of time, and recon_instant its reconstruction

varDict.update({'NEW2': {'point':varDict['VELOX'].get('point'),'ndim':varDict['VELOX'].get('ndim'),'value': snap3} })
plotSnapshot(meshDict,varDict,vars=['NEW2'],instant=0,cmap='jet',cpos='xy')

varDict.update({'NEW2': {'point':varDict['VELOX'].get('point'),'ndim':varDict['VELOX'].get('ndim'),'value': snap2} })
plotSnapshot(meshDict,varDict,vars=['NEW2'],instant=0,cmap='jet',cpos='xy')
