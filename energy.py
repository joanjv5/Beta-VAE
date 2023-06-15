

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
from dataset import Load_h5_file, pre_proces
import numpy as np
from tools import reshape_var, h5_load, plotSnapshot
from classes_2 import VariationalAutoencoder

vae = VariationalAutoencoder(16, 5)
vae.load_state_dict(torch.load('beta_vae2'))
meshDict, time, varDict = h5_load('CYLINDER.h5')

vae.eval()



with torch.no_grad():
    dataset2 = Load_h5_file('CYLINDER.h5')
    energy_dataset = DataLoader(dataset2, batch_size=151 ,shuffle=False)
    
    instant = iter(energy_dataset)
    batch = next(instant)
    print(batch.shape)
    #recon_instant = vae(batch)
    #recon_instant = np.asanyarray(recon_instant[0])
    ek = torch.tensor(np.zeros(151))
    
    
    
    
        
    num = 0  
    for i in range(batch.shape[0]):
        
        x = batch[i,0,:,:]
        x = torch.reshape(x, [1,1,448,192])
        x_recon = vae(x)
        x_recon = np.asanyarray(x_recon[0])
        x_recon = x_recon[0,0,:,:]
        
        x_recon = torch.reshape(torch.tensor(x_recon),[86016, 1])
        x = torch.reshape(x,[86016,1])
        
        print(torch.sum(x-x_recon),torch.sum(x))
        #ek[i] = pow(torch.sum(x-x_recon),2)/pow(torch.sum(x),2)
        ek[i] = torch.sum((x-x_recon)**2)/torch.sum(x**2)
        
        
        num += 1
        
    
    
    ek_mean = (1-torch.mean(ek))*100 #energy reconstructed
    
