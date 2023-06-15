
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

from classes_2 import VariationalAutoencoder, EarlyStopper, vae_loss




dataset = Load_h5_file('CYLINDER.h5')
len_train = int(0.7*len(dataset))+1
print(len_train)
len_test = int(0.1*len(dataset))
print(len_test)
len_vali = int(0.2*len(dataset))
print(len_vali)

batch_size = 32

train_dataset , test_dataset, validation_dataset = torch.utils.data.random_split(dataset,(len_train,len_test, len_vali))


validation_dataloader = DataLoader(validation_dataset,batch_size=batch_size, shuffle=True)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,)

test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,)




dataset = Load_h5_file('CYLINDER.h5')
energy_dataset = DataLoader(dataset, batch_size=151 ,shuffle=False)

train_loss_avg = []
val_loss = []
test_loss = []
detR = []
num_epochs = 200
prev_train_loss = 1e9
beta = 2.5e-2
lat_dims = [5,10,15]
betas = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 2.5e-2, 5e-2]
ek_mean = torch.zeros(len(lat_dims),len(betas))
ek = torch.zeros(len(lat_dims),151,len(betas))
for k, lat_dim in enumerate(lat_dims):
    for j , beta in enumerate(betas):
        vae = VariationalAutoencoder(16,lat_dim)
        prev_train_loss = 1e9
        early_stopper = EarlyStopper(patience=5, min_delta=0.02)
        vae.train()
        test_loss.append(0)
        detR.append(0)
        learning_rate = 3e-4
        for epoch in range(num_epochs):
            train_loss_avg.append(0)
            num_batches = 0 
    
            learning_rate = learning_rate * 1/(1 + 0.001 * epoch)   #learnign rate scheduled 
    
    
            optimizer = torch.optim.Adam(vae.parameters(), lr= learning_rate)
            for batch in train_dataloader:
                recon, mu, logvar, _ = vae(batch)
                loss = vae_loss(batch, recon, mu, logvar, beta)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
                train_loss_avg[-1] += loss.item()
        
                num_batches += 1
            with torch.no_grad():
                val_batches = 0
                val_loss.append(0)
                for val_batch in validation_dataloader:
                    val_recon, val_mu, val_logvar , _ = vae(val_batch)
                    vali_loss = vae_loss(val_batch,val_recon,val_mu,val_logvar,beta)
                    val_loss[-1] += vali_loss.item()
                    val_batches += 1
                val_loss[-1] /= val_batches
                train_loss_avg[-1] /= num_batches
            if early_stopper.early_stop(val_loss[-1], prev_train_loss, train_loss_avg[-1] ):
                print('Early Stopper Activated at %f' %epoch)
                break
            prev_train_loss = train_loss_avg[-1]  
            
            print('Epoch [%d / %d] average training error: %f' % (epoch+1, num_epochs, train_loss_avg[-1]))
    
        vae.eval()
        with torch.no_grad():
            instant = iter(energy_dataset)
            energy_batch = next(instant)
            for i in range(151):
                x = energy_batch[i,0,:,:]
                x = torch.reshape(x, [1,1,448,192])
                x_recon = vae(x)
                x_recon = np.asanyarray(x_recon[0])
                x_recon = x_recon[0,0,:,:]
        
                x_recon = torch.reshape(torch.tensor(x_recon),[86016, 1])
                x = torch.reshape(x,[86016,1])
                ek[k,i,j] = torch.sum((x-x_recon)**2)/torch.sum(x**2)
            ek_mean[k,j] = (1-torch.mean(ek[k,:,j]))*100
    
        


   
    
    
plt.figure('A')
plt.plot(betas,ek_mean[0,:],color='r',marker='o', label = 'latent dimension = 5')
plt.plot(betas,ek_mean[1,:],color='g',marker='o',label = 'latent dimension = 15')
plt.plot(betas,ek_mean[2,:],color='b',marker='o',label = 'latent dimension = 20')
plt.xlabel("beta")
plt.ylabel("$E_{k}$")
plt.legend(loc='lower left')
plt.grid(True)
plt.show()