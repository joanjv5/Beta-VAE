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


def masc2(cropped_tensor,new_dim =[449,199]):    #aquesta funcio ens permet reconstruir l'imatge a les dimensions necessaries (499x199)
    height = 448                                 #per poder plotjear amb pyvista
    padded_tensor = torch.ones([449, 199])
    width = 192

    # Calculate the coordinates of the upper-left corner of the crop
    y1 = 0
    x1 = 0
    padded_cropped_tensor = F.pad(cropped_tensor, (0, padded_tensor.shape[1]-width, 0, padded_tensor.shape[0]-height), mode='constant', value=0)
    print(padded_cropped_tensor.shape)
    return(padded_cropped_tensor)  
   

class Encoder(nn.Module):
    def __init__(self, channels, latent_dim):
        super(Encoder, self).__init__()
        
        self.c = channels
        self.lat_dim = latent_dim
        
        self.drop1 = nn.Dropout1d()
        self.drop2 = nn.Dropout2d()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels= self.c, kernel_size=4, stride= 2, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight)
        self.max = nn.MaxPool2d(kernel_size=2, stride= 2)
        self.batch1 = nn.BatchNorm2d(self.c)
        
        self.conv2 = nn.Conv2d(in_channels=self.c, out_channels= self.c*2, kernel_size=4, stride= 2, padding=1)
        nn.init.xavier_uniform_(self.conv2.weight)
        self.batch2 = nn.BatchNorm2d(self.c*2)
        
        self.conv3 = nn.Conv2d(in_channels=self.c*2, out_channels= self.c*2*2, kernel_size=4, stride= 2, padding=1)
        nn.init.xavier_uniform_(self.conv3.weight)
        self.batch3 = nn.BatchNorm2d(self.c*2*2)
        
        self.conv4 = nn.Conv2d(in_channels=self.c*2*2, out_channels= self.c*2*2*2, kernel_size=4, stride= 2, padding=1)
        nn.init.xavier_uniform_(self.conv4.weight)
        self.batch4 = nn.BatchNorm2d(self.c*2*2*2)
        
        self.conv5 = nn.Conv2d(in_channels=self.c*2*2*2, out_channels= self.c*2*2*2*2, kernel_size=4, stride= 2, padding=1)
        nn.init.xavier_uniform_(self.conv5.weight)
        self.batch5 = nn.BatchNorm2d(self.c*2*2*2*2)
        
        self.flat = nn.Flatten()
        #faltten layer
        
        self.fc1 = nn.Linear(in_features = self.c*2*2*2*2*14*6, out_features= 128)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.batch6 = nn.BatchNorm1d(128)
        
        self.mu = nn.Linear(in_features = 128, out_features = self.lat_dim)
        nn.init.xavier_uniform_(self.mu.weight)
        
        self.logvar = nn.Linear(in_features = 128, out_features = self.lat_dim)
        nn.init.xavier_uniform_(self.logvar.weight)
    
    def forward(self, x):
        
        out = torch.tanh(self.conv1(x))
        
        
        
        #out = self.max(out)
        
        out = torch.tanh(self.conv2(out))
        
        
        #out = self.max(out)
        
        out = torch.tanh(self.conv3(out))
        
        
        #out = self.max(out)
        
        out = torch.tanh(self.conv4(out))
        
        
        #out = self.max(out)
        
        out = torch.tanh(self.conv5(out))
       
        
        #out = self.max(out)
        
        out = self.flat(out)
        
        out = torch.tanh(self.fc1(out))
       
        
        mu = self.mu(out)
        
        logvar = self.logvar(out)
        
        return  mu , logvar
    
    
class Decoder(nn.Module):
    def __init__(self, channels, latent_dim):
        super(Decoder , self).__init__()
        
        
        
        self.c = channels
        self.lat_dim = latent_dim
        
        self.drop2 = nn.Dropout2d()
        self.drop1 = nn.Dropout1d()
        self.up = nn.UpsamplingNearest2d(scale_factor=2)
        
        self.fc1 = nn.Linear(in_features = self.lat_dim, out_features = 128)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.batch6 = nn.BatchNorm1d(128)
        
        self.fc2 = nn.Linear(in_features = 128, out_features = self.c*2*2*2*2*14*6)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.batch5 = nn.BatchNorm1d(self.c*2*2*2*2*14*6)
        
        #reshape layer
        
        
        self.conv5 = nn.ConvTranspose2d(in_channels = self.c*2*2*2*2, out_channels = self.c*2*2*2, kernel_size = 4, stride = 2, padding = 1)
        nn.init.xavier_uniform_(self.conv5.weight)
        self.batch4 = nn.BatchNorm2d(self.c*2*2*2)
        
        self.conv4 = nn.ConvTranspose2d(in_channels = self.c*2*2*2, out_channels = self.c*2*2, kernel_size = 4, stride = 2, padding = 1)
        nn.init.xavier_uniform_(self.conv4.weight)
        self.batch3 = nn.BatchNorm2d(self.c*2*2)
        
        self.conv3 = nn.ConvTranspose2d(in_channels = self.c*2*2, out_channels = self.c*2, kernel_size = 4, stride = 2, padding = 1)
        nn.init.xavier_uniform_(self.conv3.weight)
        self.batch2 = nn.BatchNorm2d(self.c*2)
        
        self.conv2 = nn.ConvTranspose2d(in_channels = self.c*2, out_channels = self.c, kernel_size = 4, stride = 2, padding = 1)
        nn.init.xavier_uniform_(self.conv2.weight)
        self.batch1 = nn.BatchNorm2d(self.c)
        
        self.conv1 = nn.ConvTranspose2d(in_channels = self.c, out_channels = 1 , kernel_size = 4, stride = 2, padding = 1)
        nn.init.xavier_uniform_(self.conv1.weight)
        
    def forward(self, x):
        
        out = torch.tanh(self.fc1(x))
        
        
        
        out = torch.tanh(self.fc2(out))
        
        
        out = out.view(out.size(0),self.c*2*2*2*2, 14, 6)
        #out = self.up(out)
        
        out = torch.tanh(self.conv5(out))
        
        
        #out = self.up(out)
        
        out = torch.tanh(self.conv4(out))
        
        #out = self.up(out)
        
        out = torch.tanh(self.conv3(out))
        
        #out = self.up(out)
        
        out = torch.tanh(self.conv2(out))
        
        #out = self.up(out)
        
        out = torch.tanh(self.conv1(out))
        
        return out


class VariationalAutoencoder(nn.Module):
    def __init__(self, channels, latent_dim):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder(channels = channels, latent_dim= latent_dim)
        self.decoder = Decoder(channels = channels, latent_dim = latent_dim)
        
    def reparamatrizate(self, mu, logvar):
        
        std = torch.exp(0.5*logvar)
        
        epsilon = torch.rand_like(std)  #we create a normal distribution (0 ,1 ) with the dimensions of std        
        sample = mu + std*epsilon
        
        return  sample
    
    def forward(self, x):
        
        mu, logvar = self.encoder(x)
        z = self.reparamatrizate(mu, logvar)
        recon = self.decoder(z)
        
        return recon, mu, logvar, z



    
def vae_loss(x, recon_x, mu, logvar, beta):
    
    #recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    recon_loss = F.mse_loss(recon_x.view(-1, 86016), x.view(-1, 86016),reduction='mean')
    kld = +0.5*torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    loss_fn = recon_loss - beta*kld
    print('kld:',beta*kld)
    print('rcn:', recon_loss)
    return loss_fn


def vae_loss2(x, recon_x, mu, logvar, beta):
    
    #recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    recon_loss = F.mse_loss(recon_x.view(-1, 86016), x.view(-1, 86016),reduction='mean')
    kld = 0.5*torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    print('kld:',beta*kld)
    print('rcn:', recon_loss)
    return  recon_loss, beta*kld
    
    

batch_size = 32
learning_rate = 3e-4 #Karpathy Constant
num_epochs = 100
beta = 0.05


#vae = VariationalAutoencoder(16, 5)

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


def early_stopping(train_loss, validation_loss, min_delta, tolerance):

    counter = 0
    if (validation_loss - train_loss) > min_delta:
        counter +=1
        if counter >= tolerance:
          return True
      
def early_stopping2(train_loss, prev_train_loss, min_delta, tolerance):
    counter = 0
    if  prev_train_loss < train_loss:
        counter +=1
        if counter >= tolerance:
          return 1
      
      
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
       

    def early_stop(self, validation_loss, prev_train, train):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        elif prev_train < train:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
        



train_loss_avg = []
val_loss = []
mse = []
kld = []
vae = VariationalAutoencoder(16,5)
summary(vae, input_size=(1, 448, 192))
early_stopper = EarlyStopper(patience=5, min_delta=0.02)

vae.train()
prev_train_loss = 1e99
for epoch in range(num_epochs):
    
    
    mse.append(0)
    kld.append(0)
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
        mse[-1], kld[-1] = vae_loss2(batch,recon,mu,logvar,beta)
        num_batches += 1
    with torch.no_grad():
        val_batches = 0
        val_loss.append(0)
        for val_batch in validation_dataloader:
            val_recon, val_mu, val_logvar , _ = vae(val_batch)
            vali_loss = vae_loss(val_batch,val_recon,val_mu,val_logvar,beta)
            val_loss[-1] += vali_loss.item()
            val_batches += 1
            
        val_loss[-1] /= num_batches
    mse[-1] /= num_batches
    kld[-1] /= num_batches
    train_loss_avg[-1] /= num_batches
    
    if early_stopper.early_stop(val_loss[-1], prev_train_loss, train_loss_avg[-1] ):
        print('Early Stopper Activated at %f' %epoch)
        break
    prev_train_loss = train_loss_avg[-1]  
   
    
   
    print('Epoch [%d / %d] average training error: %f' % (epoch+1, num_epochs, train_loss_avg[-1]))
    

meshDict, time, varDict = h5_load('CYLINDER.h5')



with torch.no_grad():
    
    instant = iter(train_dataloader)
    batch = next(instant)
    
    recon_instant = vae(batch)
    
    recon_instant = np.asanyarray(recon_instant[0])
    
    recon_instant = recon_instant[0,:,:]
    recon_instant = masc2(torch.tensor(recon_instant))
    _, mu, _, z = vae(batch)
    corr = np.corrcoef(z,rowvar=False)
    detR = np.linalg.det(corr)
    detR = np.round(detR * 100, 2)
    print(detR)
    plt.figure('B')
    plt.plot(range(epoch+1),mse,label='MSE')
    plt.plot(range(epoch+1),kld,label='KLD')
    plt.xlabel("Epochs")
    plt.ylabel("Loss Function")
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()
    
    plt.figure('A')
    plt.plot(range(epoch+1),train_loss_avg,label='Train Loss')
    plt.plot(range(epoch+1),val_loss,label='Validation Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss Function")
    plt.legend(loc= 'upper right')
    plt.grid(True)
    plt.show()
  

   
recon_instant =  torch.reshape(recon_instant,[89351,1])

varDict.update({'NEW2': {'point':varDict['VELOX'].get('point'),'ndim':varDict['VELOX'].get('ndim'),'value': recon_instant} })
torch.save(vae.state_dict(), './beta_vae2' )
plotSnapshot(meshDict,varDict,vars=['NEW2'],instant=0,cmap='jet',cpos='xy')








#This part of the code can be found in the other files


'''
meshDict, time, varDict = h5_load('CYLINDER.h5')

vae.load_state_dict(torch.load('beta_vae2'))
vae.eval()


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

varDict.update({'NEW2': {'point':varDict['VELOX'].get('point'),'ndim':varDict['VELOX'].get('ndim'),'value': snap3} })
plotSnapshot(meshDict,varDict,vars=['NEW2'],instant=0,cmap='jet',cpos='xy')

#varDict.update({'NEW2': {'point':varDict['VELOX'].get('point'),'ndim':varDict['VELOX'].get('ndim'),'value': snap2} })
#plotSnapshot(meshDict,varDict,vars=['NEW2'],instant=0,cmap='jet',cpos='xy')

'''
'''

#vae = VariationalAutoencoder(16, 5)
#vae.load_state_dict(torch.load('beta_vae2'))
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
        
    
    
    ek_mean = (1-torch.mean(ek))*100
    print(ek_mean)   



'''



'''


dataset = Load_h5_file('CYLINDER.h5')
energy_dataset = DataLoader(dataset, batch_size=151 ,shuffle=False)

train_loss_avg = []
val_loss = []
test_loss = []
detR = []
#early_stopper = EarlyStopper(patience=5, min_delta=0.02)
prev_train_loss = 1e9
beta = 2.5e-2
lat_dims = [20 ,17, 15,13,12,10, 5, 4, 2]
betas = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 2.5e-2,5e-2]
ek_mean = torch.zeros(len(betas))
ek = torch.zeros(151,len(betas))
lat_dim = 10
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
    
    vae.eval()
    with torch.no_grad():
        test_batches = 0
        for energy_batch in energy_dataset:
            test_recon, test_mu, test_logvar, lat_vec_or = vae(energy_batch)
            test_loss1 = vae_loss(energy_batch,test_recon,test_mu, test_logvar,beta)
            test_loss[-1] += test_loss1.item()
            coefs = np.corrcoef(lat_vec_or,rowvar=False)
            detR_ = np.linalg.det(coefs)
            detR[j] += np.round(detR_ * 100, 2)
            test_batches += 1
        detR[j] /= test_batches
        test_loss[-1] /= test_batches
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
            ek[i,j] = torch.sum((x-x_recon)**2)/torch.sum(x**2)
        ek_mean[j] = (1-torch.mean(ek[:,j]))*100
    
        
   
    print('Epoch [%d / %d] average training error: %f' % (epoch+1, num_epochs, train_loss_avg[-1]))
    
fig = plt.figure('A')
plt.plot(betas,ek_mean,marker = ".")
plt.xlabel('Beta')
plt.ylabel('$E_{k}$ % ')
plt.grid(True)
plt.show()

fig = plt.figure('b')
plt.plot(betas,detR, marker = ".")
plt.xlabel('beta')
plt.ylabel('detR')
plt.grid(True)
plt.show()
'''
exit(0)
dataset = Load_h5_file('CYLINDER.h5')
energy_dataset = DataLoader(dataset, batch_size=151 ,shuffle=False)

train_loss_avg = []
val_loss = []
test_loss = []
detR = []
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
    
        

x = np.linspace(0,10,1)
y= np.linspace(0,10,1)

   
    
    
plt.figure('A')
plt.plot(betas,ek_mean[0,:],color='r',marker='o', label = 'latent dimension = 5')
plt.plot(betas,ek_mean[1,:],color='g',marker='o',label = 'latent dimension = 15')
plt.plot(betas,ek_mean[2,:],color='b',marker='o',label = 'latent dimension = 20')
plt.xlabel("beta")
plt.ylabel("$E_{k}$")
plt.legend(loc='lower left')
plt.grid(True)
plt.show()
