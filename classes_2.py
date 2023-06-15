
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np




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
        
