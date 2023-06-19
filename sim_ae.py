
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import Load_h5_file, pre_proces
from tools import reshape_var, h5_load, plotSnapshot
import numpy as np

padded_tensor = torch.ones([449, 199])
def masc2(cropped_tensor,new_dim =[449,199]):    #aquesta funcio ens permet reconstruir l'imatge a les dimensions necessaries (499x199)
    height = 448                                 #per poder plotjear amb pyvista
    width = 192

    # Calculate the coordinates of the upper-left corner of the crop
    y1 = 0
    x1 = 0
    padded_cropped_tensor = F.pad(cropped_tensor, (0, padded_tensor.shape[1]-width, 0, padded_tensor.shape[0]-height), mode='constant', value=0)
    print(padded_cropped_tensor.shape)
    return(padded_cropped_tensor)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels= 1, out_channels = 16, kernel_size=3, stride=1, padding=1)
        self.max = nn.MaxPool2d(2,2)
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride= 1, padding= 1)
        #self.max = nn.MaxPool2d(2,2)
        
        self.conv3 = nn.Conv2d( in_channels= 32, out_channels= 64, kernel_size= 3, stride= 1, padding= 1 )
        #self.max = nn.MaxPool2d(2,2)
        
        self.conv4 = nn.Conv2d( in_channels=64, out_channels= 128, kernel_size=3, stride=1, padding= 1 )
        #self.max = nn.MaxPool2d(2,2)
        
        self.conv5 = nn.Conv2d( in_channels=128, out_channels=256, kernel_size= 3, stride=1, padding=1)
        #self.max = nn.MaxPool2d(2,2)
        #self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size= 3, stride=1, padding=1)
        #self.max = nn.MaxPool2d(2,2)
        
        self.lin1 = nn.Linear(in_features=256*14*6, out_features = 128)
        self.lin2 = nn.Linear(in_features=128, out_features= 10)
        
    def forward(self,x):
        
        print('Enc_lay:0 ', x.shape)
        out = torch.tanh(self.conv1(x))
        out = self.max(out)
        print('Enc_lay:1 ', out.shape)
        
        out = torch.tanh(self.conv2(out))
        out = self.max(out)
        print('Enc_lay:2 ', out.shape)
        
        out = torch.tanh(self.conv3(out))
        out = self.max(out)
        print('Enc_lay:3 ', out.shape)
        
        out = torch.tanh(self.conv4(out))
        out = self.max(out)
        print('Enc_lay:4 ', out.shape)
        
        out = torch.tanh(self.conv5(out))
        out = self.max(out)
        print('Enc_lay:5 ', out.shape)
        
       # out = torch.tanh(self.conv6(out))
        #out = self.max(out)
        print('Enc_lay:6 ', out.shape)
        
        out = out.view(out.size(0), -1)
        print('Enc_lay:6 resh ', out.shape)
        out = torch.tanh(self.lin1(out))
        print('Enc_lay:6 (lin) ', out.shape)
        
        out = self.lin2(out)  
        print('Enc_lay: end (lin) ', out.shape)      
        
        return out
    
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        #in features = 20
        self.lin2 = nn.Linear(in_features=10, out_features=128)
        
        self.lin1 = nn.Linear(in_features=128, out_features=256*14*6)
        
        self.conv6 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size= 3, stride=1, padding=1)
        self.up = nn.UpsamplingNearest2d(scale_factor= 2)
        
        self.conv5 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size= 3, stride=1, padding=1)
        
        self.conv4 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size= 3, stride=1, padding=1)
        
        self.conv3 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size= 3, stride=1, padding=1)
        
        self.conv2 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size= 3, stride=1, padding=1)
        
        #self.conv1 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size= 3, stride=1, padding=1)
        
    def forward(self, x):
        print('Dec_lay:0 (lin) ', x.shape)
        out = torch.tanh(self.lin2(x))
        print('Dec_lay:1 (lin) ', out.shape)
        out = torch.tanh(self.lin1(out))
        print('Dec_lay:2 (lin) ', out.shape)
        
        out = out.view(out.size(0), 256, 14, 6)
        print('Dec_lay: resh (lin) ', out.shape)
        
        out = self.up(out)
        out = torch.tanh(self.conv6(out))
        print('Dec_lay:4  ', out.shape)
        
        out = self.up(out)
        out = torch.tanh(self.conv5(out))
        print('Dec_lay:6  ', out.shape)
        
        out =self.up(out)
        out = torch.tanh(self.conv4(out))
        print('Dec_lay:7  ', out.shape)
        
        out = self. up(out)
        out = torch.tanh(self.conv3(out))
        print('Dec_lay:8  ', out.shape)
        
        out = self.up(out)
        out = torch.tanh(self.conv2(out))
        print('Dec_lay: 9  ', out.shape)
        
        #out = self.up(out)
        #out = torch.tanh(self.conv1(out))
        #print('Dec_lay: end  ', out.shape)
        
        return out
        
 
 
class Encoder1(nn.Module):
    def __init__(self):
        super(Encoder1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels= 1, out_channels = 16, kernel_size=4, stride=2, padding=1)
       
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride= 2, padding= 1)
        #self.max = nn.MaxPool2d(2,2)
        
        self.conv3 = nn.Conv2d( in_channels= 32, out_channels= 64, kernel_size= 4, stride= 2, padding= 1 )
        #self.max = nn.MaxPool2d(2,2)
        
        self.conv4 = nn.Conv2d( in_channels=64, out_channels= 128, kernel_size=4, stride=2, padding= 1 )
        #self.max = nn.MaxPool2d(2,2)
        
        self.conv5 = nn.Conv2d( in_channels=128, out_channels=256, kernel_size= 4, stride=2, padding=1)
        #self.max = nn.MaxPool2d(2,2)
        #self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size= 3, stride=1, padding=1)
        #self.max = nn.MaxPool2d(2,2)
        
        self.lin1 = nn.Linear(in_features=256*14*6, out_features = 128)
        self.lin2 = nn.Linear(in_features=128, out_features= 5)
        
    def forward(self,x):
        
        #print('Enc_lay:0 ', x.shape)
        out = torch.tanh(self.conv1(x))
        
       # print('Enc_lay:1 ', out.shape)
        
        out = torch.tanh(self.conv2(out))
        
        #print('Enc_lay:2 ', out.shape)
        
        out = torch.tanh(self.conv3(out))
        
        #print('Enc_lay:3 ', out.shape)
        
        out = torch.tanh(self.conv4(out))
        
        #print('Enc_lay:4 ', out.shape)
        
        out = torch.tanh(self.conv5(out))
        
        #print('Enc_lay:5 ', out.shape)
    
        
        out = out.view(out.size(0), -1)
        #print('Enc_lay:6 resh ', out.shape)
        out = torch.tanh(self.lin1(out))
        #print('Enc_lay:6 (lin) ', out.shape)
        
        out = self.lin2(out)  
        #print('Enc_lay: end (lin) ', out.shape)      
        
        return out
    
class Decoder1(nn.Module):
    def __init__(self):
        super(Decoder1, self).__init__()
        #in features = 20
        self.lin2 = nn.Linear(in_features=5, out_features=128)
        
        self.lin1 = nn.Linear(in_features=128, out_features=256*14*6)
        
        self.conv6 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size= 4, stride=2, padding=1)
       
        
        self.conv5 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size= 4, stride=2, padding=1)
        
        self.conv4 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size= 4, stride=2, padding=1)
        
        self.conv3 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size= 4, stride=2, padding=1)
        
        self.conv2 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size= 4, stride=2, padding=1)
        
        self.flat = nn.Flatten()
        
        
        #self.conv1 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size= 3, stride=1, padding=1)
        
    def forward(self, x):
        #print('Dec_lay:0 (lin) ', x.shape)
        out = torch.tanh(self.lin2(x))
        #print('Dec_lay:1 (lin) ', out.shape)
        out = torch.tanh(self.lin1(out))
        #print('Dec_lay:2 (lin) ', out.shape)
        
        out = out.view(out.size(0), 256, 14, 6)
       # print('Dec_lay: resh (lin) ', out.shape)
        
        
        out = torch.tanh(self.conv6(out))
       # print('Dec_lay:4  ', out.shape)
        
        
        out = torch.tanh(self.conv5(out))
       # print('Dec_lay:6  ', out.shape)
        
        
        out = torch.tanh(self.conv4(out))
      #  print('Dec_lay:7  ', out.shape)
        
        
        out = torch.tanh(self.conv3(out))
       # print('Dec_lay:8  ', out.shape)
        
    
        out = torch.tanh(self.conv2(out))
        #print('Dec_lay: 9  ', out.shape)
        
        #out = self.up(out)
        #out = torch.tanh(self.conv1(out))
        #print('Dec_lay: end  ', out.shape)
        
        return out
        
    



class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder1()
        self.decoder = Decoder1()
    
    

    def forward(self, x):

        out = self.encoder(x)
        
        x_recon = self.decoder(out) #reconstruïm
       
        return x_recon
    

batch_size = 32
    
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


ae = Autoencoder()

optimizer = torch.optim.Adam(params=ae.parameters(), lr=1e-3, weight_decay=1e-5)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.1, last_epoch=- 1, verbose=False)
ae_loss = nn.MSELoss()
meshDict, time, varDict = h5_load('CYLINDER.h5')
#retallem la malla per poder plotejar



ae.train()
num_epochs = 100



for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch_recon = ae(batch) #forward pass
            
        loss = ae_loss(batch_recon, batch) # get los

            
        optimizer.zero_grad()
    
        loss.backward()        # backpropagation
            # one step of the optmizer (using the gradients from backpropagation)
        optimizer.step()          #update weights
    
    print('epoch %d' %epoch)


torch.save(ae.state_dict(), './sim_ae' )
instant = iter(train_dataloader)

#EVALUATING THE MODEL
ae.load_state_dict(torch.load('sim_ae'))
ae.eval()
meshDict, time, varDict = h5_load('CYLINDER.h5')
_, mit_value, _  = pre_proces(varDict,time)

mit_tensor = mit_value[:,1]
mit_tensor = np.resize(mit_tensor,[89351,1])

mit_tensor = torch.tensor(mit_tensor)
with torch.no_grad():
   
    snap3 = np.float32([0,0,0,0,1])
    snap3 = np.tile(snap3,[32,1])
    snap3 = torch.from_numpy(snap3)
    snap3 = ae.decoder(snap3)   
    snap3 = snap3[1,:,:,:]
    snap3 = masc2(snap3)
    snap3 = snap3.resize_([89351,1]) 
   
   
    batch = next(instant)
    snap2 = batch[0,0,:,:]
    recon_instant = ae(batch)
    recon_instant = np.asanyarray(recon_instant[0])
    recon_instant = masc2(torch.tensor(recon_instant))
    recon_instant = recon_instant[0,:,:]
    recon_instant =  torch.reshape(recon_instant,[89351,1]) + mit_tensor
    snap2 = masc2(snap2)
    snap2 = torch.reshape(snap2, [89351,1]) + mit_tensor
    recon_instant = ae(batch)
    recon_instant = np.asanyarray(recon_instant[0])
    recon_instant = masc2(torch.tensor(recon_instant))
    recon_instant = recon_instant[0,:,:]
    recon_instant =  torch.reshape(recon_instant,[89351,1]) 
    
    print(torch.max(recon_instant), torch.min(recon_instant))
    
recon_instant =  torch.reshape( recon_instant,[89351,1] )
recon_instant += mit_tensor
varDict.update({'NEW2': {'point':varDict['VELOX'].get('point'),'ndim':varDict['VELOX'].get('ndim'),'value': snap3} })
plotSnapshot(meshDict,varDict,vars=['NEW2'],instant=0,cmap='jet',cpos='xy')
varDict.update({'NEW2': {'point':varDict['VELOX'].get('point'),'ndim':varDict['VELOX'].get('ndim'),'value': snap2} })
plotSnapshot(meshDict,varDict,vars=['NEW2'],instant=0,cmap='jet',cpos='xy')
