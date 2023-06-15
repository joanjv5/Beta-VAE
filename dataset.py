import os
import torch
from tools import reshape_var, h5_load, vtkh5_save, plotSnapshot
from torch.utils.data import Dataset
import torchvision
from skimage import io
from torchvision import datasets, transforms

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 
from matplotlib.pyplot import imshow
import numpy as np
import torchvision.transforms.functional as TF
import torch.nn.functional as F


import numpy as np

from tools import reshape_var, h5_load, plotSnapshot


meshDict, time, varDict = h5_load('CYLINDER.h5')


#selecccionem tots els snapshots tan de veloc com velox

def pre_proces(varDict,time):
    u_t0 = varDict['VELOC']['value'][:,:]  #si al segon lloc posem : tenim tots els intants 
    u_t0X = varDict['VELOX']['value'][:,:]           #al vector u_t0
    u_t0_sum = np.zeros(u_t0.shape[0],np.float32)
    u_t0X_sum = np.zeros(u_t0X.shape[0],np.float32)
    for i in range(len(time)):      #calculem la mitjana temporal
        u_t0_sum [:] = u_t0_sum [:] + u_t0[:,i]
        u_t0X_sum [:] =  u_t0X_sum [:] + u_t0X[:,i]
    mit = np.transpose(np.tile(u_t0_sum,(u_t0.shape[1],1)))/len(time)
    mitX = np.transpose(np.tile(u_t0X_sum,(u_t0X.shape[1],1)))/len(time)
    mean = mitX
    mit = u_t0 - mit
    mitX = u_t0X - mitX
    
    
    maxi = 1e-20
    for i in range(len(time)):
        maxi =   max(abs(mitX[:,i])) if maxi <  max(abs(mitX[:,i]))  else maxi #calculem el valor maxim entre tots els instants, per poder normalitzar els valors entre -1 ,1
        
       
    return maxi, mean, mitX



class Load_h5_file(Dataset):
    def __init__(self, root_dir, transform = None):
        _, self.time, self.varDict = h5_load('CYLINDER.h5')
        self.max, self.mit, self.mitX = pre_proces(self.varDict, self.time)
        #self.varDict.update({'VELOM': {'point':varDict['VELOC'].get('point'),'ndim':varDict['VELOC'].get('ndim'),'value': self.mit} })
        self.varDict.update({'VELOMX': {'point':varDict['VELOX'].get('point'),'ndim':varDict['VELOX'].get('ndim'),'value': self.mitX} })
        #print(self.varDict.keys())
        self.transform = transform
        
    def __len__(self):
        
        return len(self.time)
    
    def __getitem__(self, index):
        snap = torch.Tensor(self.varDict['VELOMX']['value'][:,index]/self.max)   #carreguem un instant de temps
        snap = snap.resize_(1,499,199)
        height = 448
        width = 192

        # Introduim les coordenades a partir de les quals farme el crop
        y1 = 0
        x1 = 0

        # apliquem el crp
        cropped_tensor = TF.crop(snap, top=y1, left=x1, height=height, width=width)
        
        if self.transform:
            cropped_tensor = self.transform(cropped_tensor)

       
        # print('ola', cropped_tensor.shape)  # Output: torch.Size([448, 192]) 
        return cropped_tensor
        
        
        
        
        
        '''
        resized_snap = snap.resize_(449,199) #posem el vector velocitat en forma de matriu
                                             #per poder alientar la xarxa neruonal
        croped_snap = resized_snap[1:449,3:195] #d'aquesta manera tindrem una matriu de 448 x 192, que 
                                                #a mesura que anem passant per les convents tindrem divisions exactes
        croped_snap = croped_snap.resize_(1,448,192) #li afegim el número de canals a la primera dimensió pq pytorch entingui el format
        #print(croped_snap.shape)
        #print(croped_snap)
        #print(self.max)
        '''
        #return croped_snap


#dataset = Load_h5_file('CYLINDER.h5')

'''

r_v = dataset[4].resize_(449,199)
r = r_v.resize_(89351,1)

meshDict, time, varDict = h5_load('CYLINDER.h5')
varDict.update({'NEW': {'point':varDict['VELOX'].get('point'),'ndim':varDict['VELOX'].get('ndim'),'value': r} })
plotSnapshot(meshDict,varDict,vars=['NEW'],instant=0,cmap='jet',cpos='xy')
'''