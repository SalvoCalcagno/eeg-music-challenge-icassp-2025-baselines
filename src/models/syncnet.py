import torch.nn.functional as F
import torch
import scipy.io
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim

import numpy as np

class Model(nn.Module):
    def __init__(self,args):
        super(Model, self).__init__()
        args_defaults=dict(
            num_channels=8, 
            input_size=1266, 
            num_classes=2, 
            num_filters=1, 
            filter_width = 40, 
            pool_size=40
        )
        for arg,default in args_defaults.items():
            setattr(self, arg, args[arg] if arg in args and args[arg] is not None else default)
        self.num_filters = self.num_filters 
        self.num_channels = self.num_channels
        self.filter_width = self.filter_width
        
        self.b = nn.Parameter(torch.FloatTensor(1,1,self.num_channels,self.num_filters).uniform_(-0.05, 0.05))
        self.bias = nn.Parameter(torch.FloatTensor(self.num_filters))
        self.omega = nn.Parameter(torch.FloatTensor(1,1,1,self.num_filters).uniform_(0, 1))
        self.cl_Wy = int(np.ceil(float(self.input_size)/float(self.pool_size)) * self.num_filters)
        
        if(self.filter_width%2 == 0):
            self.t = torch.FloatTensor(np.reshape(range(-int(self.filter_width/2),int(self.filter_width/2)),[1,int(self.filter_width),1,1]))
        else:
            self.t = torch.FloatTensor(np.reshape(range(-int((self.filter_width-1)/2),int((self.filter_width-1)/2) + 1),[1,int(self.filter_width) ,1,1]))
        self.t=nn.Parameter(self.t)
        self.phi_ini = nn.Parameter(torch.FloatTensor(1,1,self.num_channels, self.num_filters).normal_(0,0.05))
        self.beta = nn.Parameter(torch.FloatTensor(1,1,1,self.num_filters).uniform_(0, 0.05))
        
        ## Only stride and dilation values of 1 are supported. If you use different values, padding values wont be correct
        P = ((self.input_size-1)-self.input_size + (self.filter_width-1))
        if(P%2 == 0):
            self.padding = (P//2, P//2 + 1)
        else:
            self.padding = (P//2, P//2)
        
        self.pool = nn.MaxPool2d((1, self.pool_size), stride = (1, self.pool_size))
        self.classifier = nn.Linear(self.cl_Wy,self.num_classes)
    
    def forward(self,X):
        #X must be in the form of BxCx1xT or BxCxT
        self.W_osc = torch.mul(self.b,torch.cos(self.t*self.omega + self.phi_ini))
        self.W_decay = torch.exp(-torch.pow(self.t,2)*self.beta)
        self.W = torch.mul(self.W_osc,self.W_decay)
        self.W = self.W.view(self.num_filters,self.num_channels,1,self.filter_width)
        if(len(X.size()) == 3):
            X = X.unsqueeze(2)
        
        res = F.conv2d(F.pad(X,self.padding,"constant", 0).float(),self.W.float(),bias = self.bias,stride=1 )
        res = self.pool(F.pad(res,self.padding, "constant", 0))
        res = res.view(-1,self.cl_Wy)
        self.beta = nn.Parameter(torch.clamp(self.beta, min=0))
        
        return self.classifier(F.relu(res)).squeeze()
