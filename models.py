#Setup
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

import math
from torchsummary import summary
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


class DragonNet(nn.Module):
    """
    3-headed dragonet architecture 
    
    Args:
        in_channels: number of features of the input image ("depth of image")
        hidden_channels: number of hidden features ("depth of convolved images")
        out_features: number of features in output layer
    """
    
    def __init__(self, in_features, hidden_channels, out_features):
        super(DragonNet, self).__init__()

        #initialize placeholder tensors for weight and bias 
        self.weight = torch.Tensor(in_features,out_features)
        #self.bias = torch.Tensor(out_features) (why do they do not do this)

        #representation layers 3 : block1 
        # units in kera = out_features 
        self.rep_1 = nn.Linear()
        self.rep_2 = nn.Linear()
        self.rep_3 = nn.Linear()
        
    def init_params(self, std=0.1):
        """
        Initialize layer parameters. Sample weight from Gaussian distribution
        and bias uniform distribution.
        
        Args:
            std: Standard deviation of Gaussian distribution (default: 0.1)
        """
        self.weight = std*torch.randn_like(self.weight)
        self.bias = torch.rand_like(self.bias)


    def forward(self, x):
        # First convolutional layer
        x = self.conv1(x)
        # Activation function
        x = self.relu1(x)
        # Max pool
        x = self.max_pool1(x)
        # Second convolutional layer
        x = self.conv2(x)
        # Activation function
        x = self.relu2(x)
        # Max pool
        x = self.max_pool2(x)
        # Flatten
        x = x.view(x.size(0), -1)
        # Fully connected layer
        x = self.fc(x)
        return x


def make_dragonnet(input_dim, reg_l2): 
    
    """
    Args:
        input_dim (_type_): dimension of input data (X)
        reg_l2 (_type_): regularization type 
    """

    #t_l1 = 0         these were not used so we did not keep it 
    #t_l2 = reg_l2

    # assuming input dimension is an intenger
    # input dimension is the number of features per sample in X  
    inputs = torch.Tensor(input_dim)

    
    
    #Representation block 
    # creating representation layer z(X)
    x = torch.nn.Linear(out_features=200, )


    #-----3 headed architecture 
    # Head 1: Propensity score/ tratement prediction head 


    # Head 1 hypothesis  
    
    # Head 2 hypothesis 
     

    # Layer 2 
    # Layer 3
     

