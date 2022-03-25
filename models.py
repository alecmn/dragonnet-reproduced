#Setup
from turtle import forward
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


# find out why this is neccessary 
# not dicussed 
class EpsilonLayer(nn.Module): 
    def __init__(self): 
        super().__init__()

        # building epsilon trainable weight 
        self.weights = nn.Parameter(torch.Tensor(1,1))

        #initializing weight parameter with RandomNormal
        nn.init.normal_(self.weights)
    
    def forward(self,inputs):
        return torch.mm(self.weights.T, torch.ones_like(inputs)[:, 0:1])

#weight initialzation function 
def weights_init(params):
    if isinstance(params, nn.Linear): 
        torch.nn.init.normal_(params.weight, mean=0.0, std=1.0)
        torch.nn.init.zero_(params.bias)
    

class DragonNet(nn.Module):
    """
    3-headed dragonet architecture 
    
    Args:
        in_channels: number of features of the input image ("depth of image")
        hidden_channels: number of hidden features ("depth of convolved images")
        out_features: number of features in output layer
    """
    
    def __init__(self, in_features, hidden_channels, out_features=[200, 100, 1]):
        super(DragonNet, self).__init__()

        
        #representation layers 3 : block1 
        # units in kera = out_features 
        
        self.representation_block = nn.Sequential(
            nn.Linear(in_features=in_features,out_features = out_features[0]), 
            nn.ELU(), 
            nn.Linear(in_features=out_features[0], out_features=out_features[0]),
            nn.ELU(),
            nn.Linear(in_features=out_features[0], out_features=out_features[0]), 
            nn.ELU()
        )

        # -----------Propensity Head 
        self.t_predictions = nn.Sequential(nn.Linear(in_features=out_features[0], out_features=out_features[2]), 
            nn.Sigmoid())


        # -----------t0 Head 
        self.t0_head = nn.Sequential(nn.Linear(in_features= out_features[0],output_features=out_features[1]),
            nn.ELU(),
            nn.nn.Linear(in_features= out_features[1],output_features=out_features[1]), 
            nn.ELU(), 
            nn.Linear(in_features= out_features[1],output_features=out_features[2])
            )


        # ----------t1 Head 
        self.t1_head = nn.Sequential(nn.Linear(in_features= out_features[0],output_features=out_features[1]),
            nn.ELU(),
            nn.nn.Linear(in_features= out_features[1],output_features=out_features[1]), 
            nn.ELU(), 
            nn.Linear(in_features= out_features[1],output_features=out_features[2])
            )

        self.epsilon = EpsilonLayer()
                
    def init_params(self, std=1):
        """
        Initialize layer parameters. Sample weight from Gaussian distribution
        and bias uniform distribution.
        
        Args:
            std: Standard deviation of Random normal distribution (default: 1)
        """
        self.representation_block.apply(weights_init)
        

    def forward(self, input):
        x = self.representation_block(input)
        
        #------propensity scores 
        propensity_head = self.t_predictions(x)
        epsilons = self.epsilon(propensity_head)

        #------t0
        t0_out = self.t0_head(x)

        #------t1
        t1_out = self.t1_head(x)
        
        return propensity_head, t0_out, t1_out, epsilons


