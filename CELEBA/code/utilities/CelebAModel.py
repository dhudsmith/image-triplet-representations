import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict


    
    # Helper Functions
class ResNetBlock(nn.Module):
    r"""Resnet style block for predicting color shift masks on input images. 
    
    Args:
        num_in (int) - number of input channels (and output channels)
        num_features (int) - number of intermediate channels in resnet block
    """

    def __init__(self, num_in, num_mid, kernel_size=5):

        super(ResNetBlock, self).__init__()
        
        self.res = nn.Sequential(OrderedDict([
            # conv block 1
            ('conv0', nn.Conv2d(num_in, num_mid, kernel_size, stride=1,
                                padding=(kernel_size-1)//2, bias=False)),
            ('norm0', nn.BatchNorm2d(num_mid)),
            ('relu0', nn.ReLU(inplace=True)),
            # conv block 2
            ('conv1', nn.Conv2d(num_mid, num_in, kernel_size, stride=1,
                                padding=(kernel_size-1)//2, bias=False)),
            ('norm1', nn.BatchNorm2d(num_in)),

        ]))
        
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):
        # resnet style output: add input to features at relu
        return self.relu1(x + self.res(x))
    
class EncoderBlock(nn.Module):
    def __init__(self, n_in, n_mid, n_out, kernel_size=5):
        super(EncoderBlock, self).__init__()
        
        self.block = nn.Sequential(
            ResNetBlock(n_in, n_mid, kernel_size),
            nn.Conv2d(n_in, n_out, kernel_size=kernel_size, padding=(kernel_size-1)//2, stride=2),
            nn.BatchNorm2d(n_out),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.block(x)
        
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
class Encoder(nn.Module):
    def __init__(self, input_shape, n_mid=(64,32,16,16), n_res=32, kernel_size=5):
        super(Encoder, self).__init__()
        
        self.encoder = nn.Sequential(
            # start
            nn.Conv2d(input_shape[0], n_mid[0], kernel_size=7, padding=3, stride=2),
            #nn.MaxPool2d(kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(n_mid[0]),
            nn.ReLU(),
            # encoding blocks
            EncoderBlock(n_mid[0], n_res, n_mid[1], kernel_size),
            EncoderBlock(n_mid[1], n_res, n_mid[2], kernel_size),
            EncoderBlock(n_mid[2], n_res, n_mid[3], kernel_size),
            Flatten()
        )
        
    def forward(self, x):
        return self.encoder(x)
    
class TripletProbability(nn.Module):
  def __init__(self, alpha):
    super(TripletProbability, self).__init__()
    self.alpha=alpha

  def t_dist(self, d):
    return (1+d**2/self.alpha)**(-1*(self.alpha+1)/2)
  
  def forward(self, dAB, dAC):
    tAB = self.t_dist(dAB)
    tAC = self.t_dist(dAC)
    return tAB / (tAB + tAC)

class PairwiseDistance(nn.Module):
  def __init__(self, n_hid):
    super(PairwiseDistance, self).__init__()
    self.n_hid = n_hid

    self.f = nn.Sequential(
        nn.Linear(2*self.n_hid, self.n_hid),
        nn.ReLU(),
        nn.Linear(self.n_hid,1)
    )
  
  def forward(self, A, B):
    # A: [batch_size, n_hid]
    # B: [batch_size, n_hid]
    AB = torch.cat([A, B], dim=1)
    return torch.exp(self.f(AB).squeeze())
    # [batch_size]
    
    
class TripletNet(nn.Module):
    def __init__(self, n_hid=10, alpha=1, img_channels_shape=(1,64,64),n_mid=(64,32,16,16),n_res=32,kernel_size=5):
        super(TripletNet, self).__init__()

        self.n_hid=n_hid
        self.alpha=alpha

        # feature encoder
        # resuse for each input image
        self.encoder = Encoder(img_channels_shape, n_mid=n_mid, n_res=n_res, kernel_size=kernel_size)
        
        # distance computer: takes two samples and computes a distance
        # reuse this for pairs (A, B) and (A, C)
        # can make this more complex or more simple in future
        self.pairwise_distance = PairwiseDistance(n_hid=self.n_hid)

        # triplet probability computer defined in the class above
        self.triplet_probability = TripletProbability(self.alpha)
        
    def forward(self, A, B, C):
      # first compute all of the encodings
        A, B, C = [self.encoder(x) for x in (A,B,C)]
        
      # then get the pairwise distances
        dAB = self.pairwise_distance(A, B)
        dAC = self.pairwise_distance(A, C)
        
      # finally return the triplet probability
        return self.triplet_probability(dAB, dAC)