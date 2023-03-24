import torch
from torch import nn
import numpy as np

IMAGE_SIZE = 222

def getDim(level):
    dimsize = IMAGE_SIZE / level
    if(dimsize % 2 == 1):
        dimsize += 1
    return dimsize / 2

def generateFilter(weights, level):
    filter_size = IMAGE_SIZE / level
    steps = weights.size(dim=0)
    output = torch.from_numpy(np.zeros((filter_size, filter_size)))
    current_dim = 1
    if(filter_size % 2 == 0):
        current_dim += 1
    for i in range(steps):
        current_layer = np.ones((current_dim, current_dim))
        if(i != 0):
            cut = np.ones((current_dim - 2, current_dim - 2))
            cut = np.pad(cut, ((1,1),(1,1)), 'constant')
            current_layer = current_layer - cut
        padding = ((filter_size - current_dim) // 2)
        current_layer = np.pad(current_layer, ((padding, padding), (padding,padding)), 'constant')
        current_layer = torch.from_numpy(current_layer)
        current_layer = torch.mul(current_layer, weights[i])
        output = torch.add(output, current_layer)
        current_dim += 2
    return output


class DftModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.rand(getDim(1))
        self.layer2_00 = torch.rand(getDim(2))
        self.layer2_01 = torch.rand(getDim(2))
        self.layer2_10 = torch.rand(getDim(2))
        self.layer2_11 = torch.rand(getDim(2))
        self.layer3_00 = torch.rand(getDim(3))
        self.layer3_01 = torch.rand(getDim(3))
        self.layer3_02 = torch.rand(getDim(3))
        self.layer3_10 = torch.rand(getDim(3))
        self.layer3_11 = torch.rand(getDim(3))
        self.layer3_12 = torch.rand(getDim(3))
        self.layer3_20 = torch.rand(getDim(3))
        self.layer3_21 = torch.rand(getDim(3))
        self.layer3_22 = torch.rand(getDim(3))

    def forward(self, x):
        l1Filter = generateFilter(self.layer1,1)

        block_size = IMAGE_SIZE // 2
        l2Filter = torch.zeros(IMAGE_SIZE, IMAGE_SIZE)
        
        l2_00 = generateFilter(self.layer2_00,2)
        l2Filter[:block_size, :block_size] = l2_00
        l2_01 = generateFilter(self.layer2_01,2)
        l2Filter[:block_size, block_size:] = l2_01
        
        l2_10 = generateFilter(self.layer2_10,2)
        l2Filter[block_size:, :block_size] = l2_10
        l2_11 = generateFilter(self.layer2_11,2)
        l2Filter[block_size:, block_size:] = l2_11

        block_size = IMAGE_SIZE // 3
        l3Filter = torch.zeros(IMAGE_SIZE, IMAGE_SIZE)
        
        l3_00 = generateFilter(self.layer3_00,3)
        l3Filter[:block_size, :block_size] = l3_00
        l3_01 = generateFilter(self.layer3_01,3)
        l3Filter[:block_size, block_size:block_size * 2] = l3_01
        l3_02 = generateFilter(self.layer3_02,3)
        l3Filter[:block_size, block_size * 2:block_size * 3] = l3_02

        l3_10 = generateFilter(self.layer3_10,3)
        l3Filter[block_size:block_size * 2, :block_size] = l3_10
        l3_11 = generateFilter(self.layer3_11,3)
        l3Filter[block_size:block_size * 2, block_size:block_size * 2] = l3_11
        l3_12 = generateFilter(self.layer3_12,3)
        l3Filter[block_size:block_size * 2, block_size * 2:block_size * 3] = l3_12

        l3_20 = generateFilter(self.layer3_20,3)
        l3Filter[block_size * 2:, :block_size] = l3_20
        l3_21 = generateFilter(self.layer3_21,3)
        l3Filter[block_size * 2:, block_size:block_size * 2] = l3_21
        l3_22 = generateFilter(self.layer3_22,3)
        l3Filter[block_size * 2:, block_size * 2:block_size * 3] = l3_22

        filterBank = torch.zeros(IMAGE_SIZE, IMAGE_SIZE, 3)
        filterBank[:,:,0] = l1Filter
        filterBank[:,:,1] = l2Filter
        filterBank[:,:,2] = l3Filter
        
        out = torch.mul(x, filterBank)
        