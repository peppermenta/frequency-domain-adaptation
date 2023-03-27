import torch
from torch import nn
import numpy as np

IMAGE_SIZE = 222

def getDim(level):
    dimsize = IMAGE_SIZE / level
    if(dimsize % 2 == 1):
        dimsize += 1
    return int(dimsize // 2)

def generateFilter(weights, level):
    filter_size = int(IMAGE_SIZE / level)
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

class DFTModel(nn.Module):
    def __init__(self, num_classes):
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

        self.num_coeff = sum([(i**2)*getDim(i) for i in range(1,4)])
        self.num_classes = num_classes

        self.fc1 = nn.Linear(in_features=self.num_coeff, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=self.num_classes)

    def forward(self, x):
        '''
        Forward pass

        Parameters
        --------------------------
        x: torch.Tensor
            Expected to have shape (batch_size, num_slicing_levels, N, N) where N is the height/width of the original image

        Returns
        -------------------------
        out: torch.Tensor
            Scores for each predicted class
        '''
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
        
        # Multiplying the filter bank with each element in the batch
        out = torch.mul(x, filterBank)

        #Radially aggregating the outputs to form the inputs for the first FC layer
        batch_size = x.shape[0]
        coefficients = torch.zeros((batch_size, self.num_coeff))

        for batch_idx in range(batch_size):
            coeff_idx = 0
            for layer in range(1, 4):
                maxR = getDim(layer)
                block_size = int(IMAGE_SIZE // block_size)
                for block_i in range(layer):
                    for block_j in range(layer):
                        for r in range(maxR):
                            x_min = block_i*block_size+r
                            x_max = (block_i+1)*block_size-r
                            y_min = block_j*block_size+r
                            y_max = (block_j+1)*block_size-r
                            coefficients[batch_idx][coeff_idx] = torch.sum(out[batch_idx,layer-1,x_min:x_max,y_min:y_max])
                            coeff_idx += 1
                        
        out = self.fc1(coefficients)
        out = torch.nn.ReLU()(out)
        out = self.fc2(out)
        out = torch.nn.ReLU()(out)
        out = self.fc3(out)

        return out