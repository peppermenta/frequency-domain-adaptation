import numpy as np
import torchvision.datasets

import utils

class DFTFolderDataset(torchvision.datasets.ImageFolder):
  '''
  Generic class to load images from a given folder as a torch dataset
  
  Images are DFT transformed and pre-processed according to the paper https://arxiv.org/pdf/2006.15476.pdf
  for implemented NNs in the fourier domain
  '''

  def __init__(self, root:str, transform=None, num_slicing_levels=3):
    super().__init__(root, transform=transform)
    self.num_slicing_levels = num_slicing_levels

  def __getitem__(self, idx:int):
    img, label = super().__getitem__(idx)
    img_np = np.array(img)
    blocks = utils.make_blocks(img_np, num_slicing_levels=self.num_slicing_levels)
    dft_out = [utils.get_dft(b) for b in blocks]
    
    return dft_out, label
