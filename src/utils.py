import numpy as np
from scipy import fftpack as sfft

def make_blocks(img, num_slicing_levels=3):
  '''
  Divide image into 1, 4, .... equal blocks, upto num_slicing_levels**2
  Following the procedure in https://arxiv.org/pdf/2006.15476.pdf

  Parameters
  ---------------
  img: np.ndarray
    The image to divide into blocks. Assumed to be a square, grayscale image
  num_slicing_levels: int, optional
    The number of levels of slicing to perform. The default is 3

  Returns
  ---------------
  out: List[np.ndarray]
    The list of individual blocks from each level of slicing
  '''

  if(len(img.shape)!=2):
    raise NotImplementedError
  if(img.shape[0]!=img.shape[1]):
    raise Exception

  out = []
  N = img.shape[0]
  for level in range(1, num_slicing_levels+1):
    block_size = N//level

    for i in range(level):
      for j in range(level):
        out.append(img[i*block_size:(i+1)*block_size,j*block_size:(j+1)*block_size])

  return out


def get_dft(img):
  '''
  Return the magniude of shifted dft of the image. DFT is shifted to bring DC component to centre

  Parameters
  -----------------
  img: np.ndarray
    Image to apply DFT on. Currently only supports 2D grayscale images

  Returns
  -------------
  out: np.ndarray
    Magnitued of Shifted DFT
  '''
  
  dft = sfft.fft2(img)
  shifted_dft = sfft.fftshift(dft)

  return np.abs(shifted_dft)

def merge_dfts(dft_list, num_slicing_levels):
  '''
  Given a list of DFTs for individual blocks at different levels, 
  place into a single multidimensional tensor. Each channel comprises DFT blocks of
  a particular slicing level

  Parameters
  ------------------
  dft_list: List[np.ndarray]
    The list of DFTs of individual image blocks
  num_slicing_levels: int
    The number of levels of slicing carried out
  
  Returns
  ---------------------
  out: np.ndarray
    Single array containing the DFTs of all blocks
  '''

  #1st block contains the DFT of full image
  N, _ = dft_list[0].shape
  out = np.zeros((N,N,num_slicing_levels))
  curr_idx = 0

  for level in range(1, num_slicing_levels+1):
    curr_block_size = N//level
    for i in range(level):
      for j in range(level):
        out[i*curr_block_size:(i+1)*curr_block_size, j*curr_block_size:(j+1)*curr_block_size, level-1] = dft_list[curr_idx]
        curr_idx += 1

  return out