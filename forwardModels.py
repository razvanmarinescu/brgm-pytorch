from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F

#from tf_slice_assign import slice_assign

import cv2
import numpy as np
import pdb
import scipy.ndimage.morphology
#import tf_bicubic_downsample

class ForwardAbstract(ABC):
  def __init__(self):
    pass

  @abstractmethod
  def __call__(self, x):
    return x

  def calcMaskFromImg(self, img):
    pass
  
  def initVars(self):
    pass
  
  def getVars(self):
    return []

class ForwardNone(ForwardAbstract):
  def __init__(self):
    pass

  def __call__(self, x):
    return x


class ForwardDownsample(ForwardAbstract):
  def __init__(self, factor):
    self.factor = factor

  # resolution of input x can be anything, but aspect ratio should be 1:1
  def __call__(self, x): 
    #x = tf.reshape(x, (x.shape[0], x.shape[2], x.shape[3], x.shape[1]))
    #x = tf_bicubic_downsample.apply_bicubic_downsample(x, self.filter, self.factor) # BHWC format 
    #x = tf.reshape(x, (x.shape[0], x.shape[3], x.shape[1], x.shape[2]))
    x_down = F.interpolate(x, scale_factor=self.factor, mode='bicubic',recompute_scale_factor=True,align_corners=False) # BCHW 
    return x_down

class ForwardFillMask(ForwardAbstract):
  """ Takes an image with a filled-in mask (already baked in the image), and derived the mask automatically by taking the histogram over voxels. Supports free-form masks """
  def __init__(self, device):
    self.device = device
    self.mask = None

  def calcMaskFromImg(self, img):
    nrBins = 256
    grayImg = np.squeeze(np.mean(img, axis=1))
    gray1D = grayImg.ravel() # eliminate the first bin with black pixels, as it doesn't work for brains (wrong mask is estimated)
    hist,bins = np.histogram(gray1D, nrBins, [-1,1])
    print(hist, bins)
    hist = hist[1:]    
    bins = bins[1:]

    maxIndex = np.argmax(hist)
    
    #print('bins[maxIndex]', bins[maxIndex])

    self.mask = np.abs(grayImg - bins[maxIndex]) < (3.0/nrBins)
    self.mask = torch.tensor(scipy.ndimage.morphology.binary_opening(self.mask, iterations=3), dtype=torch.bool, device=self.device)
    #print('type(self.mask)', type(self.mask))
    self.mask = torch.repeat(torch.reshape(self.mask, (1, 1, *self.mask.shape)), img.shape[1], axis=1)
    #print('type(self.mask)', type(self.mask))
  
  def __call__(self, x): 
    if (self.mask is None):
      self.mask = torch.zeros(x.shape, dtype=torch.bool, device=self.device)
    #print('type(self.mask)', type(self.mask))

    whiteFill = torch.ones(x.shape, device=self.device, dtype=x.dtype)
    xFill = torch.where(self.mask, whiteFill, x) # if true, then whiteFill, else x

    return xFill


