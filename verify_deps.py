import torch
import cv2
import numpy
import scipy
import matplotlib
import sklearn
import natsort
print(torch.__version__)  # Should print 1.7.0 or 1.7.0+cu110
print(torch.cuda.is_available())  # Should print True if GPU is set up