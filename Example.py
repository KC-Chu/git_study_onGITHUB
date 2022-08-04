import PyEigerData as Eiger
import numpy as np
import matplotlib.pyplot as plt

from skimage.transform import resize, rotate
import math
import time
from numba import jit
from skimage.feature import match_template
from skimage.color import rgb2gray
import cv2
import Find_Center_def as FC

A = Eiger.GeneralData()
A.open('16M_879_master.h5')
BSMask = A.convCSV2Logical('16M_879_mask.csv') # define mask for beamstop
PMask = A.PixelMask #  define the mask for the defect of detectors
Mask = np.logical_or(BSMask,PMask) # combine masks
A.ROI = A.convMask2ROI(Mask) # convert effective mask to ROI
print(A.ROI[0,0])
A.loadData(1) # load frame or frames

A.processData() # default: normalize

# NEWXYC = []
# for i in range(0,10):
#     t1 = time.time()
#     CC = FC.Find_Center()
#     NEWXYC.append(CC.M_Skimage(A.ProcessedData, diameter = int(i*100+500)))
#     t2 = time.time()
#     print(t2-t1)

# NEWXYC = np.array(NEWXYC)
# print(np.mean(NEWXYC[:,0]),np.mean(NEWXYC[:,1]))

print(A.XPixelsInDetector, A.YPixelsInDetector)
t1 = time.time()
ttttttt = A.ProcessedData.copy()
ttttttt = np.reshape(ttttttt, [4371,4150,1])
print(np.nanmax(ttttttt))
#CC = FC.find_Center()
#print(CC.Description,123)
#NEWXYC = CC.Method_CV2(A.ProcessedData, A.ROI,'CCORR_NORMED' , diameter = 1200)
#NEWXYC = FC.Method_Skimage(A.ProcessedData, diameter = 1000)
NEWXYC = FC.Method_CV2(A.ProcessedData, A.ROI,'CCORR_NORMED', diameter = 1000)
t2 = time.time()
print(t2-t1)
print(A.BeamCenterY,A.BeamCenterX)
print(NEWXYC)
# print(CC.New_Center[0], CC.New_Center[1])
# print(CC.BeamCenterY,CC.BeamCenterX)

print(A.ProcessedData.dtype)

# plt.imshow(A.ProcessedData)
# plt.show()

