import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize, rotate
from skimage.feature import match_template
from numba import jit, cuda
import cv2
import math
import time
#KKKKKKK#


def Method_Skimage(data_ndarray, diameter = 100):

        @jit(nopython = True)
        def Intensity_normalization(normalization_ndarray):
            print('Start Intensity_normalization\n')
            Data_max = np.nanmax(normalization_ndarray)
            Data_min = np.nanmin(normalization_ndarray)

            for y in range(0, normalization_ndarray.shape[0]):
                for x in range(0, normalization_ndarray.shape[1]):
                    if  np.isnan(normalization_ndarray[y,x]) == False:
                        normalization_ndarray[y,x] = (normalization_ndarray[y,x]-Data_min)/(Data_max-Data_min)
                    elif np.isnan(normalization_ndarray[y,x]) == True:
                        normalization_ndarray[y,x] = np.random.rand()
            
            print('End of Intensity_normalization\n')
            return normalization_ndarray
        
        @jit(nopython = True)
        def Intensity_normalization_NaN(normalization_ndarray):
            print('Start Intensity_normalization_NaN\n')
            Data_max = np.nanmax(normalization_ndarray)
            Data_min = np.nanmin(normalization_ndarray)

            for y in range(0, normalization_ndarray.shape[0]):
                for x in range(0, normalization_ndarray.shape[1]):
                    if  np.isnan(normalization_ndarray[y,x]) == False:
                        normalization_ndarray[y,x] = (normalization_ndarray[y,x]-Data_min)/(Data_max-Data_min)
                    elif np.isnan(normalization_ndarray[y,x]) == True:
                        normalization_ndarray[y,x] = -np.random.rand()
            
            print('End of Intensity_normalization_NaN\n')
            return normalization_ndarray

        @jit(nopython = True)
        def findCenterofMass(CM_ndarray):
            CenterofMass = []
            TotalMass = 0.0
            print('Start findCenterofMass\n')

            for y in range(0, CM_ndarray.shape[0]):
                for x in range(0, CM_ndarray.shape[1]):
                    if  np.isnan(CM_ndarray[y,x]) == False:
                        CenterofMass.append([y*CM_ndarray[y,x], x*CM_ndarray[y,x]])
                        TotalMass += CM_ndarray[y,x]

            CenterofMass = np.array(CenterofMass)
            CenterofMass = [int(np.sum(CenterofMass[:,0], axis = 0)/TotalMass), int(np.sum(CenterofMass[:,1], axis = 0)/TotalMass)]
            
            print('End of findCenterofMass\n')
            return CenterofMass

        DiameterInTM = diameter
        All_ProcessedData = np.copy(data_ndarray)
        #YCenterInTM, XCenterInTM = 708,344
        #YCenterInTM, XCenterInTM = np.where(All_ProcessedData == np.nanmax(All_ProcessedData))
        YCenterInTM, XCenterInTM = findCenterofMass(All_ProcessedData)

        Symmetric_ProcessedData = All_ProcessedData[int(int(YCenterInTM)-(DiameterInTM/2)):int(int(YCenterInTM)+(DiameterInTM/2)),int(int(XCenterInTM)-(DiameterInTM/2)):int(int(XCenterInTM)+(DiameterInTM/2))]
        Symmetric_rotate_ProcessedData = (rotate(Symmetric_ProcessedData, 180.0 ))


        Symmetric_rotate_ProcessedData_norm = Intensity_normalization(Symmetric_rotate_ProcessedData)
        All_ProcessedData_norm = Intensity_normalization_NaN(All_ProcessedData)

        print('Start match_template\n')
        results = match_template(All_ProcessedData_norm, Symmetric_rotate_ProcessedData_norm)
        left_up = np.unravel_index(np.argmax(results), results.shape) # this point is left_up cornor
        x_pixel_posi, y_pixel_posi = left_up[::-1]
        print('End of match_template\n')
        
        results_ProcessedData = All_ProcessedData[left_up[0]:left_up[0]+DiameterInTM, left_up[1]:left_up[1]+DiameterInTM]
    
        
        results_Center = [int((left_up[0] + (left_up[0] + (DiameterInTM-1)))/2), int((left_up[1] + (left_up[1] + (DiameterInTM-1)))/2)]
        New_Center = [(results_Center[0] + int(YCenterInTM))/2, (results_Center[1] + int(XCenterInTM))/2]
        print('results_center_y: ', int((left_up[0] + (left_up[0] + (DiameterInTM-1)))/2) , 'results_center_x: ', int((left_up[1] + (left_up[1] + (DiameterInTM-1)))/2))
        print('New_center_y: ',New_Center[0] , 'New_center_x: ',New_Center[1] )



        plt.subplot(141)
        plt.title('Target image')
        plt.imshow(np.log(Symmetric_ProcessedData))

        plt.subplot(142)
        plt.title('Target "rotated image"')
        plt.imshow(np.log(Symmetric_rotate_ProcessedData))

        plt.subplot(143)
        plt.title('Match in original image')
        plt.imshow(np.log((All_ProcessedData )))
        rect = plt.Rectangle((x_pixel_posi, y_pixel_posi), Symmetric_ProcessedData.shape[1], Symmetric_ProcessedData.shape[0], edgecolor='red', facecolor='none')
        plt.subplot(143).add_patch(rect)

        plt.subplot(144)
        plt.title('Zoom in of original image')
        plt.imshow(np.log(results_ProcessedData))

        plt.show()

        return New_Center


def Method_CV2(data_ndarray, mask_ndarray, method = 'CCORR_NORMED', diameter = 100):

        @jit(nopython = True)
        def Intensity_normalization(normalization_ndarray):
            print('Start Intensity_normalization\n')
            Data_max = np.nanmax(normalization_ndarray)
            Data_min = np.nanmin(normalization_ndarray)

            for y in range(0, normalization_ndarray.shape[0]):
                for x in range(0, normalization_ndarray.shape[1]):
                    if  np.isnan(normalization_ndarray[y,x]) == False:
                        normalization_ndarray[y,x] = (normalization_ndarray[y,x]-Data_min)/(Data_max-Data_min)
                    elif np.isnan(normalization_ndarray[y,x]) == True:
                        normalization_ndarray[y,x] = 0.0
            
            #normalization_ndarray = normalization_ndarray.astype('float32')
            print('End of Intensity_normalization\n')
            return normalization_ndarray
        
        
        def def_mask(Mask_ndarray):
            # Note that 1.0 means "use this pixel" and 0.0 means "do not". 
            # If the value between 0.0 and 1.0, it represents the weight.
            print('Start def_mask\n')
            # for y in range(0, Mask_ndarray.shape[0]):
            #     for x in range(0, Mask_ndarray.shape[1]):
            #         if Mask_ndarray[y,x] == False:
            #             Mask_ndarray[y,x] = 0.0 
            #         elif Mask_ndarray[y,x] == True:
            #             Mask_ndarray[y,x] = 1.0
            Mask_ndarray[Mask_ndarray == False] = 0.0
            Mask_ndarray[Mask_ndarray == True] = 1.0
            #Mask_ndarray = Mask_ndarray.astype('float32')
            print('End of def_mask\n')
            return Mask_ndarray
        
        
        
        @jit(nopython = True)
        def findCenterofMass(CM_ndarray):
            CenterofMass = []
            TotalMass = 0.0
            print('Start findCenterofMass\n')

            for y in range(0, CM_ndarray.shape[0]):
                for x in range(0, CM_ndarray.shape[1]):
                    if  np.isnan(CM_ndarray[y,x]) == False:
                        CenterofMass.append([y*CM_ndarray[y,x], x*CM_ndarray[y,x]])
                        TotalMass += CM_ndarray[y,x]

            CenterofMass = np.array(CenterofMass)
            CenterofMass = [int(np.sum(CenterofMass[:,0], axis = 0)/TotalMass), int(np.sum(CenterofMass[:,1], axis = 0)/TotalMass)]

            print('End of findCenterofMass\n')
            return CenterofMass


        TM_methods = {'SQDIFF':0, 'SQDIFF_NORMED':1, 'CCORR':2, 'CCORR_NORMED':3, 'CCOEFF':4, 'CCOEFF_NORMED':5}
        TM_method = TM_methods[method]
        DiameterInTM = diameter
        All_ProcessedData = np.copy(data_ndarray)
        Mask_ndarray = np.copy(mask_ndarray)


        #YCenterInTM, XCenterInTM = 1500,1500
        #YCenterInTM, XCenterInTM = np.where(All_ProcessedData == np.nanmax(All_ProcessedData))
        YCenterInTM, XCenterInTM = findCenterofMass(All_ProcessedData)


        Symmetric_ProcessedData = All_ProcessedData[int(int(YCenterInTM)-(DiameterInTM/2)):int(int(YCenterInTM)+(DiameterInTM/2)),int(int(XCenterInTM)-(DiameterInTM/2)):int(int(XCenterInTM)+(DiameterInTM/2))]
        Symmetric_rotate_ProcessedData = (rotate(Symmetric_ProcessedData, 180.0 ))
        

        Symmetric_rotate_ProcessedData_norm = Intensity_normalization(Symmetric_rotate_ProcessedData)
        Symmetric_rotate_ProcessedData_norm = Symmetric_rotate_ProcessedData_norm.astype('float32')
        
        All_ProcessedData_norm = Intensity_normalization(All_ProcessedData)
        All_ProcessedData_norm = All_ProcessedData_norm.astype('float32')

        
        F32_mask = def_mask(Mask_ndarray) # F32 means 'float32'
        F32_mask = F32_mask.astype('float32')
        F32_mask = F32_mask[int(int(YCenterInTM)-(DiameterInTM/2)):int(int(YCenterInTM)+(DiameterInTM/2)),int(int(XCenterInTM)-(DiameterInTM/2)):int(int(XCenterInTM)+(DiameterInTM/2))]       
        F32_mask = cv2.rotate(F32_mask, cv2.ROTATE_180)


        print('Start matchTemplate\n')
        results = cv2.matchTemplate(All_ProcessedData_norm, Symmetric_rotate_ProcessedData_norm, TM_method, mask = F32_mask)
        print('End of matchTemplate\n')

        Min_Max_Loc = cv2.minMaxLoc(results) # return (min_val, max_val, min_loc, max_loc)
        if TM_method == 0 or TM_method == 1:  
            Min_Max_Loc = Min_Max_Loc[2] # minimum location
        else:
            Min_Max_Loc = Min_Max_Loc[3] # maximum location


        left_up = Min_Max_Loc ## this point is left_up cornor
        x_pixel_posi, y_pixel_posi = left_up[0], left_up[1]
        right_bottom = (y_pixel_posi + Symmetric_rotate_ProcessedData_norm.shape[0], x_pixel_posi + Symmetric_rotate_ProcessedData_norm.shape[1])
        results_ProcessedData = All_ProcessedData[y_pixel_posi:right_bottom[0], x_pixel_posi:right_bottom[1]]


        results_Center = [(right_bottom[0] + y_pixel_posi)/2.0, (right_bottom[1] + x_pixel_posi)/2.0]
        New_Center = [(results_Center[0] + int(YCenterInTM))/2, (results_Center[1] + int(XCenterInTM))/2]
        print('results_center_y: ',  results_Center[0], 'results_center_x: ', results_Center[1])
        print('New_center_y: ',New_Center[0] , 'New_center_x: ',New_Center[1] )

        plt.subplot(141)
        plt.title('Target image')
        plt.imshow(np.log(Symmetric_ProcessedData))

        plt.subplot(142)
        plt.title('Target "rotated image"')
        plt.imshow(np.log(Symmetric_rotate_ProcessedData))

        plt.subplot(143)
        plt.title('Match in original image')
        plt.imshow(np.log((All_ProcessedData )))
        rect = plt.Rectangle((x_pixel_posi, y_pixel_posi), Symmetric_ProcessedData.shape[1], Symmetric_ProcessedData.shape[0], edgecolor='red', facecolor='none')
        plt.subplot(143).add_patch(rect)

        plt.subplot(144)
        plt.title('Zoom in of original image')
        plt.imshow(np.log(results_ProcessedData))

        plt.show()
        
        return New_Center