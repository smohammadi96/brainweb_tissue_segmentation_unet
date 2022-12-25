import numpy as np


class ComputeMetrics:
    def __init__(self):
        pass

    def iou(self, mask, maskP):
        """ compute Intersection over union between groudtruth mask and model output

        inputs: 
        mask --> Groundtruth (numpy array) 
        maskP --> Prediction (numpy array)

        output --> iou (float)
        """

        print("Computing IOU")
        return np.sum(np.logical_and(mask, maskP))/np.sum(np.logical_or(mask, maskP))

    def dice_score(self, predImg, imgMask):
        """ compute Dice Score between groudtruth mask and model output

        inputs: 
        imgMask --> Groundtruth (numpy array) 
        predImg --> Prediction (numpy array)

        output --> iou (float)
        """        
        
        print('Computing Dice Score')
        return np.sum(predImg[imgMask == 1])*2 / (np.sum(predImg) + np.sum(imgMask))