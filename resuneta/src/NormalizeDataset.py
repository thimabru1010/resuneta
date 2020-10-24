"""
Class for normalizing the sliced images for any dataset
"""

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# def normalization(image, norm_type = 1):
#     image_reshaped = image.reshape((image.shape[0]*image.shape[1]),image.shape[2])
#     if (norm_type == 1):
#       scaler = StandardScaler()
#     if (norm_type == 2):
#       scaler = MinMaxScaler(feature_range=(0,1))
#     if (norm_type == 3):
#       scaler = MinMaxScaler(feature_range=(-1,1))
#     scaler = scaler.fit(image_reshaped)
#     image_normalized = scaler.fit_transform(image_reshaped)
#     image_normalized1 = image_normalized.reshape(image.shape[0],image.shape[1],image.shape[2])
#     return image_normalized1


class Normalize(object):
    def __init__(self, mean=None, std=None):

        if (mean == None or std == None):
            self._mean = np.array([ 127.5, 127.5, 127.5])
            self._std = np.array ([127.5, 127.5, 127.5])


        else :
            self._mean = mean
            self._std = std


    def __call__(self, img):

        # temp = img.astype(np.float32)
        # temp2 = temp.T
        # temp2 -= self._mean
        # temp2 /= self._std
        #
        # temp = temp2.T

        temp = img.astype(np.float32)
        temp2 = temp
        temp2 -= self._mean
        temp2 /= self._std

        temp = temp2

        return temp


    def restore(self, normed_img):
        # Watch out the transpose here

        d2 = normed_img.T * self._std
        d2 = d2 + self._mean
        d2 = d2.T
        d2 = np.round(d2)
        d2 = d2.astype('uint8')

        return d2
