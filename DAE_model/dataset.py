import os
import glob
import numpy as np
import SimpleITK as sitk
import torch.utils.data as Data


def normalize(x):
    Max = np.max(x)
    Min = np.min(x)
    x_ = (x-Min)/(Max-Min)
    return x_


class Dataset(Data.Dataset):
    def __init__(self, files):

        self.files = files

    def __len__(self):

        return len(self.files_fixed)

    def __getitem__(self, index):

        img_arr = sitk.GetArrayFromImage(sitk.ReadImage(self.files[index]))[np.newaxis, ...]
        normalize(img_arr)
        # add noise
        img_arr_addNoise = img_arr + np.random.normal(0,0.1,img_arr.shape)

        return img_arr_addNoise,img_arr