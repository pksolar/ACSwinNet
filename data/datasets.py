import os, glob
import torch, sys
from torch.utils.data import Dataset
from .data_utils import pkload
import matplotlib.pyplot as plt
import SimpleITK  as sitk

import numpy as np
def scale_volume(input_volume, upper_bound=255, lower_bound=0):
    max_value = np.max(input_volume)
    min_value = np.min(input_volume)

    k = (upper_bound - lower_bound) / (max_value - min_value)
    scaled_volume = k * (input_volume - min_value) + lower_bound
    return scaled_volume

class CTDataset(Dataset):
    def __init__(self, data_path,transforms):
        self.paths = data_path
        # self.seg_paths = data_seg_path
        self.transforms = transforms

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        mip = self.paths[index] #mi_path
        fip = mip.replace("mi","fi")
        mi = sitk.GetArrayFromImage(sitk.ReadImage(mip))[np.newaxis, ...]
        fi = sitk.GetArrayFromImage(sitk.ReadImage(fip))[np.newaxis, ...]

        if 'test' not in mip:
            msp = mip.replace("mi", "msi")
            fsp = mip.replace("mi", "fsi")
            msi = sitk.GetArrayFromImage(sitk.ReadImage(msp))[np.newaxis, ...]
            fsi = sitk.GetArrayFromImage(sitk.ReadImage(fsp))[np.newaxis, ...]

            total_arr = np.concatenate((mi,fi),axis=0)
            total_arr = scale_volume(total_arr,upper_bound=1.0,lower_bound=0.0)
            return total_arr,msi,fsi
        else:
            msp = mip.replace("mi","msi")
            fsp = mip.replace("mi","fsi")
            total_arr = np.concatenate((mi, fi), axis=0)
            total_arr = scale_volume(total_arr, upper_bound=1.0, lower_bound=0.0)

            return total_arr,mip,fip,msp,fsp



class CycDataset(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        x, y = pkload(path)
        x, y = x[None, ...], y[None, ...]
        x,y = self.transforms([x, y])
        x = np.ascontiguousarray(x)
        x = torch.from_numpy(x)
        y = np.ascontiguousarray(y)
        y = torch.from_numpy(y)
        return x, y

    def __len__(self):
        return len(self.paths)

class CTSegDataset(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        x, x_seg = pkload(self.paths[index])
        y = pkload('D:/DATA/Duke/XCAT/phan.pkl')
        y_seg = pkload('D:/DATA/Duke/XCAT/label.pkl')
        y = np.flip(y, 1)
        y_seg = np.flip(y_seg, 1)
        # transforms work with nhwtc
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]
        x,y = self.transforms([x, y])
        x_seg, y_seg = self.transforms([x_seg, y_seg])
        x = np.ascontiguousarray(x)
        x = torch.from_numpy(x)
        y = np.ascontiguousarray(y)
        y = torch.from_numpy(y)

        x_seg = np.ascontiguousarray(x_seg).astype(np.uint8)
        x_seg = torch.from_numpy(x_seg)
        y_seg = np.ascontiguousarray(y_seg).astype(np.uint8)
        y_seg = torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.paths)