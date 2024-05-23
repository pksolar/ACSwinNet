import os, utils, glob, losses
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch, ACSwinNet
from ACSwinNet import CONFIGS as CONFIGS_TM
import SimpleITK as sitk
from Model import losses
import argparse

def compute_label_dice(gt, pred):
    cls_lst = np.unique(gt)
    cls_lst.remove(0)
    dice_lst = []
    for cls in cls_lst:
        dice = losses.DSC(gt == cls, pred == cls) #
        dice_lst.append(dice)
    return np.mean(dice_lst)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_images_file', type=str, default="dataset/test/*.nii.gz")
    parser.add_argument('--model_file', type=str,default="pth")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--input_size', type=int, default=(160, 192, 160))
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--results_dir', type=str, default="result_test/")
    parser.add_argument('--ref_img_path', type=str, default="")

    args = parser.parse_args()
    config = CONFIGS_TM['ACSwinNet-Rigid']

    ref_image = sitk.ReadImage(args.ref_img_path)
    model = ACSwinNet.ACSwinNetRigid(config)
    model.load_state_dict(torch.load(args.model_file))['state_dict']
    model.cuda()

    """val"""
    test_files = glob.glob(args.test_images_file)
    test_set = datasets.CTDataset(data_path=test_files, transforms=None)  # transforms = train_composed
    val_loader = DataLoader(test_set, batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

    DSC = []
    DSC_b = []
    with torch.no_grad():
        for data, mseg, fseg in val_loader:

            model.eval()
            data = data.to(args.device).float()
            mseg = mseg.to(args.device).float()
            fseg = fseg.to(args.device).float()

            ####################
            # Rigid transform
            ####################
            x = data[:, :1, :, :, :]
            y = data[:, 1:, :, :, :]

            ct_rigid, out_seg, mat, inv_mats = model(data, mseg)

            x_seg_numpy = mseg[0, 0, ...].cpu().detach().numpy()
            y_seg_numpy = fseg[0, 0, ...].cpu().detach().numpy()
            out_seg_numpy = out_seg[0, 0, ...].cpu().detach().numpy()
            dice_before = compute_label_dice(y_seg_numpy, x_seg_numpy)
            dice = compute_label_dice(y_seg_numpy, out_seg_numpy)
            print("dice_before: ", dice_before)
            print("dice_after: ", dice)
        DSC.append(dice)
        DSC_b.append(dice_before)


    print("mean(DSC): ", np.mean(DSC), "   std(DSC): ", np.std(DSC))
    print("mean(DSC_b): ", np.mean(DSC_b), "   std(DSC): ", np.std(DSC_b))

if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    main()