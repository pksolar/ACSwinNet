import os, utils, glob, losses
import sys
from torch.utils.data import DataLoader
from data import datasets
import numpy as np
import ACSwinNet
from torch import optim
import matplotlib.pyplot as plt
from ACSwinNet import CONFIGS as CONFIGS_TM
from natsort import natsorted
from infer import compute_label_dice
import torch.nn as nn
import torch
from  DAE_model import network
import argparse
import random




class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir+"logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def MSE(x, y):
    return torch.mean((x - y) ** 2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_images_file', type=str,default="dataset/train/*.nii.gz")
    parser.add_argument('--val_images_file', type=str,default="dataset/val/*.nii.gz")
    parser.add_argument('--autoencoder_file_seg', type=str,default="pth/encoder_seg.pth")
    parser.add_argument('--autoencoder_file_image', type=str, default="pth/encoder_image.pth")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--image_weight', type=float, default=1.0)
    parser.add_argument('--ae_weight', type=float, default=0.1)
    parser.add_argument('--dice_weight', type=float, default=1.0)
    parser.add_argument('--input_size', type=int, default=(240,240,64))
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--results_dir', type=str,default="result/")
    parser.add_argument('--save_model', action='store_true')

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    save_dir = args.results_dir
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    if not os.path.exists(args.results_dir+"//log"):
        os.makedirs(args.results_dir+"//log")
    sys.stdout = Logger(args.results_dir+"//log")
    lr =args.lr
    epoch_start = 0
    max_epoch = args.epochs
    cont_training = False

    '''
    Initialize model
    '''
    config = CONFIGS_TM['ACSwinNet-Rigid']
    model = ACSwinNet.ACSwinNetRigid(config)
    model.cuda()


    '''
    Continue training
    '''
    if cont_training:
        epoch_start = 335
        model_dir = save_dir
        best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[0])['state_dict']
        model.load_state_dict(best_model)

    """train"""
    train_files_mi = glob.glob(args.train_images_file)
    # train_seg_files = glob.glob(os.path.join(path_seg_moved, '*.nii.gz'))
    train_set = datasets.CTDataset(data_path=train_files_mi, transforms=None)  # transforms = train_composed
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    """val"""
    val_files = glob.glob(args.val_images_file)
    val_set = datasets.CTDataset(data_path=val_files, transforms=None)  # transforms = train_composed
    val_loader = DataLoader(val_set, batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

    # Optimizers
    optimizer = optim.Adam(model.parameters(), lr=lr, amsgrad=True)

    best_mse = 1e10

    dae_encoder_seg = network.Encoder().to(args.device)
    dae_encoder_image = network.Encoder().to(args.device)

    dae_encoder_seg.load_state_dict(torch.load(args.autoencoder_file_seg))
    dae_encoder_image.load_state_dict(torch.load(args.autoencoder_file_image))

    dae_encoder = dae_encoder_seg.eval().encode()
    mnmodel =  dae_encoder_image.eval().metric_net()



    dice_loss_fn = losses.Dice()
    ae_loss_fn = losses.L2Squared()
    perceptual_loss_fn = nn.L1Loss()

    for epoch in range(epoch_start, max_epoch):
        print('Training Starts')
        '''
        Training
        '''
        dsc_ = []
        loss_all = utils.AverageMeter()
        idx = 0


        for data,mseg,fseg in train_loader:
            idx += 1
            model.train()

            data = data.to(args.device).float()

            mseg =mseg.to(args.device).float()
            fseg = fseg.to(args.device).float()

            # moving image
            x = data[:,:1, :, :,:]
            # fixed iamge
            y = data[:,1:, :, :,:]

            out_x, out_seg, mat, inv_mats = model(data, mseg)

            out_seg_enc = dae_encoder(out_seg)
            f_seg_enc = dae_encoder(fseg)

            x_feature= mnmodel(out_x)  #
            y_feature = mnmodel(y)
            perceptual_loss =  perceptual_loss_fn(x_feature, y_feature)

            loss = args.image_weight * perceptual_loss
            loss += args.ae_weight * ae_loss_fn(out_seg_enc, f_seg_enc)
            loss += args.dice_weight * dice_loss_fn(out_seg, fseg)

            optimizer.zero_grad()
            loss.backward()


            print('Iter {} of {} loss {:.6f} '.format(idx, len(train_loader), loss.item()))
        print('Epoch {}, loss {:.4f}'.format(epoch, loss_all.avg))

        '''
        Validation
        '''
        eval_mse = utils.AverageMeter()
        with torch.no_grad():
            for data,mseg,fseg in val_loader:
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

                phan = y
                #mse_ori = MSE_torch(phan, ct_rigid)
                mse_seg = MSE(fseg, out_seg)
                mse =  mse_seg
                eval_mse.update(mse.item(), x.size(0))
                print(eval_mse.avg)
                print("mse_seg:",mse_seg.item())
        print("mean_dsc: ",np.mean(dsc_))
        best_mse = min(eval_mse.avg, best_mse)
        print("save_check")
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_mse': best_mse,
            'optimizer': optimizer.state_dict(),
        }, save_dir=save_dir, filename='mse{:.4f}.pth.tar'.format(eval_mse.avg))

        plt.switch_backend('agg')
        xcat_fig = comput_fig(phan)
        ct_fig = comput_fig(x)
        rigid_fig = comput_fig(ct_rigid)

        plt.close(rigid_fig)

        plt.close(xcat_fig)

        plt.close(ct_fig)
        loss_all.reset()
        eval_mse.reset()


def comput_fig(img):
    img = img.detach().cpu().numpy()[0, 0, :, 72:88, :]
    fig = plt.figure(figsize=(12,12), dpi=180)
    for i in range(img.shape[1]):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.imshow(img[:, i, :], cmap='gray')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig


def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=8):
    torch.save(state, save_dir+filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[-1])
        model_lists = natsorted(glob.glob(save_dir + '*'))

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