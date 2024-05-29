import torch
from torch import nn
from glob import glob
import os
from dataset import Dataset
import torch.utils.data as Data
import SimpleITK as sitk
import argparse

class Conv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=2,
                 padding=1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm3d(out_channels),

        )

    def forward(self, x):
        x = self.block(x)
        return x



class Encoder(nn.Module):

    def __init__(self):
        super().__init__()

        ### Convolutional section
        padding_value = 1
        self.relu = nn.ReLU(True)
        self.encoder1 = Conv(1,8,3)
        self.encoder2 = Conv(8, 16, 3)
        self.encoder3 = Conv(16, 32, 3)
        self.encoder4 = Conv(32, 64, 3)
        self.encoder5 = Conv(64, 128, 3)
        self.encoder6 = Conv(128, 256, 3,stride=(2,2,1))
        self.encoder_linear = nn.Linear(8192, 512) #b,c,其中b=1,c=512
    def encode(self, x): # x
        x = self.encoder1(x)
        x = self.relu(x)
        x = self.encoder2(x)
        x = self.relu(x)
        x = self.encoder3(x)
        x = self.relu(x)
        x = self.encoder4(x)
        x = self.relu(x)
        x = self.encoder5(x)
        x = self.relu(x)
        x = self.encoder6(x)
        x = self.relu(x)
        x = x.view(1, -1)
        x = self.encoder_linear(x)
        return x

    def metric_net(self,x):
        x1 = self.encoder1(x)
        x = self.relu(x1)
        x2 = self.encoder2(x)
        x = self.relu(x2)
        x3 = self.encoder3(x)
        return x1,x2,x3

    def forward(self, x):
        x = self.encode(x) #1 256 3 3 3
        x = self.relu(x)
        return x

class DeConv(nn.Module):
    def __init__(self,inh,outh,kernel,stride,padding,outpadding=(0,1,1)):
        super().__init__()
        self.deconv = nn.Sequential(
                                    nn.ConvTranspose3d(inh, outh, kernel_size=kernel, stride=stride,padding=padding, output_padding=outpadding),
                                    nn.BatchNorm3d(outh),
                                    nn.ReLU(True))
    def forward(self,x):
        x = self.deconv(x)
        return x

class Decoder(nn.Module):

    def __init__(self):
        super().__init__()

        # self.unflatten = nn.Unflatten(dim=1,
        #                               unflattened_size=(512, 3, 3))

        self.deconv1 = DeConv(256,128,3,(2,2,1),(1,1,1),(1,1,0))
        self.deconv2 = DeConv(128, 64, 3, 2, 1,(0,0,1))
        self.deconv3 = DeConv(64, 32, 3, 2,1, 1)
        self.deconv4 = DeConv(32, 16, 3, 2, 1,1)
        self.deconv5 = DeConv(16, 8, 3, 2, 1,1) #8,33,120,120
        self.deconv6 = DeConv(8, 3, 3, 2,1,1) #3,65,240,240
        self.deconv7 = nn.Conv3d(3,1,3,1,1)

        self.decoder_linear = nn.Linear(512, 8192)

    def forward(self, x):
        x = self.decoder_linear(x)
        x = x.view(1,256,4,4,2)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        x = self.deconv6(x)
        x = self.deconv7(x)
        # x = torch.sigmoid(x)
        return x

def save_image(img, ref_img, name):
    img = sitk.GetImageFromArray(img[0, 0, ...].cpu().detach().numpy())
    img.SetOrigin(ref_img.GetOrigin())
    img.SetDirection(ref_img.GetDirection())
    img.SetSpacing(ref_img.GetSpacing())
    sitk.WriteImage(img, os.path.join("check", name))






if __name__=="__main__":
    device = torch.device('cuda')
    torch.manual_seed(41)
    random_tensor = torch.rand(1,1,240,240,64).to(device)
    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    x = encoder(random_tensor)  # 1,512
    metric = encoder.metric_net(random_tensor)
    x = decoder(x)
    print("d")




