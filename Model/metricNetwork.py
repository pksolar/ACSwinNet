import torch
import torch.nn as nn

class Conv3dBlock(nn.Module):
    """
        conv3d
        bn
        relu
        """
    def __init__(self,in_channels,out_channels,kernelsz,stride):
        super(Conv3dBlock,self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernelsz, stride),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.2)
        )
    def forward(self,x):
        x = self.block(x)
        return x

class SW(nn.Module):
    def __init__(self,in_channel):
        super(SW,self).__init__()
        self.conv1 = nn.Conv3d(in_channels=in_channel,out_channels=1,kernel_size=1,stride=1,bias=False)
        self.conv2 = nn.Conv3d(in_channels=1,out_channels = 1,kernel_size=3,padding=1,bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        x  = x * identity
        return x

class SE(nn.Module):
    def __init__(self,in_channel):
        super(SE,self).__init__()
        #avtivate
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        #c
        self.globalpooling = nn.AdaptiveAvgPool2d((1,1))
        self.conv1 = nn.Conv3d(in_channels = in_channel , out_channels= int(in_channel/2) ,kernel_size=1,stride=1,)
        self.conv2 = nn.Conv3d(in_channels=int( in_channel/2),out_channels= in_channel ,kernel_size=1,stride=1)
        #s
        self.convs1 = nn.Conv3d(in_channels = in_channel,out_channels=1,kernel_size=1)

    def forward(self,x):
        #s
        xs = self.convs1(x)
        xs = self.sigmoid(xs)
        xs = xs * x

        #c
        xc = self.globalpooling(x)
        xc = self.conv1(xc)
        xc = self.relu(xc)
        xc = self.conv2(xc)
        xc = self.sigmoid(xc)
        sc = xc * x

        return torch.maximum(xc,xs)

        #max-out
class Metric(nn.Module):
    """
    this is similiar with perspectvie loss ,
    cbct and ct data share the parameters.
    """
    def __init__(self):
        super(Metric,self).__init__()
        self.conv3dblock1 = Conv3dBlock(1,16,3,1)
        self.sw1 = SE(16)
        self.conv3dblock2 = Conv3dBlock(16, 32, 3,1)
        self.sw2 = SE(32)
        self.conv3dblock3 = Conv3dBlock(32, 46, 3,1)
        self.sw3 = SE(46)
        self.maxpool = nn.MaxPool3d(2)

    def forward(self,x):
        x = self.conv3dblock1(x)
        x_1 = x
        x = self.maxpool(x)
        x = self.conv3dblock2(x)
        x_2 = x
        x = self.maxpool(x)
        x = self.conv3dblock3(x)
        return  x_1,x_2,x








