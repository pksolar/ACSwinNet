import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


class U_Network(nn.Module):
    def __init__(self, dim, enc_nf, dec_nf, bn=None, full_size=True):
        """
        enc_nf ,dec_nf ：通道数的列表吗
        """
        super(U_Network, self).__init__()
        self.bn = bn     #是否batchnormalize
        self.dim = dim   #维度
        self.enc_nf = enc_nf    #encoder-nf 这是一个列表，里面极有可能是数字
        self.full_size = full_size
        #vm2 要么是1，要么是0
        self.vm2 = len(dec_nf) == 7 # PK: dec的层数为7，这里判断是voxelm1 还是2， 此处vm2 为1 或者为0
        # Encoder functions
        self.enc = nn.ModuleList()  #编码操作
        for i in range(len(enc_nf)):
            prev_nf = 2 if i == 0 else enc_nf[i - 1]   #prev_nf is what? ,
            #pk: prev_nf  =2  or  enc_nf(i-1)
            #prev_nf[i] 是in_channels, enc_nf[i]是out_channels, enc_nf 是编码各层channels的集合。
            self.enc.append(self.conv_block(dim, prev_nf, enc_nf[i], 4, 2, batchnorm=bn))
        # Decoder functions
        self.dec = nn.ModuleList()
        #enc最后一层的输出，dec第一层的输入。后面的通道数输入x2是因为cat了。
        self.dec.append(self.conv_block(dim, enc_nf[-1], dec_nf[0], batchnorm=bn))  # 1
        self.dec.append(self.conv_block(dim, dec_nf[0] * 2, dec_nf[1], batchnorm=bn))  # 2
        self.dec.append(self.conv_block(dim, dec_nf[1] * 2, dec_nf[2], batchnorm=bn))  # 3

        self.dec.append(self.conv_block(dim, dec_nf[2] + enc_nf[0], dec_nf[3], batchnorm=bn))  # 4
        self.dec.append(self.conv_block(dim, dec_nf[3], dec_nf[4], batchnorm=bn))  # 5

        if self.full_size:
            self.dec.append(self.conv_block(dim, dec_nf[4] + 2, dec_nf[5], batchnorm=bn))
        if self.vm2:
            self.vm2_conv = self.conv_block(dim, dec_nf[5], dec_nf[6], batchnorm=bn)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # One conv to get the flow field
        #PK: 最后得到变形场：，dim是几通道，几通道的图片得到几通道的变形场。
        #PK: flow也是一次卷积，dim = 3，将dec最后一层拿出来。做一次卷积。
        conv_fn = getattr(nn, 'Conv%dd' % dim)
        self.flow = conv_fn(dec_nf[-1], dim, kernel_size=3, padding=1)
        # Make flow weights + bias small. Not sure this is necessary.
        nd = Normal(0, 1e-5)
        self.flow.weight = nn.Parameter(nd.sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))
        self.batch_norm = getattr(nn, "BatchNorm{0}d".format(dim))(3)
    #pk: conv_block 就是卷积+（bn）+relu
    def conv_block(self, dim, in_channels, out_channels, kernel_size=3, stride=1, padding=1, batchnorm=False):
        conv_fn = getattr(nn, "Conv{0}d".format(dim))
        bn_fn = getattr(nn, "BatchNorm{0}d".format(dim))
        if batchnorm:
            layer = nn.Sequential(
                conv_fn(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
                bn_fn(out_channels),
                nn.LeakyReLU(0.2))
        else:
            layer = nn.Sequential(
                conv_fn(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
                nn.LeakyReLU(0.2))
        return layer

    def forward(self, src, tgt):
        x = torch.cat([src, tgt], dim=1)   #利用cat将两种图拼接在一起。

        # Get encoder activations
        x_enc = [x]
        #enc是一个ModuleList，l是list里的layer，5个卷积。
        for i, l in enumerate(self.enc):
            #
            x = l(x_enc[-1])  #已经在层里计算了。后的尺寸小是因为在卷积。
            x_enc.append(x)  #并将每层得到的结果保存
        # Three conv + upsample + concatenate series
        y = x_enc[-1]
        for i in range(3):
            y = self.dec[i](y)
            y = self.upsample(y)
            y = torch.cat([y, x_enc[-(i + 2)]], dim=1)
        # Two convs at full_size/2 res
        y = self.dec[3](y)
        y = self.dec[4](y)
        # Upsample to full res, concatenate and conv
        if self.full_size:
            y = self.upsample(y)
            y = torch.cat([y, x_enc[0]], dim=1)
            y = self.dec[5](y)
        # Extra conv for vm2
        if self.vm2:
            y = self.vm2_conv(y)
        flow = self.flow(y)
        if self.bn:
            flow = self.batch_norm(flow)
        return flow


class SpatialTransformer(nn.Module):
    def __init__(self, size, mode='bilinear'): # size is the vol_size(240,240,64)
        super(SpatialTransformer, self).__init__()
        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]

        grids = torch.meshgrid(vectors)

        grid = torch.stack(grids)  # y, x, z

        grid = torch.unsqueeze(grid, 0)  # add batch

        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

        self.mode = mode

    def forward(self, src, flow):

        new_locs = self.grid + flow #[1,3,240,240,64]

        shape = flow.shape[2:]

        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)#[2,3,4] is used to locate.
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, mode=self.mode)
