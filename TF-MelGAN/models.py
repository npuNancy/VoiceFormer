import torch
import torch.nn as nn
from torch.nn import Conv1d
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

import numpy as np
from pytorch_wavelets import DWTForward
from utils import init_weights, get_padding
from torch.nn.utils import weight_norm, spectral_norm


LRELU_SLOPE = 0.1


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    # elif classname.find("InstanceNorm1d") != -1:
    #     m.weight.data.normal_(1.0, 0.02)
    #     m.bias.data.fill_(0)


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def SNConv1d(*args, **kwargs):
    return spectral_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


class ResBlock(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock, self).__init__()
        self.h = h
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class Generator(torch.nn.Module):
    def __init__(self, h):
        super(Generator, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.conv_pre = weight_norm(Conv1d(80, h.upsample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(h.upsample_initial_channel//(2**i), h.upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        # norm_f: 归一化函数的使用, 根据use_spectral_norm来
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 128, 15, 1, padding=7)),   # [b, 128, seglen=2048]
            norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),  # [b, 128, 1024]
            norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),     # [b, 256, 512]
            norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),     # [b, 512, 128]
            norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),    # [b, 1024, 32]
            norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),   # [b, 1024, 32]
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),                # [b, 1024, 32]
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))   # [b, 1, 32]

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)  # [b, 32]

        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True),
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        self.meanpools = nn.ModuleList([
            AvgPool1d(4, 2, padding=2),
            AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, y, y_hat):
        y_d_rs = []     # real samples: 真实输出
        y_d_gs = []     # generated_samples: 生成输出
        fmap_rs = []    # fmaps真
        fmap_gs = []    # 生成
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i-1](y)
                y_hat = self.meanpools[i-1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorW(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorW, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.LRELU_SLOPE = 0.1
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, self.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class WaveletDiscriminator(nn.Module):
    def __init__(self, J=3) -> None:
        super().__init__()
        self.level = J
        self.dwt = DWTForward(J=self.level, mode='zero', wave='db1')
        self.discriminator = DiscriminatorW()

    def forward(self, x):
        x = x.unsqueeze(2)  # x.shape=[B, 1, 1, Length]
        wavelet_L, wavelet_H = self.dwt(x)
        if len(wavelet_H) != self.level:
            print("ERROR")
            exit()
        result = []
        feature_map = []
        for level in range(self.level):
            # HH = LH
            wavelet_HH = wavelet_H[level][:, :, 2, :, :].squeeze(1)
            res_HH, fmap_HH = self.discriminator(wavelet_HH)
            result.append(res_HH)
            feature_map.append(fmap_HH)
        res_L, fmap_L = self.discriminator(wavelet_L.squeeze(2))
        result.append(res_L)
        feature_map.append(fmap_L)

        return result, feature_map


class NLayerDiscriminator(nn.Module):
    def __init__(self, ndf, n_layers, downsampling_factor):
        super().__init__()
        model = nn.ModuleDict()

        model["layer_0"] = nn.Sequential(
            nn.ReflectionPad1d(7),
            WNConv1d(1, ndf, kernel_size=15),
            nn.LeakyReLU(0.2, True),
        )

        nf = ndf
        stride = downsampling_factor
        for n in range(1, n_layers + 1):
            nf_prev = nf
            nf = min(nf * stride, 1024)

            model["layer_%d" % n] = nn.Sequential(
                WNConv1d(
                    nf_prev,
                    nf,
                    kernel_size=stride * 10 + 1,
                    stride=stride,
                    padding=stride * 5,
                    groups=nf_prev // 4,
                ),
                nn.LeakyReLU(0.2, True),
            )

        nf = min(nf * 2, 1024)
        model["layer_%d" % (n_layers + 1)] = nn.Sequential(
            WNConv1d(nf_prev, nf, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2, True),
        )

        model["layer_%d" % (n_layers + 2)] = WNConv1d(
            nf, 1, kernel_size=3, stride=1, padding=1
        )

        self.model = model

    def forward(self, x):
        results = []
        for key, layer in self.model.items():
            x = layer(x)
            results.append(x)
        return results


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss*2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1-dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def discriminator_loss_for_wd(disc_real_outputs, disc_generated_outputs, J=3):
    """
    描述: 
        计算判别器的损失，for小波判别器
        小波判别器给高频的权重高，低频的权重低
    参数: 
        disc_real_outputs: 真实样本 [B, 4]
        disc_generated_outputs: 生成样本 [B, 4]
        J: 小波分解的层数
    """
    # 生成权重，[J, J-1, ……， 2, 1]
    loss_weight = np.arange(J+1, 0, -1)

    loss = 0
    r_losses = []
    g_losses = []
    for i, (dr, dg) in enumerate(zip(disc_real_outputs, disc_generated_outputs)):
        r_loss = torch.mean((1-dr)**2)
        g_loss = torch.mean(dg**2)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())
        loss += (r_loss + g_loss) * loss_weight[i]

    return loss, r_losses, g_losses


def generator_loss_for_wd(disc_outputs, J=3):
    """
    描述: 
        计算生成器的损失，for小波判别器
        小波判别器给高频的权重高，低频的权重低
    参数: 
        disc_outputs: 生成样本 [B, 4]
        J: 小波分解的层数
    """
    # 生成权重，[J, J-1, ……， 2, 1]
    loss_weight = np.arange(J+1, 0, -1)

    loss = 0
    gen_losses = []
    for i, dg in enumerate(disc_outputs):
        l = torch.mean((1-dg)**2)
        gen_losses.append(l)
        loss += l * loss_weight[i]

    return loss, gen_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1-dg)**2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses


def wavelet_loss(real_data, gen_data, J=3):
    """
    描述:
        计算小波损失
    参数:
        real_data: 真实数据 shape=[B, 1, 1, Length]
        gen_data: 生成数据 shape=[B, 1, 1, Length]
        J: 小波分解层数
    返回:
        loss: 小波损失
    """
    dwt = DWTForward(J=J, mode='zero', wave='db1').cuda()
    wavelet_real_L, wavelet_real_H = dwt(real_data.unsqueeze(2))
    wavelet_real_LL = wavelet_real_L.squeeze(2)
    wavelet_real_H1 = wavelet_real_H[0][:, :, 2, :, :].squeeze(1)
    wavelet_real_H2 = wavelet_real_H[1][:, :, 2, :, :].squeeze(1)
    wavelet_real_H3 = wavelet_real_H[2][:, :, 2, :, :].squeeze(1)

    wavelet_gen_L, wavelet_gen_H = dwt(gen_data.unsqueeze(2))
    wavelet_gen_LL = wavelet_gen_L.squeeze(2)
    wavelet_gen_H1 = wavelet_gen_H[0][:, :, 2, :, :].squeeze(1)
    wavelet_gen_H2 = wavelet_gen_H[1][:, :, 2, :, :].squeeze(1)
    wavelet_gen_H3 = wavelet_gen_H[2][:, :, 2, :, :].squeeze(1)

    wavelet_loss_LL = F.mse_loss(wavelet_real_LL, wavelet_gen_LL)
    wavelet_loss_H1 = F.l1_loss(wavelet_real_H1, wavelet_gen_H1)
    wavelet_loss_H2 = F.l1_loss(wavelet_real_H2, wavelet_gen_H2)
    wavelet_loss_H3 = F.l1_loss(wavelet_real_H3, wavelet_gen_H3)

    wavelet_loss_res = wavelet_loss_LL + wavelet_loss_H1 + wavelet_loss_H2 + wavelet_loss_H3
    del dwt
    return wavelet_loss_res
