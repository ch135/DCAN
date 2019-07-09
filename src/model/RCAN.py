# -*- coding: utf-8 -*-
# @Time    : 2019/6/23 9:57
# @Author  : chenhao
# @FileName: RCAN.py
# @Software: PyCharm
# @Desc: 残差版本Net
import torch.nn as nn
from Config import args
import torch
import math


# 残差块
class _Res_block(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(_Res_block, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=3, padding=1, stride=1)
        self.relu = nn.ReLU()
        if args.n_feats == 256:
            self.res_scale = args.res_scale1
        else:
            self.res_scale = args.res_scale2

    def forward(self, input):
        conv1 = self.relu(self.conv2d(input))
        conv2 = self.conv2d(conv1)
        conv2 *= self.res_scale
        out = torch.add(input, conv2)
        return out


# 主网络模型
class RCAN(nn.Module):
    def __init__(self):
        super(RCAN, self).__init__()
        self.channel_in = args.n_channels
        self.channel_out = args.n_feats
        self.n_resblocks = args.n_resblocks
        self.scale = args.scale
        self.conv2d_1 = nn.Conv2d(in_channels=self.channel_in, out_channels=self.channel_out, kernel_size=3, padding=1,
                                  stride=1)
        self.res_layer = self.make_layer(_Res_block, self.n_resblocks, self.channel_out, self.channel_out)
        self.conv2d_2 = nn.Conv2d(in_channels=self.channel_out, out_channels=self.channel_out, kernel_size=3, padding=1,
                                  stride=1)
        self.conv2d_3 = nn.Conv2d(in_channels=self.channel_out, out_channels=self.channel_in, kernel_size=3, padding=1,
                                  stride=1)
        self.conv2d_4 = nn.Conv2d(in_channels=self.channel_in, out_channels=self.channel_in * self.scale * self.scale,
                                  kernel_size=3, padding=1, stride=1)  # lr1 通道转化
        self.conv2d_5 = nn.Conv2d(in_channels=self.channel_out, out_channels=self.channel_in * self.scale * self.scale,
                                  kernel_size=3, padding=1, stride=1)  # hr 通道转化
        self.sr = nn.Sequential(
            nn.PixelShuffle(self.scale),  # 亚像素卷积;没有参数w和b
            nn.Conv2d(in_channels=self.channel_in, out_channels=self.channel_in, kernel_size=3, stride=1, padding=1,
                      bias=False)
        )
        self.conv2d_6 = nn.Conv2d(in_channels=self.channel_in, out_channels=self.channel_in, kernel_size=3, padding=1,
                                  stride=1)

        # 参数初始化
        for m in self.modules():
            # if isinstance(m, nn.Conv2d):
            #     nn.init.kaiming_uniform_(m.weight.data)
            #     m.weight.data = m.weight.data.double()
            #     if m.bias is not None:
            #         m.bias.data.zero_()
            #         m.bias.data = m.bias.data.double()

            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.weight.data = m.weight.data.double()
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.data = m.bias.data.double()

    def make_layer(self, block, n_blocks, channel_in, channel_out):
        layers = []
        for i in range(n_blocks):
            layers.append(block(channel_in, channel_out))
        return nn.Sequential(*layers)

    def forward(self, input):
        # ResNet
        out1 = self.conv2d_1(input)
        out2 = self.res_layer(out1)
        out3 = self.conv2d_2(out2)
        res_body = torch.add(out1, out3)

        lr1 = self.conv2d_3(res_body)

        # lsr sub-pixel
        lsr = torch.sub(input, lr1)
        lsr = self.conv2d_4(lsr)
        lsr = self.sr(lsr)
        lsr = self.conv2d_6(lsr)
        # hsr sub-pixel
        hr = self.conv2d_5(res_body)
        hsr = self.sr(hr)

        hsr = torch.add(hsr, lsr)
        return input, lr1, hsr, lsr
