# -*- coding: utf-8 -*-
# @Time    : 2019/6/4 11:20
# @Author  : chenhao
# @FileName: __init__.py
# @Software: PyCharm
# @Desc: 网络模型
import torch.nn as nn
from Config import args
import torch
import math


# 密集块
class _Dense_Block(nn.Module):
    def __init__(self, channel_in):
        super(_Dense_Block, self).__init__()
        self.relu = nn.ReLU()
        self.increase_rate = args.increase_rate
        self.conv1 = nn.Conv2d(in_channels=channel_in, out_channels=self.increase_rate, kernel_size=3, stride=1,
                               padding=1)
        self.conv2 = nn.Conv2d(in_channels=self.increase_rate * 1, out_channels=self.increase_rate, kernel_size=3,
                               stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=self.increase_rate * 2, out_channels=self.increase_rate, kernel_size=3,
                               stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=self.increase_rate * 3, out_channels=self.increase_rate, kernel_size=3,
                               stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=self.increase_rate * 4, out_channels=self.increase_rate, kernel_size=3,
                               stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=self.increase_rate * 5, out_channels=self.increase_rate, kernel_size=3,
                               stride=1, padding=1)
        self.conv7 = nn.Conv2d(in_channels=self.increase_rate * 6, out_channels=self.increase_rate, kernel_size=3,
                               stride=1, padding=1)
        self.conv8 = nn.Conv2d(in_channels=self.increase_rate * 7, out_channels=self.increase_rate, kernel_size=3,
                               stride=1, padding=1)

    def forward(self, input):
        conv1 = self.relu(self.conv1(input))

        conv2 = self.relu(self.conv2(conv1))
        cout2_dense = self.relu(torch.cat([conv1, conv2], 1))

        conv3 = self.relu(self.conv3(cout2_dense))
        cout3_dense = self.relu(torch.cat([conv1, conv2, conv3], 1))

        conv4 = self.relu(self.conv4(cout3_dense))
        cout4_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4], 1))

        conv5 = self.relu(self.conv5(cout4_dense))
        cout5_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4, conv5], 1))

        conv6 = self.relu(self.conv6(cout5_dense))
        cout6_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4, conv5, conv6], 1))

        conv7 = self.relu(self.conv7(cout6_dense))
        cout7_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4, conv5, conv6, conv7], 1))

        conv8 = self.relu(self.conv8(cout7_dense))
        cout8_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8], 1))

        return cout8_dense


# 网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.increase_rate = args.increase_rate
        self.n_channels = args.n_channels
        self.scale = args.scale
        self.max_rgb = args.rgb_max
        self.min_rgb = args.rgb_min
        self.lowlevel = nn.Conv2d(in_channels=self.n_channels, out_channels=self.increase_rate * 8, kernel_size=3,
                                  stride=1, padding=1)
        self.relu = nn.ReLU()
        self.denseblock1 = self.make_layer(_Dense_Block, self.increase_rate * 8 * 1)
        self.denseblock2 = self.make_layer(_Dense_Block, self.increase_rate * 8 * 2)
        self.denseblock3 = self.make_layer(_Dense_Block, self.increase_rate * 8 * 3)
        self.denseblock4 = self.make_layer(_Dense_Block, self.increase_rate * 8 * 4)
        self.denseblock5 = self.make_layer(_Dense_Block, self.increase_rate * 8 * 5)
        self.denseblock6 = self.make_layer(_Dense_Block, self.increase_rate * 8 * 6)
        self.denseblock7 = self.make_layer(_Dense_Block, self.increase_rate * 8 * 7)
        self.denseblock8 = self.make_layer(_Dense_Block, self.increase_rate * 8 * 8)

        self.bottleneck1 = nn.Conv2d(in_channels=self.increase_rate * 8 * 9,  # 瓶颈层1,生成的特征矩阵用于和 lowlevel 相减
                                     out_channels=self.increase_rate * 8, kernel_size=1,
                                     stride=1, padding=0, bias=False)
        self.bottleneck2 = nn.Conv2d(in_channels=self.increase_rate * 8,  # 瓶颈层2,生成的特征矩阵作为亚像素卷积的输入
                                     out_channels=self.n_channels * self.scale ** 2, kernel_size=1,
                                     stride=1, padding=0, bias=False)
        self.sr = nn.Sequential(
            nn.PixelShuffle(self.scale),  # 亚像素卷积;没有参数w和b
            nn.Conv2d(in_channels=self.n_channels, out_channels=self.n_channels, kernel_size=3, stride=1, padding=1,
                      bias=False)
        )

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

    def make_layer(self, block, channel_in):
        layers = []
        layers.append(block(channel_in))
        return nn.Sequential(*layers)

    # 控制像素范围在0~255; 等同于 torch.clamp()
    def changValue(self, datas):
        for index, data in enumerate(datas):
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    for k in range(data.shape[2]):
                        for m in range(data.shape[4]):
                            if (data[i][j][k][m] < self.min_rgb):
                                data[i][j][k][m] = self.min_rgb
                            elif (data[i][j][k][m] > self.max_rgb):
                                data[i][j][k][m] = self.max_rgb
            datas[index] = data
        return datas

    def forward(self, input):    # *input 会使输入不是 tensor
        residual = self.relu(self.lowlevel(input))

        out = self.denseblock1(residual)
        concat = torch.cat([residual, out], 1)
        out = self.denseblock2(concat)
        concat = torch.cat([concat, out], 1)
        out = self.denseblock3(concat)
        concat = torch.cat([concat, out], 1)
        out = self.denseblock4(concat)
        concat = torch.cat([concat, out], 1)
        out = self.denseblock5(concat)
        concat = torch.cat([concat, out], 1)
        out = self.denseblock6(concat)
        concat = torch.cat([concat, out], 1)
        out = self.denseblock7(concat)
        concat = torch.cat([concat, out], 1)
        out = self.denseblock8(concat)
        out = torch.cat([concat, out], 1)

        out = self.bottleneck1(out)
        sub_out = torch.sub(residual, out)
        out1 = self.bottleneck2(out)
        out2 = self.bottleneck2(sub_out)

        out1 = self.sr(out1)  # 原始特征矩阵生成的SR图像
        out2 = self.sr(out2)  # 相减特征生成的SR图像
        out1 = torch.add(out1, out2)  # 最终生成的 SR 图像

        # residual = torch.clamp(residual, min=self.min_rgb, max=self.max_rgb)
        # sub_out = torch.clamp(sub_out, min=self.min_rgb, max=self.max_rgb)
        # out1 = torch.clamp(out1, min=self.min_rgb, max=self.max_rgb)

        return residual, out, out1
