# -*- coding: utf-8 -*-
# @Time    : 2019/6/9 9:02
# @Author  : chenhao
# @FileName: __init__.py
# @Software: PyCharm
# @Desc:
import torchvision.models as models
import torch
import torch.nn as nn
from utils import tool
import numpy as np
from Config import args

tool = tool()
cuda_gpu = torch.cuda.is_available()
id_GPUs = args.id_GPUs

# calculate content loss by VGG16；data=[HR, SR]
def content_loss(data,pretrained=True):
    model = models.vgg16(pretrained).features
    if cuda_gpu:
        model.cuda(device=id_GPUs[1])
    for m in model:
        if isinstance(m, nn.Conv2d):
            m.weight.data = m.weight.data.double()
            if m.bias is not None:
                m.bias.data = m.bias.data.double()

    content_w = [0.1, 0.8, 0.1]
    index = 0
    loss = []
    for i in range(len(model)):
        data[0] = model[i](data[0])  # HR features
        data[1] = model[i](data[1])  # SR features
        if i in [2, 7, 12]:  # the features in 2,4,6 con2d layer from VGG16
            loss.append(tool.getRMSE(data[0], data[1]) * content_w[index])
            index += 1
    return np.mean(loss)


# HR:原始高分辨率图像；SR: SR 图像；LR1: 相减之后的特征矩阵；Residual: LR经过一个卷积的特征矩阵
def getMultiLoss(HR, SR, LR1, Residual):
    loss1 = tool.getMAE(Residual, LR1)
    loss2 = tool.getMAE(HR, SR)
    c_loss = content_loss([HR, SR])
    loss_w = [0.2, 1.0, 1.0]
    loss = loss1 * loss_w[0] + loss2 * loss_w[1] + c_loss * loss_w[2]
    # loss: 总的loss; loss1: 低频信息loss; loss2: SR loss; content_loss: 内容 Loss
    return loss, loss1, loss2, c_loss


# 获取SR loss; 对应指标 PSNR
def getSimpleLoss(HR, SR):
    loss = tool.getMSE(HR, SR)
    return loss
