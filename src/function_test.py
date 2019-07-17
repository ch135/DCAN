# -*- coding: utf-8 -*-
# @Time    : 2019/5/30 21:19
# @Author  : chenhao
# @FileName: function_test.py.py
# @Software: PyCharm
# @Desc: 测试代码
from utils import tool
import torchvision.models as models
import torch
import cv2
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.image as mping
tool = tool()

# model = models.densenet201(pretrained=True)
# print(model)
#
#
# LR_paths, HR_paths = tool.getPath("test")
# ################### 获取路径测试 ######################
# print(LR_paths)
# print(HR_paths)

################### psnr, ssim, mae, mse测试 ######################
# img = cv2.imread("../test/psnr_ssim.png")
# print(tool.getPsnr(img, img))
# print(tool.getSSIM(img, img))
# print(tool.getMAE(img, img))
# print(tool.getMSE(img, img))


# ################## 曲线图方法测试 #######################
# data = np.arange(0, 100, 1).reshape((4, 25))
# layer = ["test1", "test2", "test3", "test4"]
# tool.resultShow(data, layer)

# ################## pytorch 自带密码网络测试 #######################
# model = models.vgg16(pretrained=True)
# print(model)

# ################## pytorch 获取某一层的输出数据测试 #######################
# model = models.vgg16(pretrained=True).features
# net = []
# data = torch.randn(3,3,16,16)
# for i in range(len(model)):
#     data = model[i](data)
#     if i in [2, 7, 12]:     # 获取 VGG16 第2、4、6个卷积层的输出
#         print(data)

# ################## 获取数据集数目测试 #######################
# print(tool.getDataNumber("train"))
#
# # ################## 图像resize测试 #######################
# img = cv2.imread("../test/psnr_ssim.png")
# imgs = cv2.resize(img, dsize=(int(img.shape[1]/4), int(img.shape[0]/4)), interpolation=cv2.INTER_CUBIC)
# print(img.shape)
# print(imgs.shape)
# cv2.imshow("1", img)
# cv2.imshow("2", imgs)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # ################## matplotlib 绘制多张图像测试 #######################

# img = mping.imread("../test/psnr_ssim.png")
# # plt.figure(figsize=(2,2)) # 加上这句话会使图像变难看
# plt.subplot(2,2,1)
# # plt.axis('off')   # 隐藏坐标轴和标题
# plt.xticks([])
# plt.yticks([])
# plt.xlabel("psnr/ssim:23/89")
# plt.imshow(img)
# plt.subplot(222)
# plt.imshow(img)
# plt.savefig("test.png")     # 先保存，在显示
# plt.show()

# # ################## matplotlib画表格测试 #######################
# col_labels = ['SSIM','PSNR']
# row_labels = ["Set5"]
# table_vals = [[11,12]]
# plt.table(cellText=table_vals, colWidths=[0.1]*2,rowLabels=row_labels, colLabels=col_labels,loc="best")
# plt.show()

# # ################## pytorch Loss测试 #######################
# loss = nn.L1Loss(reduce=True, size_average=True)
# input = Variable(torch.randn(1, 3, 3, 4))
# target = Variable(torch.randn(1, 3, 3, 4))
# result = loss(input, target)
# print(input)
# print(target)
# print(result)

# # ################## txt文件读取、更改数据测试 #######################
# tool.saveBestSSIM(12343)
# print(tool.getBestSSIM())

# # ################## 模型参数输出设置 #######################
import model.RCAN as rcan

net = rcan.RCAN()
model_path ="../experiment/model/net_X2.pkl"
net.load_state_dict(torch.load(model_path, map_location="cpu"))

for name, parameters in net.named_parameters():
   print(name, ":", parameters)
