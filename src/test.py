# -*- coding: utf-8 -*-
# @Time    : 2019/6/10 10:02
# @Author  : chenhao
# @FileName: test.py
# @Software: PyCharm
# @Desc: 模型测试
from Config import args
import model
import model.DCAN_change as cmodel
import model.RCAN as rcan
from utils import tool
from torch.autograd import Variable
import torch
import torch.nn as nn
import os
from dataLoad import Load

tool = tool()
# cnn = model.Net()
# cnn = cmodel.Net()  # 改版密集网络模型
cnn = rcan.RCAN()   # 残差网络版本网络模型

class test():
    def __init__(self):
        self.model_path = args.model_path.format(args.scale)
        self.cuda_gpu = torch.cuda.is_available()
        self.id_GPUs = args.id_GPUs

    def run(self):
        if not os.path.exists(self.model_path):
            print("place train before test!")
        else:
            result = []
            if self.cuda_gpu:
                model = cnn.cuda(device=self.id_GPUs[0])
                model = nn.DataParallel(model, device_ids=self.id_GPUs, output_device=self.id_GPUs[1])
                model.load_state_dict(torch.load(self.model_path))

            load = Load()
            _, paths = tool.getPath("test")
            for path in paths:
                dict = {}
                x_test, y_test = load.getTestData(path)
                if self.cuda_gpu:
                    x_test = Variable(torch.from_numpy(x_test)).cuda(self.id_GPUs[1])
                    y_test = Variable(torch.from_numpy(y_test)).cuda(self.id_GPUs[1])
                else:
                    x_test = Variable(torch.from_numpy(x_test))
                    y_test = Variable(torch.from_numpy(y_test))
                _, _, out, _ = model(x_test)
                psnr = tool.getPsnr(y_test, out)
                ssim = tool.getSSIM(out, y_test)

                dict["path"] = path
                dict["psnr"] = psnr
                dict["ssim"] = ssim
                result.append(dict)
            tool.testShow(result)
