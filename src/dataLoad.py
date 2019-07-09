# -*- coding: utf-8 -*-
# @Time    : 2019/6/9 22:20
# @Author  : chenhao
# @FileName: dataLoad.py
# @Software: PyCharm
# @Desc: 数据加载
from Config import args
import numpy as np
import cv2, random
from utils import tool

tool = tool()


class Load():
    def __init__(self):
        self.batch_size = args.batch_size
        self.patch_size = args.patch_size
        self.scale = args.scale
        self.valid_batch_size = args.valid_batch_size

    def getTrainData(self):
        trainLRPath, trainHRPath = tool.getPath("train")
        x_train, y_train = self.LoadData(trainLRPath, trainHRPath)
        return x_train, y_train

    def getValidData(self):
        validLRPath, validHRPath = tool.getPath("valid")
        x_valid, y_valid = self.LoadData(validLRPath, validHRPath)
        return x_valid, y_valid

    def getTestData(self, path):
        hr = cv2.imread(path)
        hr_data = []
        lr_data = []
        row = random.randrange(hr.shape[0] - self.patch_size)
        col = random.randrange(hr.shape[1] - self.patch_size)
        # if row + self.patch_size > hr.shape[0]:
        #     row = hr.shape[0] - self.patch_size
        # if col + self.patch_size > hr.shape[1]:
        #     col = hr.shape[1] - self.patch_size
        hr_data.append(hr[row:row + self.patch_size, col:col + self.patch_size])
        lr_data.append(cv2.resize(src=hr[row:row + self.patch_size, col:col + self.patch_size], dsize=(int(
            self.patch_size / self.scale), int(self.patch_size / self.scale)), interpolation=cv2.INTER_CUBIC))

        hr_data = np.array(hr_data, dtype=np.float64).transpose((0, 3, 1, 2))
        lr_data = np.array(lr_data, dtype=np.float64).transpose((0, 3, 1, 2))
        return lr_data, hr_data

    def LoadData(self, LRpaths, HRpaths):
        LR_data = []
        HR_data = []
        ids = np.random.randint(0, len(HRpaths), self.batch_size)
        # if((index+1)*batch_size>len(HRpaths)):
        #     end = len(HRpaths)
        # else:
        #     end = (index+1)*batch_size
        # indexs = np.arange(index*batch_size, end)
        for id in ids:
            hr = cv2.imread(HRpaths[id])
            lr = cv2.imread(LRpaths[id])
            row = random.randrange(lr.shape[0] - self.patch_size)
            col = random.randrange(lr.shape[1] - self.patch_size)
            # if row + self.patch_size > lr.shape[0]:
            #     row = lr.shape[0] - self.patch_size
            # if col + self.patch_size > lr.shape[1]:
            #     col = lr.shape[1] - self.patch_size
            # if (self.scale * (row + self.patch_size) > hr.shape[0] or self.scale * (col + self.patch_size) > hr.shape[1]):
            #     print(HRpaths[id])
            # if (row + self.patch_size > lr.shape[0] or col + self.patch_size > lr.shape[1]):
            #     print(LRpaths[id])
            LR_data.append(lr[row:row + self.patch_size, col:col + self.patch_size])
            HR_data.append(hr[self.scale * row:self.scale * (row + self.patch_size),
                           self.scale * col:self.scale * (col + self.patch_size)])

        LR_data = np.array(LR_data, dtype=np.float64).transpose((0, 3, 1, 2))
        HR_data = np.array(HR_data, dtype=np.float64).transpose((0, 3, 1, 2))
        return LR_data, HR_data
