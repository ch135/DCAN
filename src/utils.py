# -*- coding: utf-8 -*-
# @Time    : 2019/5/31 8:58
# @Author  : chenhao
# @FileName: utils.py
# @Software: PyCharm
# @Desc: 模型工具包
from Config import args
import os,time
import numpy as np
from skimage.measure import compare_ssim as ssim, compare_psnr as psnr
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.image as pimage
from math import sqrt
import math
import logging

class tool():
    def __init__(self):
        self.train_dir = args.train_dir
        self.valid_dir = args.valid_dir
        self.test_dir = args.test_dir
        self.train_name = args.data_train
        self.valid_name = args.data_valid
        self.test_name = args.data_test
        self.train_LR = args.LR_train
        self.train_HR = args.HR_train
        self.valid_LR = args.LR_valid
        self.valid_HR = args.HR_valid
        self.scale = args.scale
        self.result_Path = args.result_path
        self.data_path = args.data_path
        # reduce=True: 输出一个标量误差(False 时输出向量误差)；size_average=True: 误差求均(False 误差求总和);前两个参数
        # 将被reduction="mean"代替
        # self.l1_fn = nn.L1Loss(reduce=True, size_average=True, reduction="mean")
        self.l1_fn = nn.L1Loss(reduction="mean")
        self.MSE_fn = nn.MSELoss(reduction="mean")
        self.ssim_path = args.ssim_path
        self.psnr_path = args.psnr_path
        self.log_path = args.log_path

    # 获取文件路径
    def getPath(self, style="train"):
        LR_root = None
        HR_root = None
        LR_paths = []
        HR_paths = []
        if style == "train":
            LR_root = "{0}/{1}/{2}/X{3}".format(self.train_dir, self.train_name, self.train_LR, self.scale)
            HR_root = "{0}/{1}/{2}".format(self.train_dir, self.train_name, self.train_HR)
        elif style == "valid":
            LR_root = "{0}/{1}/{2}/X{3}".format(self.valid_dir, self.valid_name, self.valid_LR, self.scale)
            HR_root = "{0}/{1}/{2}".format(self.valid_dir, self.valid_name, self.valid_HR)
        elif style == "test":
            HR_root = "{0}/{1}".format(self.test_dir, self.test_name)

        if LR_root == None:
            HR_paths = list(os.path.join(HR_root, name)for name in os.listdir(HR_root))
            # for file1 in os.listdir(HR_root):
            #     path1 = os.path.join(HR_root, file1)
            #     HR_paths.append(path1)
        else:
            for file in os.listdir(HR_root):
                # path1 = os.path.join(LR_root, file1)
                path2 = os.path.join(HR_root, file)
                path1 = os.path.join(LR_root, "{0}x{1}.{2}".format(file.split('.')[0], self.scale, file.split('.')[1]))
                LR_paths.append(path1)
                HR_paths.append(path2)

        return LR_paths, HR_paths

    # 获取指定数据集数量
    def getDataNumber(self, style="valid"):
        if style == "train":
            root = "{0}/{1}/{2}".format(self.train_dir, self.train_name, self.train_HR)
        elif style == "valid":
            root = "{0}/{1}/{2}".format(self.valid_dir, self.valid_name, self.valid_HR)
        elif style == "test":
            root = "{0}/{1}".format(self.test_dir, self.test_name)
        number = len(os.listdir(root))
        return number

    # 计算图像psnr; psnr=10*log10(MAX**2/MSE)
    def getPsnr(self, img1, img2):
        img1 = img1.byte().cpu().data.numpy()
        img2 = img2.byte().cpu().data.numpy()
        psnrs = []
        PIXEL_MAX = 1
        if (len(img1.shape) == 4):
            img1 = np.transpose(img1, (0, 2, 3, 1))
            img2 = np.transpose(img2, (0, 2, 3, 1))
        else:
            img1 = np.transpose(img1, (1, 2, 0))
            img2 = np.transpose(img2, (1, 2, 0))

        for i in range(img1.shape[0]):
            mse = np.mean((img1[i] / 255. - img2[i] / 255.) ** 2)
            if (mse < 1.0e-10):
                psnrs.append(100)
            else:
                ipsnr = 20 * math.log10(PIXEL_MAX/math.sqrt(mse))
                psnrs.append(ipsnr)
        return np.mean(psnrs)

    # 计算图像ssim
    def getSSIM(self, img1, img2):
        img1 = img1.cpu().data.numpy()
        img2 = img2.cpu().data.numpy()
        ssims = []
        if (len(img1.shape) == 4):
            img1 = np.transpose(img1, (0, 2, 3, 1))
            img2 = np.transpose(img2, (0, 2, 3, 1))
        else:
            img1 = np.transpose(img1, (1, 2, 0))
            img2 = np.transpose(img2, (1, 2, 0))

        for i in range(img1.shape[0]):
            ssims.append(ssim(img1[i], img2[i], multichannel=True))
        return np.mean(ssims)

    # 计算绝对值误差
    def getMAE(self, data1, data2):
        return self.l1_fn(data1, data2)

    # 计算均方误差
    def getMSE(self, data1, data2):
        return self.MSE_fn(data1, data2)

    # 计算均方根误差
    def getRMSE(self, data1, data2):
        return sqrt(self.getMSE(data1, data2))

    # 绘制结果曲线
    # data: 结果数组; layer: 标签数组;有四种颜色，最多画4条曲线
    def resultShow(self, datas, labels, name):
        colors = ['r', 'k', 'y', 'c']  # 曲线颜色
        x = np.arange(0, len(datas[0]))
        if not os.path.exists(self.result_Path):
            os.makedirs(self.result_Path)
        for data, label, color in zip(datas, labels, colors):
            plt.plot(x, data, color, label=label)
        plt.legend(loc=0)
        if name==None:
            data_time = time.strftime("%Y-%m-%d_%H_%M", time.localtime())
        else:
            data_time = name
        plt.savefig("{0}/{1}.png".format(self.result_Path, data_time))
        plt.show()

    # 测试结果显示
    def testShow(self, datas):
        psnrs = []
        ssims = []
        col = 4  # col: 列; row: 行
        if (len(datas) % col == 0):
            row = len(datas) / col
        else:
            row = len(datas) / col + 1
        for i in range(len(datas)):
            dict = datas[i]
            psnrs.append(dict["psnr"])
            ssims.append(dict["ssim"])
            image = pimage.imread(dict["path"])
            plt.subplot(row, col, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.xlabel("psnr/ssim: {0}/{1}".format(np.around(dict["psnr"], decimals=2), np.around(dict["ssim"], decimals=2)), )
            plt.imshow(image)
        plt.tight_layout()

        # 在图片中画表格，显示不出来
        # col_labels = ["PSNR", "SSIM"]
        # row_labels = [self.test_name]
        # table_vals = [[np.around(np.mean(psnrs), decimals=2), np.around(np.mean(ssims), decimals=2)]]
        # plt.table(cellText=table_vals, colWidths=[0.1] * 3, rowLabels=row_labels, colLabels=col_labels, loc="best")

        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        plt.savefig(self.data_path + "/" + self.test_name + ".png")

        if not os.path.exists(self.data_path + "/"+self.test_name+".txt"):
            with open(self.data_path + "/"+self.test_name+".txt", "w") as file:
                file.write("\t\tPSNR\tSSIM\n")
                file.write(self.test_name+"\t{0}\t{1}".format(np.around(np.mean(psnrs), decimals=2), np.around(np.mean(ssims), decimals=2)))
                file.write("\n")
        else:
            with open(self.data_path + "/" + self.test_name + ".txt", "a") as file:
                file.write(self.test_name + "\t{0}\t{1}".format(np.around(np.mean(psnrs), decimals=2), np.around(np.mean(ssims), decimals=2)))
                file.write("\n")
        # plt.show()

    def getBestSSIM(self):
        ssim = 0.0
        if not os.path.exists(self.ssim_path):
            if not os.path.exists(self.ssim_path.split("ssim")[0]):
                os.makedirs(self.ssim_path.split("ssim")[0])
            with open(self.ssim_path, "w") as file:
                file.write("best_ssim\t0.0")
        else:
            with open(self.ssim_path, "r") as file:
                data = file.read()
                ssim = data.split()[1]
                file.close()

        return np.float64(ssim)

    def getBestPsnr(self):
        psnr = 0.0
        epoch = 0
        if not os.path.exists(self.psnr_path):
            if not os.path.exists(self.psnr_path.split("psnr")[0]):
                os.makedirs(self.psnr_path.split("psnr")[0])
            with open(self.psnr_path, "w") as file:
                file.write("Epoch:\t0\tbest_psnr\t0.0")
                file.close()
        else:
            with open(self.psnr_path, "r") as file:
                data = file.read()
                epoch = data.split()[1]
                psnr = data.split()[3]
                file.close()

        return np.int(epoch), np.float64(psnr)

    def saveBestSSIM(self, best_ssim):
        with open(self.ssim_path, "r+") as file1:
            data = file1.read()
            data = data.split()[0]+"\t"+str(best_ssim)

        with open(self.ssim_path, "r+") as file2:
            file2.write(data)

    def saveBestPsnr(self, epoch, best_psnr):
        with open(self.psnr_path, "r+") as file1:
            data = file1.read()
            data = data.split()[0]+"\t"+str(epoch)+data.split()[2]+"\t"+str(best_psnr)

        with open(self.psnr_path, "r+") as file2:
            file2.write(data)

    # 日志输出到控制台和日志文件中
    def log(self):
        path = self.log_path
        if not os.path.exists(path):
            os.makedirs(path)
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)  # 输出到不同平台的总开关

        file = logging.FileHandler(path+'/log.txt', mode='a')  # 日志输出到文件的设置
        file.setLevel(logging.DEBUG)

        ch = logging.StreamHandler()  # 日志输出到控制台的设置
        ch.setLevel(logging.WARNING)

        formatter = logging.Formatter('%(message)s')  # 日志输出格式设置
        file.setFormatter(formatter)
        ch.setFormatter(formatter)

        # 将不同控制模块添加到总控制台
        logger.addHandler(file)
        logger.addHandler(ch)