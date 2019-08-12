# -*- coding: utf-8 -*-
# @Time    : 2019/6/9 8:25
# @Author  : chenhao
# @FileName: train.py
# @Software: PyCharm
# @Desc: 训练并优化模型
from Config import args
import model
import model.DCAN_change as cmodel
import model.RCAN as rcan
from utils import tool
from torch.optim import Adam, SGD, RMSprop
from torch.autograd import Variable
import torch.nn as nn
import loss as loss
import os, time, logging, torch
from dataLoad import Load
import numpy as np
from lookahead import Lookahead

tool = tool()
load = Load()
# cnn = model.Net()
# cnn = cmodel.Net()  # 改版密集网络模型
cnn = rcan.RCAN()   # 残差网络版本网络模型


class train():
    def __init__(self):
        self.n_threads = args.n_threads
        self.cpu = args.cpu
        self.n_GPUs = args.n_GPUs
        self.id_GPUs = args.id_GPUs
        self.epoch = args.epochs
        self.batch_size = args.batch_size
        self.valid_batch_size = args.valid_batch_size
        self.patch_size = args.patch_size
        self.lr = args.lr
        self.decay = args.decay
        self.gamma = args.gamma
        self.optimizer = args.optimizer
        self.momentum = args.momentum  # SGD momentum
        self.betas = args.betas  # ADAM betas
        self.epsilon = args.epsilon  # ADAM epsilon
        self.save_models = args.save_models
        self.cuda_gpu = torch.cuda.is_available()
        self.optimizer = args.optimizer
        self.valid_step = args.valid_step
        self.model_path = args.model_path.format(args.scale)
        self.ssim_path = args.ssim_path
        self.weight_decay = args.weight_decay
        self.log_path = args.log_path
        tool.log()  # 日志规则

    # 学习效率调整
    def adjust_learning_rate(self, optimizer, epoch):
        lr = self.lr * (self.gamma ** (epoch // self.decay))
        print("Epoch {0}:   lr={1}".format(epoch, lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # L1 正则化
    def L1_Regularization(self, model):
        regularization = 0
        for param in model.parameters():
            regularization += torch.sum(torch.abs(param))
        return self.weight_decay * regularization

    # 预训练模型加载
    def loadModel(self):
        file = self.model_path.split("net")[0]
        if not os.path.exists(file):
            os.makedirs(file)
        elif (os.path.exists(self.model_path)):
            print("loading model...")
            model.load_state_dict(torch.load(self.model_path))

    # 模型验证
    def eval(self, model, input, output):
        model.eval()  # 测试阶段
        _, _, out1, _ = model(input)
        psnr = tool.getPsnr(output, out1)
        return psnr

    def run(self):
        epoch, best_psnr = tool.getBestPsnr()   # 获取最好历史记录
        step = tool.getDataNumber("train") // self.batch_size  # 遍历整个数据集需要的次数
        if self.cuda_gpu:
            model = cnn.cuda(device=self.id_GPUs[0])
            model = nn.DataParallel(model, device_ids=self.id_GPUs, output_device=self.id_GPUs[0])
        self.loadModel()    # 加载预训练模型
        # 使用 L2正则化
        if self.optimizer == 'ADAM':
            optimizer = Adam(params=model.parameters(), lr=self.lr, betas=self.betas,
                             weight_decay=self.weight_decay)
        elif self.optimizer == 'SGD':
            optimizer = SGD(params=model.parameters(), lr=self.lr, momentum=self.momentum,
                            weight_decay=self.weight_decay)
        elif self.optimizer == 'RMSprop':
            optimizer = RMSprop(params=model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        optimizer = Lookahead(optimizer, k=5, alpha=0.5)    # lookahead 优化策略
        '''
            L1 正则化，不适用于此场景(参数在gpu0, loss结果在gpu1;只有将loss结果在gpu0输出，此时又用有其他问题)
            finalLoss += self.L1_Regularization(model)
            loss1 += self.L1_Regularization(model)
            loss2 += self.L1_Regularization(model)
            content_loss += self.L1_Regularization(model)
        '''
        for i in range(epoch, self.epoch):
            logging.info("{} Epoch:{}".format(time.strftime('%Y-%m-%d %H-%M-%S', time.localtime()), i))
            aloss=srloss=slrloss=closs=0
            if (i != 0 and i % self.decay == 0):
                self.adjust_learning_rate(optimizer, i)
            for j in range(step):
                x_train, y_train = load.getTrainData()
                if self.cuda_gpu:
                    x_train = Variable(torch.from_numpy(x_train)).cuda(self.id_GPUs[0])
                    y_train = Variable(torch.from_numpy(y_train)).cuda(self.id_GPUs[0])
                else:
                    x_train = torch.from_numpy(x_train)
                    y_train = torch.from_numpy(y_train)
                model.train()
                optimizer.zero_grad()  # 初始化梯度为零
                residual, sub_out, out1, lsr = model(x_train)
                finalLoss, loss1, loss2, content_loss = loss.getMultiLoss(y_train, out1, sub_out, residual)
                finalLoss.backward()
                optimizer.step()        # 参数更新
                psnr = tool.getPsnr(y_train, out1)
                logging.info("Step/Steps:{}/{}\t-AllLoss:{:.4f}\t-SLRLoss:{:.4f}\t-SHRLoss:{:.4f}\t-ContentLoss:{:.4f}\t-SR_PSNR:{}".format(j, step, finalLoss, loss1, loss2, content_loss, psnr))
                aloss += finalLoss
                slrloss += loss1
                srloss += loss2
                closs += content_loss
                if (j == step - 1):
                    step_valid = tool.getDataNumber("valid") // self.valid_batch_size
                    psnrs = []
                    for k in range(step_valid):
                        x_valid, y_valid = load.getValidData()
                        if self.cuda_gpu:
                            x_valid = Variable(torch.from_numpy(x_valid)).cuda(self.id_GPUs[1])
                            y_valid = Variable(torch.from_numpy(y_valid)).cuda(self.id_GPUs[1])
                        else:
                            x_valid = Variable(torch.from_numpy(x_valid))
                            y_valid = Variable(torch.from_numpy(y_valid))
                        psnr = self.eval(model, x_valid, y_valid)
                        psnrs.append(psnr.item())
                    psnr = np.mean(psnrs)
                    if (psnr > best_psnr):
                        logging.info("result:{0}, best_psnr:{1}, save result...".format(psnr, best_psnr))
                        torch.save(model.state_dict(), self.model_path)
                        tool.saveBestPsnr(i, psnr)
                        best_psnr = psnr
                    elif (psnr <= best_psnr):
                        logging.info("result:{0}, best_psnr:{1}, don't save model...".format(psnr, best_psnr))
                    if not os.path.exists(self.log_path+'/history.txt'):
                        with open(self.log_path+'/history.txt', "a") as f:  # "a"：没有文件时创建文件；有文件时添加数据
                            f.write("Finalloss\tLRloss\tHRloss\tCLoss\n")
                            f.write("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(aloss/step, slrloss/step, srloss/step, closs/step))
                    else:
                        with open(self.log_path+'/history.txt', "a") as f:
                            f.write("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(aloss/step, slrloss/step, srloss/step, closs/step))
