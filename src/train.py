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
import torch
import os
from dataLoad import Load
import numpy as np

tool = tool()
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
        self.loss_path = args.loss_path
        self.weight_decay = args.weight_decay

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

    # 模型验证
    def eval(self, model, input, output):
        model.eval()  # 测试阶段
        _, _, out1, _ = model(input)
        psnr = tool.getPsnr(output, out1)
        return psnr

    def run(self):
        if self.cuda_gpu:
            model = cnn.cuda(device=self.id_GPUs[0])
            model = nn.DataParallel(model, device_ids=self.id_GPUs, output_device=self.id_GPUs[1])
        # 模型加载
        file = self.model_path.split("net")[0]
        if not os.path.exists(file):
            os.makedirs(file)
        elif (os.path.exists(self.model_path)):
            print("loading model...")
            model.load_state_dict(torch.load(self.model_path))

        # 使用 L2正则化
        if self.optimizer == 'ADAM':
            optimizer = Adam(params=model.parameters(), lr=self.lr, betas=self.betas,
                             weight_decay=self.weight_decay)
        elif self.optimizer == 'SGD':
            optimizer = SGD(params=model.parameters(), lr=self.lr, momentum=self.momentum,
                            weight_decay=self.weight_decay)
        elif self.optimizer == 'RMSprop':
            optimizer = RMSprop(params=model.parameters(), lr=self.lr, weight_decay=self.weight_decay)


        best_psnr = tool.getBestPsnr()
        step = tool.getDataNumber("train") // self.batch_size  # 整个数据集训练一遍的次数
        load = Load()
        laybels = ['FinalLoss', 'SLRLoss', 'SHRLoss', 'ContentLoss']
        for i in range(400, self.epoch):
            aloss, lrloss, hrloss, closs = [], [], [], []   # epoch 历史纪录
            Allloss, SLRLoss, SHRLoss, ContentLoss = [], [], [], [] # step 历史记录
            print("Epoch:{0}".format(i))
            if (i != 0 and i % self.decay == 0):
                self.adjust_learning_rate(optimizer, i)
            for j in range(step):
                x_train, y_train = load.getTrainData()
                if self.cuda_gpu:
                    x_train = Variable(torch.from_numpy(x_train)).cuda(self.id_GPUs[1])
                    y_train = Variable(torch.from_numpy(y_train)).cuda(self.id_GPUs[1])
                else:
                    x_train = torch.from_numpy(x_train)
                    y_train = torch.from_numpy(y_train)


                model.train()
                optimizer.zero_grad()  # 初始化梯度为零
                residual, sub_out, out1, lsr = model(x_train)
                finalLoss, loss1, loss2, content_loss = loss.getMultiLoss(y_train, out1, sub_out, residual)

                # L1 正则化，不适用于此场景(参数在gpu0, loss结果在gpu1;只有将loss结果在gpu0输出，此时又用有其他问题)
                # finalLoss += self.L1_Regularization(model)
                # loss1 += self.L1_Regularization(model)
                # loss2 += self.L1_Regularization(model)
                # content_loss += self.L1_Regularization(model)

                finalLoss.backward()
                optimizer.step()        # 参数更新
                psnr = tool.getPsnr(y_train, out1)
                psnr_l = tool.getPsnr(y_train, lsr)
                ssim = tool.getSSIM(out1, y_train)
                Allloss.append(finalLoss.item())
                SLRLoss.append(loss1.item())
                SHRLoss.append(loss2.item())
                ContentLoss.append(content_loss.item())
                print("Step/Steps:{0}/{1}   -AllLoss:{2}  -SLRLoss:{3}  -SHRLoss:{4}  -ContentLoss:{5}   -SSIM:{6}    "
                      "-SR_PSNR:{7}   -LR_PSNR:{8}".format(j, step, finalLoss, loss1, loss2, content_loss, ssim, psnr, psnr_l))


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
                        print("result:{0}, best_psnr:{1}, save result...".format(psnr, best_psnr))
                        torch.save(model.state_dict(), self.model_path)
                        tool.saveBestPsnr(psnr)
                        best_psnr = psnr
                    elif (psnr <= best_psnr):
                        print("result:{0}, best_psnr:{1}, don't save model...".format(psnr, best_psnr))


                    Finaloss = np.mean(Allloss)
                    LRloss = np.mean(SLRLoss)
                    HRloss = np.mean(SHRLoss)
                    CLoss = np.mean(ContentLoss)
                    if not os.path.exists(self.loss_path.split("history")[0]):
                        os.makedirs(self.loss_path.split("history")[0])
                        with open(self.loss_path, "a") as f:    # "a"：没有文件时创建文件；有文件时添加数据
                            f.write("Finalloss\t\tLRloss\t\t\tHRloss\t\t\tCLoss\n")
                            f.write("{0}\t{1}\t{2}\t{3}\n".format(Finaloss, LRloss, HRloss, CLoss))
                            f.close()
                    else:
                        with open(self.loss_path, "a") as f:
                            f.write("{0}\t{1}\t{2}\t{3}\n".format(Finaloss, LRloss, HRloss, CLoss))
                            f.close()
                    # datas = [Allloss, SLRLoss, SHRLoss, ContentLoss]
                    # tool.resultShow(datas, laybels, None)


            if (i % 5 == 0 and i != 0):
                with open(self.loss_path, "r") as f:
                    f.readline()
                    line = f.readlines()
                    for g in range(len(line)):
                        lines = line[g].split("\t")
                        aloss.append(np.float64(lines[0]))
                        lrloss.append(np.float64(lines[1]))
                        hrloss.append(np.float64(lines[2]))
                        closs.append(np.float64(lines[3].split("\n")[0]))
                    f.close()
                edatas = [aloss, lrloss, hrloss, closs]
                tool.resultShow(edatas, laybels, "epoch{0}".format(i))
