# -*- coding: utf-8 -*-
# @Time    : 2019/6/9 8:25
# @Author  : chenhao
# @FileName: main.py
# @Software: PyCharm
# @Desc: 模型运行主函数（日志记录还没有实现）
from Config import args
import torch
from test import test
from train import train

test_sess = test()
train_sess = train()

torch.manual_seed(args.seed)

if args.test_only:
    test_sess.run()
else:
    train_sess.run()



