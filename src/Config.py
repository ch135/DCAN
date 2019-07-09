# -*- coding: utf-8 -*-
# @Time    : 2019/5/30 20:12
# @Author  : chenhao
# @FileName: Config.py
# @Software: PyCharm
# @Desc: 模型参数配置
import argparse

parser = argparse.ArgumentParser(description="DCAN")

# Hardware specification
parser.add_argument("--n_threads", type=int, default=6, help="the number of threads")
parser.add_argument("--cpu", type=bool, default=False, help="use cpu only")
parser.add_argument("--n_GPUs", type=int, default=2, help="the number of GPU")
parser.add_argument("--id_GPUs", type=list, default=[0, 1], help="the index of GPUs")
parser.add_argument("--seed", type=int, default=1, help="random seed")

# Data specification
parser.add_argument("--train_dir", type=str, default="/media/wangct/E7D0AC3987855C5C/dataset", help="thr dir of train data")
parser.add_argument("--data_train", type=str, default="DIV2K", help="train data name")
parser.add_argument("--LR_train", type=str, default="DIV2K_train_LR_bicubic_c", help="name of LR_train data")
parser.add_argument("--HR_train", type=str, default="DIV2K_train_HR_c", help="name of HR_train data")
parser.add_argument("--valid_dir", type=str, default="/media/wangct/E7D0AC3987855C5C/dataset", help="thr dir of train data")
parser.add_argument("--data_valid", type=str, default="DIV2K", help="valid data name")
parser.add_argument("--LR_valid", type=str, default="DIV2K_valid_LR_bicubic_c", help="name of LR_valid data")
parser.add_argument("--HR_valid", type=str, default="DIV2K_valid_HR_c", help="name of HR_valid data")
parser.add_argument("--test_dir", type=str, default="/media/wangct/E7D0AC3987855C5C/dataset", help="thr dir of train data")
parser.add_argument("--data_test", type=str, default="Set14", help="test data name")
parser.add_argument("--scale", type=int, default=2, help="the scale of super resolution")
parser.add_argument("--rgb_max", type=int, default=255, help="maximum value of RGB")
parser.add_argument("--rgb_min", type=int, default=0, help="minimum value of RGB")
parser.add_argument("--n_channels", type=int, default=3, help="number of input channel")
parser.add_argument("--model_path", type=str, default="../experiment/model/net_X{0}.pkl", help="path of the best model")

# train
parser.add_argument("--model", default="DCAN", help="model name")
parser.add_argument("--res_scale", type=float, default=1.0, help="residual scaling")
parser.add_argument("--shift_mean", default=True, help="subtract pixel mean from the input")
parser.add_argument('--dilation', action='store_true', help='use dilated convolution')
parser.add_argument("--epochs", type=int, default=1000, help="the epoch of training")
parser.add_argument('--batch_size', type=int, default=16, help='input batch size for training')
parser.add_argument("--valid_batch_size", type=int, default=10, help="input batch size of valid")
parser.add_argument("--patch_size", type=int, default=64, help="the number of patch size")
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--decay', type=int, default=100, help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM', choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999), help='ADAM beta')
parser.add_argument('--epsilon', type=float, default=1e-8, help='ADAM epsilon for numerical stability')
parser.add_argument("--save_models", type=bool, default=True, help="save training model")
parser.add_argument("--increase_rate", type=int, default=16, help="thr increase rate of Densenet")
parser.add_argument("--ssim_path", type=str, default="../experiment/train/ssim.txt", help="the path of the best ssim of training")
parser.add_argument("--psnr_path", type=str, default="../experiment/train/psnr.txt", help="the path of the best psnr of training")
parser.add_argument("--loss_path", type=str, default="../experiment/loss/history.txt", help="the path of the train history")
parser.add_argument("--weight_decay", type=int, default=0.01, help="weight1 decay")

# train_RCAN
parser.add_argument("--n_resblocks", type=int, default=32, help="the number of resblock")
parser.add_argument("--n_feats", type=int, default=256, help="the number of feat maps")
parser.add_argument("--res_scale1", type=float, default=0.1, help="residual1 scaling")
parser.add_argument("--res_scale2", type=float, default=1.0, help="residual2 scaling")

# valid
parser.add_argument("--valid_step", type=int, default=1, help="do valid of every N epoch")
# parser.add_argument('--self_ensemble', action='store_true', help='use self-ensemble method for valid')

# test
parser.add_argument('--test_only', action='store_true', help='set this option to test the model')

# Log specification
parser.add_argument("--save", type=str, default="test", help="file name to save")
parser.add_argument('--print_every', type=int, default=100, help='how many batches to wait before logging training status')
parser.add_argument('--save_results', action='store_true', help='save output results')
parser.add_argument('--save_gt', action='store_true',  help='save low-resolution and high-resolution images together')
parser.add_argument("--result_path", type=str, default="../experiment/graph", help="the path graph")
parser.add_argument("--data_path", type=str, default="../experiment/result", help="the path of the image from  the test result")

args = parser.parse_args()
