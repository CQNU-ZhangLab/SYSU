
import time
import datetime
import os
import torch as t
import numpy as np
from __future__ import print_function
import argparse
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from matplotlib import pyplot as plt
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from data_loader import SYSUData, RegDBData, TestData
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb
from model_mine import embed_net
from utils import *
from loss import OriTripletLoss, CenterTripletLoss, CrossEntropyLabelSmooth, TripletLoss_WRT, BarlowTwins_loss, \
    TripletLoss, local_loss_idx, global_loss_idx
from tensorboardX import SummaryWriter

import time
from datetime import datetime
import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
from util.Data import Data
import torch
import models
from util.utils import extract_feature_test
from util.utils import sort_img, imshow
from util.data.market1501_1 import Market1501
from train_trihard import args


parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='sysu', help='dataset name: regdb or sysu]')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')

dataset = args.dataset
if dataset == 'sysu':
    data_path = 'E:/hy/dataset/SYSU-MM01/'
    log_path = args.log_path + 'sysu_log/'
    test_mode = [1, 2]  # thermal to visible
elif dataset == 'regdb':
    data_path = 'E:/hy/dataset/RegDB/'
    log_path = args.log_path + 'regdb_log/'
    test_mode = [1, 2]

query_index = 777
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Baseline = models.init_model(name='resnet50', num_classes=751, loss={'softmax', 'metric'})

# Adf.load_state_dict(t.load('F:/My_ReID/experiment_data/0401/weights/model_444.pt'))
checkpoint = torch.load('E:/gyx/reid_tutorial/log/checkpoint_ep200.pth.tar')
Baseline.load_state_dict(checkpoint['state_dict'])

# testing set
query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=0)

market_train = Market1501(dtype='train', root='E:/gyx/')
market_test = Market1501(dtype='test', root='E:/gyx/')
market_query = Market1501(dtype='query', root='E:/gyx/')

# trainset,num_train_pids,num_train_imgs = market_train._process_dir()
testset, num_test_pids, test_camid, num_test_imgs = market_test._process_dir(
    dir_path='E:/gyx/market1501/bounding_box_test/', relabel=False)
queryset, num_query_pids, query_camid, num_query_imgs = market_query._process_dir(dir_path='E:/gyx/market1501/query/',
                                                                                  relabel=False)

data = Data(args)
test_loader = data.test_loader
query_loader = data.query_loader

# print('queryset',queryset)
# print('testset',testset)

Baseline.eval()
dataset_query, query_label, query_camid, query_path = queryset, num_query_pids, query_camid, num_query_imgs
dataset_test, gallery_label, test_camid, gallery_path = testset, num_test_pids, test_camid, num_test_imgs

# print('query_label',query_label)
# print('gallery_label',gallery_label)
# print('query_path',query_path)
# print('gallery_path',gallery_path)

# Extract feature
query_feature = extract_feature_test(Baseline, query_loader)
gallery_feature = extract_feature_test(Baseline, test_loader)
query_feature = query_feature.cuda()
gallery_feature = gallery_feature.cuda()

i = query_index

index = sort_img(query_feature[i], query_label[i], query_camid[i], gallery_feature, gallery_label, test_camid)

query_path = query_path[i]
query_label = query_label[i]
print(query_path)
print('Top 10 images are as follow:')

try:  # Visualize Ranking Result
    # Graphical User Interface is needed
    fig = plt.figure(figsize=(16, 4))
    ax = plt.subplot(1, 11, 1)
    ax.axis('off')
    imshow(query_path, 'query')
    for i in range(10):
        ax = plt.subplot(1, 11, i + 2)
        ax.axis('off')
        img_path = gallery_path[index[i]]
        label = gallery_label[index[i]]
        imshow(img_path)
        if label == query_label:
            ax.set_title('%d' % (i + 1), color='green')
        else:
            ax.set_title('%d' % (i + 1), color='red')
        print(img_path)
except RuntimeError:
    for i in range(10):
        # log_path = "./show" + '/Log %d.txt' % query_index
        # if not os.path.exists(log_path):
        #     os.system(r"touch {}".format(log_path))
        img_path = gallery_path.imgs[index[i]]
        print(img_path[0])
        # f = open(log_path, 'a')
        # f.write(img_path + '\n')
    print('If you want to see the visualization of the ranking result, graphical user interface is needed.')

fig.savefig("./show/show %d.png" % query_index)