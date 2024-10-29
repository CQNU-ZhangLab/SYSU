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
from re_rank import random_walk, k_reciprocal
import time
from datetime import datetime
import numpy as np
import scipy.io as io

np.set_printoptions(threshold=np.inf)

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='sysu', help='dataset name: regdb or sysu]')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str,
                    help='network baseline:resnet18 or resnet50')
parser.add_argument('--resume', '-r', default='', type=str,
                    help='resume from checkpoint')  # default 为ckpt 文件名
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--model_path', default='save_model/', type=str,
                    help='model save path')
parser.add_argument('--save_epoch', default=100, type=int,
                    metavar='s', help='save model every 10 epochs')
parser.add_argument('--log_path', default='log/', type=str,
                    help='log save path')
parser.add_argument('--vis_log_path', default='log/vis_log/', type=str,
                    help='log save path')
parser.add_argument('--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=144, type=int,
                    metavar='imgw', help='img width')
parser.add_argument('--img_h', default=288, type=int,
                    metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=8, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=64, type=int,
                    metavar='tb', help='testing batch size')
parser.add_argument('--method', default='base', type=str,
                    metavar='m', help='method type: base or agw')
parser.add_argument('--margin', default=0.3, type=float,
                    metavar='margin', help='triplet loss margin')

parser.add_argument('--seed', default=0, type=int,
                    metavar='t', help='random seed')
parser.add_argument('--gpu', default='0', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')

parser.add_argument('--trial', default=2, type=int,
                    metavar='t', help='trial (only for RegDB dataset)')

parser.add_argument('--num_pos', default=4, type=int,
                    help='num of pos per identity in each modality')
parser.add_argument('--share_net', default=2, type=int,
                    metavar='share', help='[1,2,3,4,5]the start number of shared network in the two-stream networks')

parser.add_argument('--re_rank', default='k_reciprocal', type=str,
                    help='performing reranking. [random_walk | k_reciprocal | no]')
parser.add_argument('--pcb', default='on', type=str, help='performing PCB, on or off')
parser.add_argument('--w_center', default=2.0, type=float, help='the weight for center loss')

parser.add_argument('--local_feat_dim', default=512, type=int,
                    help='feature dimention of each local feature in PCB，256 ')
parser.add_argument('--num_strips', default=6, type=int,
                    help='num of local strips in PCB')

parser.add_argument('--label_smooth', default='on', type=str, help='performing label smooth or not')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

set_seed(args.seed)

dataset = args.dataset
if dataset == 'sysu':
    data_path = 'D:/hy/dataset/SYSU-MM01/'
    log_path = args.log_path + 'sysu_log/'
    test_mode = [1, 2]  # thermal to visible
elif dataset == 'regdb':
    data_path = 'D:/hy/dataset/RegDB/'
    log_path = args.log_path + 'regdb_log/'
    test_mode = [1, 2]

checkpoint_path = args.model_path

if not os.path.isdir(log_path):
    os.makedirs(log_path)
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)
if not os.path.isdir(args.vis_log_path):
    os.makedirs(args.vis_log_path)

suffix = dataset + '_pcb_wc{}'.format(args.w_center)
if args.pcb == 'on':
    suffix = suffix + '_s{}_fd{}'.format(args.num_strips, args.local_feat_dim)

suffix = suffix + '_share{}'.format(args.share_net)
if args.method == 'agw':
    suffix = suffix + '_agw_k{}_p{}_lr_{}'.format(args.num_pos, args.batch_size, args.lr)
else:
    suffix = suffix + '_base_k{}_p{}_lr_{}'.format(args.num_pos, args.batch_size, args.lr)

if not args.optim == 'sgd':
    suffix = suffix + '_' + args.optim

if dataset == 'regdb':
    suffix = suffix + '_trial_{}_testmode{}_to_{}'.format(args.trial, test_mode[0], test_mode[1])

our_method = 'HC_SSL_short_relation'
if dataset == 'sysu':
    suffix = suffix + '_{}search_{}'.format(args.mode, our_method)

sys.stdout = Logger(log_path + suffix + '_os.txt')

vis_log_dir = args.vis_log_path + suffix + '/'

if not os.path.isdir(vis_log_dir):
    os.makedirs(vis_log_dir)
writer = SummaryWriter(vis_log_dir)
print("==========\nArgs:{}\n==========".format(args))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0

end = time.time()

print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Pad(10),
    transforms.RandomCrop((args.img_h, args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])
transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(),
    normalize,
])

end = time.time()
if dataset == 'sysu':
    # training set
    trainset = SYSUData(data_path, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=0)

elif dataset == 'regdb':
    # training set
    trainset = RegDBData(data_path, args.trial, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    if test_mode[0] == 2:
        ### V -> I
        query_img, query_label = process_test_regdb(data_path, trial=args.trial, modal='visible')
        gall_img, gall_label = process_test_regdb(data_path, trial=args.trial, modal='thermal')

    elif test_mode[0] == 1:
        #### I -> V
        query_img, query_label = process_test_regdb(data_path, trial=args.trial, modal='thermal')
        gall_img, gall_label = process_test_regdb(data_path, trial=args.trial, modal='visible')

gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))

# testing data loader
gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

n_class = len(np.unique(trainset.train_color_label))
nquery = len(query_label)
ngall = len(gall_label)

print('Dataset {} statistics:'.format(dataset))
print('  ------------------------------')
print('  subset   | # ids | # images')
print('  ------------------------------')
print('  visible  | {:5d} | {:8d}'.format(n_class, len(trainset.train_color_label)))  # 395 |    22258
print('  thermal  | {:5d} | {:8d}'.format(n_class, len(trainset.train_thermal_label)))  # 395 |    11909
print('  ------------------------------')
print('  query    | {:5d} | {:8d}'.format(len(np.unique(query_label)), nquery))  # 96  3803
print('  gallery  | {:5d} | {:8d}'.format(len(np.unique(gall_label)), ngall))  # 96  301
print('  ------------------------------')
print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

print('==> Building model..')
if args.method == 'base':
    net = embed_net(n_class, no_local='off', gm_pool='on', arch=args.arch, share_net=args.share_net, pcb=args.pcb,
                    local_feat_dim=args.local_feat_dim, num_strips=args.num_strips)
else:
    net = embed_net(n_class, no_local='on', gm_pool='on', arch=args.arch, share_net=args.share_net, pcb=args.pcb)
net.to(device)

cudnn.benchmark = True

if len(args.resume) > 0:
    model_path = checkpoint_path + args.resume
    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['net'])
        print('==> loaded checkpoint {} (epoch {})'
              .format(args.resume, checkpoint['epoch']))
    else:
        print('==> no checkpoint found at {}'.format(args.resume))

# define loss function
if args.label_smooth == 'on':
    criterion_id = nn.CrossEntropyLoss()
else:
    criterion_id = CrossEntropyLabelSmooth(n_class)

if args.method == 'agw':
    criterion_tri = TripletLoss_WRT()
else:
    loader_batch = args.batch_size * args.num_pos
    # criterion_tri= OriTripletLoss(batch_size=loader_batch, margin=args.margin)
    criterion_tri = CenterTripletLoss(batch_size=loader_batch, margin=args.margin)

g_tri_loss = global_loss_idx(batch_size=loader_batch, margin=args.margin)
l_tri_loss = local_loss_idx(batch_size=loader_batch, margin=args.margin)

twinsloss = BarlowTwins_loss(batch_size=loader_batch, margin=args.margin)

criterion_id.to(device)
criterion_tri.to(device)
twinsloss.to(device)

if args.optim == 'sgd':
    if args.pcb == 'on':
        ignored_params = list(map(id, net.local_conv_list.parameters())) \
                         + list(map(id, net.fc_list.parameters()))

        base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

        optimizer = optim.SGD([
            {'params': base_params, 'lr': 0.1 * args.lr},
            {'params': net.local_conv_list.parameters(), 'lr': args.lr},
            {'params': net.fc_list.parameters(), 'lr': args.lr}
        ],
            weight_decay=5e-4, momentum=0.9, nesterov=True)
    else:
        ignored_params = list(map(id, net.bottleneck.parameters())) \
                         + list(map(id, net.classifier.parameters()))

        base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

        optimizer = optim.SGD([
            {'params': base_params, 'lr': 0.1 * args.lr},
            {'params': net.bottleneck.parameters(), 'lr': args.lr},
            {'params': net.classifier.parameters(), 'lr': args.lr}],
            weight_decay=5e-4, momentum=0.9, nesterov=True)


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch < 10:
        lr = args.lr * (epoch + 1) / 10
    elif epoch >= 10 and epoch < 20:
        lr = args.lr
    elif epoch >= 20 and epoch < 50:
        lr = args.lr * 0.1
    elif epoch >= 50:
        lr = args.lr * 0.01

    optimizer.param_groups[0]['lr'] = 0.1 * lr
    for i in range(len(optimizer.param_groups) - 1):
        optimizer.param_groups[i + 1]['lr'] = lr

    return lr


def train(epoch):
    current_lr = adjust_learning_rate(optimizer, epoch)
    train_loss = AverageMeter()
    id_loss = AverageMeter()
    tri_loss = AverageMeter()
    relation_id_loss = AverageMeter()
    relation_tri_loss = AverageMeter()
    twins_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    correct = 0
    total = 0

    global_loss_idx_loss = AverageMeter()
    local_loss_idx_loss = AverageMeter()

    # switch to train mode
    net.train()
    end = time.time()

    for batch_idx, (input1, input2, label1, label2) in enumerate(trainloader):

        labels = torch.cat((label1, label2), 0)

        input1 = Variable(input1.cuda())
        input2 = Variable(input2.cuda())

        labels = Variable(labels.cuda())
        data_time.update(time.time() - end)

        if args.pcb == 'on':
            feat, out0, feat_all, \
            global_feature_extract, local_feat_extract, \
            relation_final_feat_list, relation_logits_local_list, relation_final_feat_all = net(input1, input2)

            ## feat, out0, feat_all, local_feat_extract, global_feature_extract = net(input1, input2)

            # ### raltionship part loss
            loss_id_relation = criterion_id(relation_logits_local_list[0], labels.long())

            loss_tri_l_relation, batch_acc_relation = criterion_tri(relation_final_feat_list[0], labels)
            for i in range(len(relation_final_feat_list) - 1):
                loss_id_relation += criterion_id(relation_logits_local_list[i + 1], labels.long())
                loss_tri_l_relation += criterion_tri(relation_final_feat_list[i + 1], labels)[0]
            loss_tri_relation, batch_acc_relation = criterion_tri(relation_final_feat_all, labels)
            loss_tri_relation += loss_tri_l_relation * args.w_center

            ###
            loss_id = criterion_id(out0[0], labels.long())

            loss_tri_l, batch_acc = criterion_tri(feat[0], labels)
            for i in range(len(feat) - 1):
                loss_id += criterion_id(out0[i + 1], labels.long())
                loss_tri_l += criterion_tri(feat[i + 1], labels)[0]
            loss_tri, batch_acc = criterion_tri(feat_all, labels)
            loss_tri += loss_tri_l * args.w_center

            ### SSL  loss
            loss_twins = twinsloss(feat_all, labels)

            ### Aligned loss
            loss_global_loss_inx, p_inds, n_inds, _, _ = g_tri_loss(global_feature_extract, labels.long())
            loss_local_loss_inx, _, _ = l_tri_loss(local_feat_extract, p_inds, n_inds, labels)

            # total loss
            correct += batch_acc
            loss = loss_id + loss_tri + loss_twins + loss_local_loss_inx + loss_tri_relation  # loss_id_relation +#+ loss_global_loss_inx
            # loss = loss_id + loss_tri + loss_twins + loss_local_loss_inx + loss_tri_relation  # HC + SSL + shrot + relationship
            # loss = loss_id + loss_tri + loss_twins + loss_tri_relation # HC + SSL + relationship
            # loss = loss_id + loss_tri + loss_twins # HC + SSL

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update P
        train_loss.update(loss.item(), 2 * input1.size(0))
        id_loss.update(loss_id.item(), 2 * input1.size(0))
        tri_loss.update(loss_tri, 2 * input1.size(0))

        relation_id_loss.update(loss_id_relation, 2 * input1.size(0))
        relation_tri_loss.update(loss_tri_relation, 2 * input1.size(0))

        twins_loss.update(loss_twins.item(), 2 * input1.size(0))

        global_loss_idx_loss.update(loss_global_loss_inx.item(), 2 * input1.size(0))
        local_loss_idx_loss.update(loss_local_loss_inx.item(), 2 * input1.size(0))

        total += labels.size(0)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % 50 == 0:
            print('{} '
                  'Epoch: [{}][{}/{}] '
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'lr:{:.3f} '
                  'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '
                  'iLoss: {id_loss.val:.4f} ({id_loss.avg:.4f}) '
                  'TLoss: {tri_loss.val:.4f} ({tri_loss.avg:.4f}) '
                  'Accu: {:.2f}'.format(
                datetime.now(),
                epoch, batch_idx, len(trainloader), current_lr,
                100. * correct / total, batch_time=batch_time,
                train_loss=train_loss, id_loss=id_loss, tri_loss=tri_loss))

    writer.add_scalar('total_loss', train_loss.avg, epoch)
    writer.add_scalar('id_loss', id_loss.avg, epoch)
    writer.add_scalar('tri_loss', tri_loss.avg, epoch)
    writer.add_scalar('twins_loss', twins_loss.avg, epoch)
    writer.add_scalar('lr', current_lr, epoch)


import cv2


def plot(tensor, ims, label):
    b, c, h, w = ims.shape
    N, C, H, W = tensor.shape
    num = 0
    for n in range(N):
        num = num + 1
        im_name = label[n]
        im_name = im_name.split('.')
        for c in range(64):
            feature = tensor.cpu().detach().numpy()[n, c, :, :]

            # feature = 1.0 / (1 + np.exp(-1 * feature))
            # print(feature)
            feature = np.asarray(feature * 255, dtype=np.uint8)
            # print(feature)
            feature_new = cv2.resize(feature, (w, h))

            img = ims.cpu().detach().numpy()[n, :, :, :]
            img = 1.0 / (1 + np.exp(-1 * img))
            img = np.round(img * 255)
            img = np.transpose(img, (1, 2, 0))
            ret, mask = cv2.threshold(feature_new, 50, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)

            # img_reponse = cv2.cvtColor(feature_new, cv2.COLOR_GRAY2BGR)
            img_reponse = cv2.applyColorMap(feature_new, cv2.COLORMAP_JET)
            cv2.imwrite('./featureMapShow/' + im_name[0] + '_' + str(c) + 'M.jpg', img_reponse)

            img_fusion = np.round(img * 0.5 + img_reponse * 0.5)

            # img_fusion = img_fusion.numpy()
            # print(type(img_fusion))
            b, g, r = cv2.split(img_fusion)
            img_fusion = cv2.merge([r, g, b])
            img = np.asarray(img, dtype=np.uint8)
            b, g, r = cv2.split(img)
            img = cv2.merge([r, g, b])
            cv2.imwrite('./featureMapShow/' + im_name[0] + '_O.jpg', img)
            # img_fusion = cv2.bitwise_and(img,img,mask = mask)
            # print(img_fusion.shape)

            cv2.imwrite('./featureMapShow/' + im_name[0] + '_' + str(c) + '_F.jpg', img_fusion)



# Show result
def imshow(path, title=None):
    """Imshow for Tensor."""
    im = plt.imread(path)
    plt.imshow(im)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


#######################################################################
# sort the images
def sort_img(qf, ql, qc, gf, gl, gc):
    query = qf.view(-1, 1)
    # print(query.shape)
    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]
    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl == ql)
    # same camera
    camera_index = np.argwhere(gc == qc)

    # good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl == -1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1)

    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]
    return index


import scipy.io as sio
from PIL import Image
if __name__ == '__main__':

    # weights_dict = torch.load('save_model/sysu_conv1.t')
    # a = weights_dict['net']
    # net.load_state_dict(a)
    # net.eval()
    # print('Extracting Gallery Feature...')
    # start = time.time()
    # ptr = 0
    # if args.pcb == 'on':
    #     feat_dim = args.num_strips * args.local_feat_dim
    # else:
    #     feat_dim = 2048
    # gall_feat = np.zeros((ngall, feat_dim))
    # gall_feat_att = np.zeros((ngall, feat_dim))
    # with torch.no_grad():
    #     for batch_idx, (input, label) in enumerate(gall_loader):
    #         batch_num = input.size(0)
    #         input = Variable(input.cuda())
    #         if args.pcb == 'on':
    #             feat= net(input, input, test_mode[0])
    #
    #             gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
    #         else:
    #             feat, feat_att = net(input, input, test_mode[0])
    #             gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
    #             gall_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
    #         ptr = ptr + batch_num
    #     # io.savemat('features/gallery.mat', {'gall': gall_feat, 'labels': gall_label})
    # print('Extracting Time:\t {:.3f}'.format(time.time() - start))
    #
    # # switch to evaluation
    # net.eval()
    # print('Extracting Query Feature...')
    # start = time.time()
    # ptr = 0
    #
    # query_feat = np.zeros((nquery, feat_dim))
    # query_feat_att = np.zeros((nquery, feat_dim))
    # with torch.no_grad():
    #     for batch_idx, (input, label) in enumerate(query_loader):
    #         batch_num = input.size(0)
    #         input = Variable(input.cuda())
    #         if args.pcb == 'on':
    #             feat = net(input, input, test_mode[1])
    #             query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
    #         else:
    #             feat, feat_att = net(input, input, test_mode[1])
    #             query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
    #             query_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
    #         ptr = ptr + batch_num
    #     # io.savemat('features/query.mat', {'query': query_feat, 'labels': query_label})
    # print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    q_mat_path = 'features/our approach/query.mat'
    g_mat_path = 'features/our approach/gallery.mat'
    mat = sio.loadmat(q_mat_path)
    query_feat = mat["query"]  # 3803*3072
    query_label = mat["labels"].squeeze()

    mat = sio.loadmat(g_mat_path)
    gall_feat = mat["gall"]
    gall_label = mat["labels"].squeeze()

    query_index = 999

    i = query_index
    query_feat = torch.tensor(query_feat)
    gall_feat = torch.tensor(gall_feat)

    index = sort_img(query_feat[i], query_label[i], query_cam[i], gall_feat, gall_label, gall_cam)

    query_path = query_img[i]
    query_label = query_label[i]
    print(query_path)
    print('Top 10 images are as follow:')

    try:  # Visualize Ranking Result
        # Graphical User Interface is needed
        fig = plt.figure(figsize=(16, 4))
        ax = plt.subplot(1, 11, 1)      # 1行11列
        ax.axis('off')
        imshow(query_path, 'query')       # 第一张图片为query
        for i in range(10):
            ax = plt.subplot(1, 11, i + 2)
            ax.axis('off')
            img_path = gall_img[index[i]]
            label = gall_label[index[i]]
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
            img_path = gall_img.imgs[index[i]]
            print(img_path[0])
            # f = open(log_path, 'a')
            # f.write(img_path + '\n')
        print('If you want to see the visualization of the ranking result, graphical user interface is needed.')

    fig.savefig("show %d.png" % query_index)


    # net.load_state_dict(b)
    # net.eval()
    # feat, visualize_x = net(img, img, test_mode[0])  # (二维数据 test_mode[0]=1)
    # label_v = [str(i) for i in range(101)]
    # plot(visualize_x, img, label_v)

