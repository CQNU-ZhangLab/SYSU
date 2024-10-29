import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from resnet import resnet50, resnet18

import torch.utils.model_zoo as model_zoo
from utils import weights_init_kaiming
from utils import weights_init_classifier
from resnet import model_urls, conv3x3, remove_fc
from resnet import BasicBlock, Bottleneck


class SwitchBatchNorm2d(nn.Module):
    def __init__(self, num_features, switch, eps=1e-05, momentum=0.1, affine=True,
                 track_running_stats=True, with_relu=True):
        super(SwitchBatchNorm2d, self).__init__()
        self.thermal_bn = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
        self.visible_bn = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
        self.sharable_bn = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
        self.relu = nn.ReLU(inplace=True) if with_relu else None
        self.switch = switch  # 0 for split_x, 1 for share_x

    def forward(self, x, mode=0):
        if self.switch == 0:
            # split bn
            if mode == 0:
                x_v, x_t = torch.split(x, x.size(0) // 2, dim=0)
                x_v = self.visible_bn(x_v)
                x_t = self.thermal_bn(x_t)
                split_x = torch.cat((x_v, x_t), 0)  # [32,64,128,64]*2
            elif mode == 1:
                split_x = self.visible_bn(x)
            elif mode == 2:
                split_x = self.thermal_bn(x)
            split_x = self.relu(split_x) if self.relu is not None else split_x
            return split_x
        elif self.switch == 1:
            # share bn
            share_x = self.sharable_bn(x)
            share_x = self.relu(share_x) if self.relu is not None else share_x
            return share_x
        else:
            raise ValueError('Invalid switch value: {}, must be 0 or 1.'.format(self.switch))


class BasicBlockSwitchBN(nn.Module):
    expansion = 1

    def __init__(self, config, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockSwitchBN, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = SwitchBatchNorm2d(planes, config.pop(0), with_relu=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = SwitchBatchNorm2d(planes, config.pop(0), with_relu=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, mode=0):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out, mode)

        out = self.conv2(out)
        out = self.bn2(out, mode)

        if self.downsample is not None:
            residual = self.downsample(x, mode)

        out += residual
        out = self.relu(out)

        return out


class BottleneckSwitchBN(nn.Module):
    expansion = 4

    def __init__(self, config, inplanes, planes, stride=1, downsample=None):
        super(BottleneckSwitchBN, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = SwitchBatchNorm2d(planes, config.pop(0), with_relu=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = SwitchBatchNorm2d(planes, config.pop(0), with_relu=True)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = SwitchBatchNorm2d(planes * 4, config.pop(0), with_relu=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, mode=0):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out, mode)

        out = self.conv2(out)
        out = self.bn2(out, mode)

        out = self.conv3(out)
        out = self.bn3(out, mode)

        if self.downsample is not None:
            residual = self.downsample(x, mode)

        out += residual
        out = self.relu(out)

        return out


class DownsampleSwitchBN(nn.Module):
    def __init__(self, config, in_channels, out_channels, kernel_size, stride, bias):
        super(DownsampleSwitchBN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride, bias=bias)
        self.bn = SwitchBatchNorm2d(out_channels, config.pop(0), with_relu=False)

    def forward(self, x, mode=0):
        x = self.conv(x)
        x = self.bn(x, mode)

        return x


class ResNetUnfoldSwitchBN(nn.Module):
    def __init__(self, block, layers, config=None, last_stride=1):
        assert config is not None
        self.inplanes = 64
        super(ResNetUnfoldSwitchBN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = SwitchBatchNorm2d(64, config.pop(0), with_relu=True)
        # self.relu = nn.ReLU(inplace=True)   # add missed relu
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(config, block, 64, layers[0])
        self.layer2 = self._make_layer(config, block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(config, block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(config, block, 512, layers[3], stride=last_stride)


    def _make_layer(self, config, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DownsampleSwitchBN(config, self.inplanes, planes * block.expansion,
                                            kernel_size=1, stride=stride, bias=False)

        layers = nn.ModuleList()
        layers.append(block(config, self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(config, self.inplanes, planes))

        return layers

    def forward(self, x, mode=0):
        x = self.conv1(x)  # [64,3,256,128]====>[64,64,128,64]
        x = self.bn1(x, mode)
        # global bn_x
        # bn_x = x
        # x = self.relu(x)    # add missed relu
        x = self.maxpool(x)  #=====>[64,64,64,32]

        for layer in self.layer1:
            x = layer(x, mode) # 1==>[64,256,64,32]
        for layer in self.layer2:
            x = layer(x, mode)  #1==>[64,512,32,16]
        for layer in self.layer3:
            x = layer(x, mode)  #1==>[64,1024,16,8]
        for layer in self.layer4:
            x = layer(x, mode)  #1==>[64,2048,16,8]


        return x

    def load_param(self, pretrained_weights):
        pretrained_state_dict = remove_fc(torch.load(pretrained_weights))
        self.load_state_dict(pretrained_state_dict)

    def load_state_dict(self, pretrained_state_dict):
        for key in pretrained_state_dict:
            if 'bn' in key:
                key_items = key.split('.')
                model_key = '.'.join(key_items[:-1]) + '.thermal_bn.' + key_items[-1]
                self.state_dict()[model_key].copy_(pretrained_state_dict[key])
                model_key = '.'.join(key_items[:-1]) + '.visible_bn.' + key_items[-1]
                self.state_dict()[model_key].copy_(pretrained_state_dict[key])
                model_key = '.'.join(key_items[:-1]) + '.sharable_bn.' + key_items[-1]
                self.state_dict()[model_key].copy_(pretrained_state_dict[key])
            elif 'downsample.0' in key:
                key_items = key.split('.')
                model_key = '.'.join(key_items[:-2]) + '.conv.' + key_items[-1]
                self.state_dict()[model_key].copy_(pretrained_state_dict[key])
            elif 'downsample.1' in key:
                key_items = key.split('.')
                model_key = '.'.join(key_items[:-2]) + '.bn.thermal_bn.' + key_items[-1]
                self.state_dict()[model_key].copy_(pretrained_state_dict[key])
                model_key = '.'.join(key_items[:-2]) + '.bn.visible_bn.' + key_items[-1]
                self.state_dict()[model_key].copy_(pretrained_state_dict[key])
                model_key = '.'.join(key_items[:-2]) + '.bn.sharable_bn.' + key_items[-1]
                self.state_dict()[model_key].copy_(pretrained_state_dict[key])
            else:
                self.state_dict()[key].copy_(pretrained_state_dict[key])


def resnet18unfold_switchbn(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetUnfoldSwitchBN(BasicBlockSwitchBN, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(remove_fc(model_zoo.load_url(model_urls['resnet18'])))
    return model


def resnet34unfold_switchbn(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetUnfoldSwitchBN(BasicBlockSwitchBN, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(remove_fc(model_zoo.load_url(model_urls['resnet34'])))
    return model


def resnet50unfold_switchbn(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetUnfoldSwitchBN(BottleneckSwitchBN, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(remove_fc(model_zoo.load_url(model_urls['resnet50'])))
    return model


def resnet101unfold_switchbn(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetUnfoldSwitchBN(BottleneckSwitchBN, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(remove_fc(model_zoo.load_url(model_urls['resnet101'])))
    return model


def resnet152unfold_switchbn(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetUnfoldSwitchBN(BottleneckSwitchBN, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(remove_fc(model_zoo.load_url(model_urls['resnet152'])))
    return model


"""""""""""""""--------- 以上是第三者---------"""""""""""""""""""""""


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class Non_local(nn.Module):
    def __init__(self, in_channels, reduc_ratio=2):
        super(Non_local, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = reduc_ratio // reduc_ratio

        self.g = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                      padding=0),
        )

        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels),
        )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)

        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        '''
                :param x: (b, c, t, h, w)
                :return:
                '''

        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        # f_div_C = torch.nn.functional.softmax(f, dim=-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


# #####################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)


class visible_module(nn.Module):
    def __init__(self, arch='resnet50', share_net=1):
        super(visible_module, self).__init__()

        model_v = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.share_net = share_net  # 1

        if self.share_net == 0:
            pass
        else:
            self.visible = nn.ModuleList()
            self.visible.conv1 = model_v.conv1  # nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)
            self.visible.bn1 = model_v.bn1
            self.visible.relu = model_v.relu
            self.visible.maxpool = model_v.maxpool
            if self.share_net > 1:
                for i in range(1, self.share_net):
                    setattr(self.visible, 'layer' + str(i), getattr(model_v, 'layer' + str(i)))

    def forward(self, x):
        if self.share_net == 0:
            return x
        else:
            x = self.visible.conv1(x)
            x = self.visible.bn1(x)
            x = self.visible.relu(x)
            x = self.visible.maxpool(x)

            if self.share_net > 1:
                for i in range(1, self.share_net):
                    x = getattr(self.visible, 'layer' + str(i))(x)
            return x


class thermal_module(nn.Module):
    def __init__(self, arch='resnet50', share_net=1):
        super(thermal_module, self).__init__()

        model_t = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.share_net = share_net

        if self.share_net == 0:
            pass
        else:
            self.thermal = nn.ModuleList()
            self.thermal.conv1 = model_t.conv1
            self.thermal.bn1 = model_t.bn1
            self.thermal.relu = model_t.relu
            self.thermal.maxpool = model_t.maxpool
            if self.share_net > 1:
                for i in range(1, self.share_net):
                    setattr(self.thermal, 'layer' + str(i), getattr(model_t, 'layer' + str(i)))

    def forward(self, x):
        if self.share_net == 0:
            return x
        else:
            x = self.thermal.conv1(x)
            x = self.thermal.bn1(x)
            x = self.thermal.relu(x)
            x = self.thermal.maxpool(x)

            if self.share_net > 1:
                for i in range(1, self.share_net):
                    x = getattr(self.thermal, 'layer' + str(i))(x)
            return x


class base_resnet(nn.Module):
    def __init__(self, arch='resnet50', share_net=1):
        super(base_resnet, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.share_net = share_net
        if self.share_net == 0:
            self.base = model_base
        else:
            self.base = nn.ModuleList()

            if self.share_net > 4:
                pass
            else:
                for i in range(self.share_net, 5):
                    setattr(self.base, 'layer' + str(i), getattr(model_base, 'layer' + str(i)))

    def forward(self, x):
        if self.share_net == 0:
            x = self.base.conv1(x)
            x = self.base.bn1(x)
            x = self.base.relu(x)
            x = self.base.maxpool(x)

            x = self.base.layer1(x)
            x = self.base.layer2(x)
            x = self.base.layer3(x)
            x = self.base.layer4(x)
            return x
        elif self.share_net > 4:
            return x
        else:
            for i in range(self.share_net, 5):
                x = getattr(self.base, 'layer' + str(i))(x)
            return x


class embed_net(nn.Module):
    def __init__(self, class_num, no_local='off', gm_pool='on', arch='resnet50', share_net=1, pcb='on',
                 local_feat_dim=256, num_strips=4):
        super(embed_net, self).__init__()
        # self.conv = nn.Conv2d(2048, 2048, (3, 2), (1, 1), (2, 1))  # 硬凑的

        self.thermal_module = thermal_module(arch=arch, share_net=share_net)
        self.visible_module = visible_module(arch=arch, share_net=share_net)
        config = open('config/conv1.cfg').readline()
        config = [int(x) for x in config.strip().split(' ')]
        self.base_resnet = resnet50unfold_switchbn(config=config, pretrained=True, last_stride=1)

        # weights_dict = torch.load('save_model/sysu_conv2.t', map_location='cuda:0')
        # for k in list(weights_dict.keys()):
        #     if "cmc" or "mAP" or "mINP" or 'epoch' in k:
        #         del weights_dict[k]
        #     '''
        #     if "mAP" in a:
        #         del weights_dict[k]
        #     if "cmc" in k:
        #         del weights_dict[k]
        #     '''
        # # self.base_resnet.load_state_dict(weights_dict, strict=False)
        # self.base_resnet.load_state_dict(weights_dict)
        #
        # print("加载成功！")



        self.non_local = no_local
        self.pcb = pcb
        if self.non_local == 'on':
            pass

        ### Alingn
        planes = 2048
        # local_feat_dim = 256
        local_conv_out_channels = local_feat_dim
        self.local_conv = nn.Conv2d(planes, local_conv_out_channels, 1)
        self.local_bn = nn.BatchNorm2d(local_conv_out_channels)
        self.local_relu = nn.ReLU(inplace=True)

        self.fc = nn.Linear(planes, class_num)
        init.normal(self.fc.weight, std=0.001)
        init.constant(self.fc.bias, 0)

        pool_dim = 2048
        self.l2norm = Normalize(2)
        self.gm_pool = gm_pool

        if self.pcb == 'on':  # true
            self.num_stripes = num_strips  # 6
            local_conv_out_channels = local_feat_dim  # 256

            self.local_conv_list = nn.ModuleList()  # 一个容器包含conv和其他操作
            for _ in range(self.num_stripes):
                conv = nn.Conv2d(pool_dim, local_conv_out_channels, 1)
                conv.apply(weights_init_kaiming)  # 将fn函数递归地应用到网络模型的每个子模型中，主要用在参数的初始化。
                self.local_conv_list.append(nn.Sequential(
                    conv,
                    nn.BatchNorm2d(local_conv_out_channels),
                    nn.ReLU(inplace=True)
                ))

            self.fc_list = nn.ModuleList()
            for _ in range(self.num_stripes):
                fc = nn.Linear(local_conv_out_channels, class_num)
                init.normal_(fc.weight, std=0.001)
                init.constant_(fc.bias, 0)
                self.fc_list.append(fc)


        else:
            self.bottleneck = nn.BatchNorm1d(pool_dim)
            self.bottleneck.bias.requires_grad_(False)  # no shift

            self.classifier = nn.Linear(pool_dim, class_num, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.num_stripes = num_strips
        self.num_classes = class_num

        self.local_6_conv_list = nn.ModuleList()

        self.rest_6_conv_list = nn.ModuleList()

        self.relation_6_conv_list = nn.ModuleList()

        self.global_6_max_conv_list = nn.ModuleList()

        self.global_6_rest_conv_list = nn.ModuleList()

        self.global_6_pooling_conv_list = nn.ModuleList()

        for i in range(self.num_stripes):
            self.local_6_conv_list.append(nn.Sequential(
                nn.Conv2d(2048, local_conv_out_channels, 1),
                nn.BatchNorm2d(local_conv_out_channels),
                nn.ReLU(inplace=True)))

        for i in range(self.num_stripes):
            self.rest_6_conv_list.append(nn.Sequential(
                nn.Conv2d(2048, local_conv_out_channels, 1),
                nn.BatchNorm2d(local_conv_out_channels),
                nn.ReLU(inplace=True)))

        self.global_6_max_conv_list.append(nn.Sequential(
            nn.Conv2d(2048, local_conv_out_channels, 1),
            nn.BatchNorm2d(local_conv_out_channels),
            nn.ReLU(inplace=True)))

        self.global_6_rest_conv_list.append(nn.Sequential(
            nn.Conv2d(2048, local_conv_out_channels, 1),
            nn.BatchNorm2d(local_conv_out_channels),
            nn.ReLU(inplace=True)))

        for i in range(self.num_stripes):
            self.relation_6_conv_list.append(nn.Sequential(
                nn.Conv2d(local_conv_out_channels * 2, local_conv_out_channels, 1),
                nn.BatchNorm2d(local_conv_out_channels),
                nn.ReLU(inplace=True)))

        self.global_6_pooling_conv_list.append(nn.Sequential(
            nn.Conv2d(local_conv_out_channels * 2, local_conv_out_channels, 1),
            nn.BatchNorm2d(local_conv_out_channels),
            nn.ReLU(inplace=True)))

        if self.num_classes > 0:
            self.fc_local_6_list = nn.ModuleList()
            for _ in range(self.num_stripes):
                fc = nn.Linear(local_conv_out_channels, self.num_classes)
                init.normal_(fc.weight, std=0.001)
                init.constant_(fc.bias, 0)
                self.fc_local_6_list.append(fc)

            self.fc_rest_6_list = nn.ModuleList()
            for _ in range(self.num_stripes):
                fc = nn.Linear(local_conv_out_channels, self.num_classes)
                init.normal_(fc.weight, std=0.001)
                init.constant_(fc.bias, 0)
                self.fc_rest_6_list.append(fc)

            self.fc_local_rest_6_list = nn.ModuleList()
            for _ in range(self.num_stripes):
                fc = nn.Linear(local_conv_out_channels, self.num_classes)
                init.normal_(fc.weight, std=0.001)
                init.constant_(fc.bias, 0)
                self.fc_local_rest_6_list.append(fc)

            self.fc_global_6_list = nn.ModuleList()
            fc = nn.Linear(local_conv_out_channels, self.num_classes)
            init.normal_(fc.weight, std=0.001)
            init.constant_(fc.bias, 0)
            self.fc_global_6_list.append(fc)

            self.fc_global_max_6_list = nn.ModuleList()
            fc = nn.Linear(local_conv_out_channels, self.num_classes)
            init.normal_(fc.weight, std=0.001)
            init.constant_(fc.bias, 0)
            self.fc_global_max_6_list.append(fc)

            self.fc_global_rest_6_list = nn.ModuleList()
            fc = nn.Linear(local_conv_out_channels, self.num_classes)
            init.normal_(fc.weight, std=0.001)
            init.constant_(fc.bias, 0)
            self.fc_global_rest_6_list.append(fc)

    def forward(self, x1, x2, modal=0):
        if modal == 0:
            # x1 = self.visible_module(x1)
            # x2 = self.thermal_module(x2)
            x = torch.cat((x1, x2), 0)
        elif modal == 1:
            x = x1 # 这里是可见光的特征
        elif modal == 2:
            x = x2  # 这里是红外光的特征

        # shared block

        x = self.base_resnet(x,modal)  # modal=0: [64,256,72,36]===>[64,2048,18,9]  ''''''[64,256,64,32]===>[64,2048,16,8]
        visualize_x = x
        # x = self.conv(x)

        ## relationship
        criterion = nn.CrossEntropyLoss()
        feat = x


        # feat = self.base(x)
        assert (feat.size(2) % self.num_stripes == 0)
        stripe_h_6 = int(feat.size(2) / self.num_stripes)

        local_6_feat_list = []

        final_feat_list = []
        logits_list = []
        rest_6_feat_list = []

        logits_local_rest_list = []
        logits_local_list = []
        logits_rest_list = []
        logits_global_list = []

        for i in range(self.num_stripes):  # 6
            # local_6_feat = F.max_pool2d(
            #     feat[:, :, i * stripe_h_6: (i + 1) * stripe_h_6, :],
            #     (stripe_h_6, feat.size(-1)))

            local_6_feat = feat[:, :, i * stripe_h_6: (i + 1) * stripe_h_6, :]
            b, c, h, w = local_6_feat.shape
            local_6_feat = local_6_feat.view(b, c, -1)
            p = 10  # regDB: 10.0    SYSU: 3.0
            local_6_feat = (torch.mean(local_6_feat ** p, dim=-1) + 1e-12) ** (1 / p)
            local_6_feat = local_6_feat.view(local_6_feat.size(0), local_6_feat.size(1), 1, 1)

            local_6_feat_list.append(local_6_feat)

        for i in range(self.num_stripes):
            rest_6_feat_list.append((local_6_feat_list[(i + 1) % self.num_stripes]
                                     + local_6_feat_list[(i + 2) % self.num_stripes]
                                     + local_6_feat_list[(i + 3) % self.num_stripes]
                                     + local_6_feat_list[(i + 4) % self.num_stripes]
                                     + local_6_feat_list[(i + 5) % self.num_stripes]) / 5)

        for i in range(self.num_stripes):

            local_6_feat = self.local_6_conv_list[i](local_6_feat_list[i]).squeeze(3).squeeze(2)
            input_rest_6_feat = self.rest_6_conv_list[i](rest_6_feat_list[i]).squeeze(3).squeeze(2)

            input_local_rest_6_feat = torch.cat((local_6_feat, input_rest_6_feat), 1).unsqueeze(2).unsqueeze(3)

            local_rest_6_feat = self.relation_6_conv_list[i](input_local_rest_6_feat)

            local_rest_6_feat = (local_rest_6_feat
                                 + local_6_feat.unsqueeze(2).unsqueeze(3)).squeeze(3).squeeze(2)

            final_feat_list.append(local_rest_6_feat)

            if self.num_classes > 0:
                logits_local_rest_list.append(self.fc_local_rest_6_list[i](local_rest_6_feat))
                logits_local_list.append(self.fc_local_6_list[i](local_6_feat))
                logits_rest_list.append(self.fc_rest_6_list[i](input_rest_6_feat))

        final_feat_all = [lf for lf in final_feat_list]
        final_feat_all = torch.cat(final_feat_all, dim=1)
        # if self.num_classes > 0:
        #     logits_global_list.append(self.fc_global_6_list[0](global_6_feat))
        #     logits_global_list.append(self.fc_global_max_6_list[0](global_6_max_feat.squeeze(3).squeeze(2)))
        #     logits_global_list.append(self.fc_global_rest_6_list[0](global_6_rest_feat.squeeze(3).squeeze(2)))

        # return final_feat_list, logits_local_rest_list, logits_local_list, logits_rest_list, logits_global_list

        # return final_feat_list


        x_pool = F.adaptive_max_pool2d(x, 1)
        x_pool = x_pool.view(x_pool.size(0), x_pool.size(1))  # global_feature.
        global_feature_short = x_pool

        local_feat_x = torch.mean(x, -1, keepdim=True)
        local_feat_conv = self.local_conv(local_feat_x)
        local_feat_bn = self.local_bn(local_feat_conv)
        local_feat_relu = self.local_relu(local_feat_bn)
        local_feat_sq = local_feat_relu.squeeze(-1).permute(0, 2, 1)
        local_feat_extract = local_feat_sq

        if self.pcb == 'on':
            feat_1 = x
            assert feat_1.size(2) % self.num_stripes == 0
            stripe_h = int(feat_1.size(2) / self.num_stripes)
            # print('分块为：',stripe_h)
            local_feat_list = []
            logits_list = []

            for i in range(self.num_stripes):
                # shape [N, C, 1, 1]
                # average pool
                # local_feat = F.avg_pool2d(feat[:, :, i * stripe_h: (i + 1) * stripe_h, :],(stripe_h, feat.size(-1)))
                if self.gm_pool == 'on':
                    # gm pool
                    local_feat = feat_1[:, :, i * stripe_h: (i + 1) * stripe_h, :]
                    b, c, h, w = local_feat.shape
                    local_feat = local_feat.view(b, c, -1)
                    p = 10  # regDB: 10.0    SYSU: 3.0
                    local_feat = (torch.mean(local_feat ** p, dim=-1) + 1e-12) ** (1 / p)
                else:
                    # average pool
                    # local_feat = F.avg_pool2d(feat[:, :, i * stripe_h: (i + 1) * stripe_h, :],(stripe_h, feat.size(-1)))
                    local_feat = F.max_pool2d(feat[:, :, i * stripe_h: (i + 1) * stripe_h, :],
                                              (stripe_h, feat.size(-1)))

                # shape [N, c, 1, 1]
                local_feat = self.local_conv_list[i](local_feat.view(feat_1.size(0), feat.size(1), 1, 1))

                # shape [N, c]
                local_feat = local_feat.view(local_feat.size(0), -1)
                local_feat_list.append(local_feat)

                if hasattr(self, 'fc_list'):
                    logits_list.append(self.fc_list[i](local_feat))

            feat_all = [lf for lf in local_feat_list]
            # feat_all_align = feat_all
            feat_all = torch.cat(feat_all, dim=1)

            #####  align 
            # 
            # feat_all_add_dim = torch.stack(feat_all_align, dim=2)
            # 
            # feat_all_add_dim_local = feat_all_add_dim.permute(0, 2, 1)

            if self.training:
                return local_feat_list, logits_list, feat_all, feat_all, local_feat_extract, final_feat_list, logits_local_list, final_feat_all
                # return final_feat_list, logits_local_list, feat_all, feat_all, local_feat_extract
            else:
                return self.l2norm(feat_all), visualize_x #, bn_x


if __name__ == "__main__":
    # modal=0: [64,256,72,36]===>[64,2048,18,9]  ''''''[64,256,64,32]===>[64,2048,16,8]


    x1 = torch.randn(2048, 16, 8)
    conv = nn.Conv2d(2048,2048,(1,2),(1,1),(1,1))
    x2 = conv(x1)
   # model = embed_net(9)
    # y1, y2, y3, y4, y5, y6, y7, y8 = model(x1, x2)
    print(x2.shape)
