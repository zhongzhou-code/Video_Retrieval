import math

import torch
import torch.nn.functional as F
import torch.nn as nn
from functools import partial
from torch.autograd import Variable


def conv3x3x3(in_planes, out_planes, stride=1):
    """
        Constructs a conv3*3*3 function.
    """
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


def conv1x1x1(in_planes, out_planes, stride=1):
    """
        Constructs a conv1*1*1 function.
    """
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True)


def downsample_basic_block(x, planes, stride):
    """
        Constructs a downsample_basic_block function.
    """
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)

    zero_pads = torch.Tensor(out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4)).zero_()

    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    """
    Constructs a BasicBlock class model.
    """
    expansion = 1
    """

    """

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        """
        :param inplanes:
        :param planes:
        :param stride:
        :param downsample:
        """
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """
        :param x:
        :return:
        """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
        Constructs a Bottleneck class model.
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    Constructs a ResNet model.
    """
    """
    block:  Which block?  BasicBlock or Bottleneck
    layers: layers=[2,2,2,2] or layers=[3, 4, 6, 3] or layers=[3, 4, 23, 3], represent the number of blocks in each layer
    shortcut_type='A' : 
    """

    def __init__(self, block, layers, shortcut_type='A'):  # A   layers=[2,2,2,2]
        #
        self.inplanes = 64
        #
        super(ResNet, self).__init__()
        # The first convolution layer, input = 3, output = 64, kernel_size = 7, stride = (1, 2, 2), padding = (3, 3, 3)
        self.conv1 = nn.Conv3d(3, self.inplanes, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias=True)
        # The first convolution layer,
        self.bn1 = nn.BatchNorm3d(self.inplanes)
        # The first activation layer, inplace=True
        self.relu = nn.ReLU(inplace=True)
        # The first max pool layer, kernel_size=(3, 3, 3), stride=2, padding=1
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        # The first block layer,
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        # The second block layer, block = BasicBlock or Bottleneck, input = 128, shortcut_type='A', stride = 2
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        # The third block layer,
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2)
        # The forth block layer,
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=2)
        # The last average pool layer,
        self.avgpool = nn.AvgPool3d((1, 4, 4), stride=1)

        # self.modules() 采用深度优先遍历的方式，存储了net的所有模块
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    """
    Parameters:
        block:  Which block?  BasicBlock or Bottleneck
        planes: input_channel == 64 or 128 or 256 or 512
        blocks: the number of blocks in each layer
        shortcut_type: 'A'
        stride=1: default stride = (1, 1, 1) or stride = (2, 2, 2)
    """

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        # (stride != 1) represent need to perform the down sampling operation
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                # through function downsample_basic_block() perform the down sampling operation
                downsample = partial(downsample_basic_block, planes=planes * block.expansion, stride=stride)
            else:
                # through Conv3d() and BatchNorm3d() perform the down sampling operation
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm3d(planes * block.expansion)
                )

        # Initialization a list : layers = []
        layers = []
        # layers.append:
        # print(self.inplanes, planes, stride, downsample)
        layers.append(block(inplanes=self.inplanes, planes=planes, stride=stride, downsample=downsample))
        # when performed the down sampling operation, change self.inplanes = planes * block.expansion
        self.inplanes = planes * block.expansion
        # Build blocks, the number of (1 ~ blocks), append to list of layers
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        # Elements in the list implement the Sequential operation
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # print(x.size())     # 32 * 64 * 8 * 28 * 28
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # print(x.size())     # 32 * 512 * 1 * 4 * 4

        x = self.avgpool(x)
        # print(x.size())     # 32 * 512 * 1 * 1 * 1

        # out = x.view(x.size(0), -1)
        # out = torch.squeeze(x)
        out = x.squeeze(-1).squeeze(-1)
        # print(out.size())   # 32 * 512 * 1

        return out


def resnet18(**kwargs):
    """
    Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """
    Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """
    Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """
    Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


class Res3D_Hash_Model(nn.Module):
    """
    Constructs a ( C3D + Max Pooling + Hashing ) model.
    """

    def __init__(self, model_depth, hash_length, class_num,pretrained=False):
        """
        :param model_depth:
        :param hash_length:
        """
        super(Res3D_Hash_Model, self).__init__()

        assert model_depth in [18, 34, 50, 101, 152, 200]

        if model_depth == 18:
            self.resnet = resnet18()
        elif model_depth == 34:
            self.resnet = resnet34()
        elif model_depth == 50:
            self.resnet = resnet50()
        elif model_depth == 101:
            self.resnet = resnet101()
        #
        # load_state(self.resnet, "./pretrain/resnet-18-kinetics.pth")

        self.avgpooling = TemporalAvgPool()

        self.hash_layer = HashLayer(hash_length)

    def forward(self, x):
        resnet_feature = self.resnet(x)  # resnet_feature.shape: [batch_size, 512, 1]

        avgpooling_feature = self.avgpooling(resnet_feature)  # [batch_size, 512]
        # print(avgpooling_feature.shape)

        hash_feature = self.hash_layer(avgpooling_feature)  # [batch_size, hash_length]

        # print(hash_feature.shape)
        return hash_feature  # hash_feature


class TemporalAvgPool(nn.Module):
    def __init__(self):
        super(TemporalAvgPool, self).__init__()
        self.filter = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        out = self.filter(x)
        out = torch.squeeze(out)
        return out


class HashLayer(nn.Module):
    """
    Constructs a ( HashLayer ) model.
    """

    def __init__(self, hash_length):
        super(HashLayer, self).__init__()
        self.hash_coder = nn.Sequential(nn.Linear(512, hash_length), nn.Tanh())

    def forward(self, x):
        if x.size() == 5:
            x = x.view()
        h = self.hash_coder(x)
        # print(h.size()) #
        return h


def load_state(model, model_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location="cpu")["state_dict"]
    key = list(pretrained_dict.keys())[0]
    # 1. filter out unnecessary keys
    # 1.1 multi-GPU ->CPU
    if (str(key).startswith("module.")):
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if
                           k[7:] in model_dict and v.size() == model_dict[k[7:]].size()}
    else:
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                           k in model_dict and v.size() == model_dict[k].size()}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
