# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn
import torch.nn.functional as F

from .backbones.resnet import ResNet, BasicBlock, Bottleneck
from .backbones.senet import SENet, SEResNetBottleneck, SEBottleneck, SEResNeXtBottleneck


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Baseline(nn.Module):
    #in_planes = 16384
    #in_planes = 14336
    in_planes = 2048
    #in_planes = 4096

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice):
        super(Baseline, self).__init__()
        if model_name == 'resnet18':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride, 
                               block=BasicBlock, 
                               layers=[2, 2, 2, 2])
        elif model_name == 'resnet34':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet50':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet101':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck, 
                               layers=[3, 4, 23, 3])
        elif model_name == 'resnet152':
            self.base = ResNet(last_stride=last_stride, 
                               block=Bottleneck,
                               layers=[3, 8, 36, 3])
            
        elif model_name == 'se_resnet50':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 4, 6, 3], 
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride) 
        elif model_name == 'se_resnet101':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 4, 23, 3], 
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnet152':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 8, 36, 3],
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)  
        elif model_name == 'se_resnext50':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 6, 3], 
                              groups=32, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride) 
        elif model_name == 'se_resnext101':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 23, 3], 
                              groups=32, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'senet154':
            self.base = SENet(block=SEBottleneck, 
                              layers=[3, 8, 36, 3],
                              groups=64, 
                              reduction=16,
                              dropout_p=0.2, 
                              last_stride=last_stride)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')

        #self.gap = nn.MaxPool2d((1,4), stride=(1,1))
        #self.gap = nn.AvgPool2d((1,4), stride=(1,1))
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat

        if self.neck == 'no':
            self.classifier = nn.Linear(self.in_planes, self.num_classes)
            # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)     # new add by luo
            # self.classifier.apply(weights_init_classifier)  # new add by luo
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.in_planes * 2)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)

        featdim = 128
        self.fc1 = nn.Linear(4096, featdim, bias=False)
        self.fc2 = nn.Linear(1024, featdim, bias=False)
        self.bottleneck_feat = nn.BatchNorm1d(featdim)
        self.bottleneck_feat.bias.requires_grad_(False)  # no shift
        self.bottleneck_feat.apply(weights_init_kaiming)
        self.dp = nn.Dropout(p=0.5)

    def forward(self, x):
        low_feat1, low_feat2, low_feat3, conv_feat = self.base(x)
        n,c,h,w = conv_feat.shape
#        low_feat1 = F.normalize(self.gap(low_feat1).view(n, -1))
#        low_feat2 = F.normalize(self.gap(low_feat2).view(n, -1))
#        low_feat3 = F.normalize(self.gap(low_feat3).view(n, -1))
        gap_feat1 = self.gap(conv_feat).view(n, -1)
        gmp_feat1 = self.gmp(conv_feat).view(n, -1)
        final_feat1 = torch.cat((gap_feat1, gmp_feat1), 1)
#        final_feat1 = self.gap(conv_feat[:,:,2:14,2:6]).view(n, -1)
#        final_feat1 = conv_feat[:,:,int(h/2),int(w/2)].view(n, -1)
        final_feat1 = self.bottleneck(final_feat1)
#        final_feat2 = self.gmp(conv_feat[:,int(c/2):,:,:]).view(n, -1)
#        final_feat2 = self.gap(conv_feat).view(n, -1)
#        print(low_feat.shape, final_feat.shape)
#        print("feat shape:",(n,c,h,w))
#        half_feat1 = conv_feat[:,:int(c/2),:,:]
#        half_feat2 = conv_feat[:,int(c/2):,:,:]
#        global_feat_max = self.gmp(half_feat1).view(n, -1)  # (b, 2048, 1, 1)
#        global_feat_avg = self.gap(half_feat2).view(n, -1)  # (b, 2048, 1, 1)
#        global_feat_max = self.fc1(self.gmp(half_feat1).view(n, -1))  # (b, 2048, 1, 1)
#        global_feat_avg = self.fc2(self.gap(half_feat2).view(n, -1))  # (b, 2048, 1, 1)
         
#        global_feat = final_feat
#        low_feat = torch.cat((low_feat1, low_feat2, low_feat3), 1)
#        low_feat = low_feat2
#        print("low feat len:{} final feat len:{} all len:{}".format(low_feat.shape, final_feat.shape, global_feat.shape))
#        print("cat feat shape:",global_feat.shape)
#        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
#        feat =   # normalize for angular softmax
#        feat = torch.cat((final_feat1, final_feat2), 1)
        final_feat1 = self.dp(final_feat1)
        final_feat1 = self.fc1(final_feat1)
#        feat2 = self.fc2(final_feat2)

        feat_sm1 = self.bottleneck_feat(final_feat1)
        if self.training:
#            score = self.classifier(feat_sm1)
            return final_feat1, feat_sm1
        else:
            return feat_sm1
#        feat_sm2 = self.bottleneck_feat(feat2)

        

    def forward_old(self, x):

        global_feat = self.gap(self.base(x))  # (b, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)  # normalize for angular softmax

        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, global_feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            # if 'classifier' in i:
            #     continue
            self.state_dict()[i].copy_(param_dict[i])
