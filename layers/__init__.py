# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F

from .triplet_loss import TripletLoss, CrossEntropyLabelSmooth
from .cluster_loss import ClusterLoss
from .center_loss import CenterLoss
from .range_loss import RangeLoss
from .norm_margin import *


def make_loss(cfg, num_classes):    # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
        triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
    elif cfg.MODEL.METRIC_LOSS_TYPE == 'cluster':
        cluster = ClusterLoss(cfg.SOLVER.CLUSTER_MARGIN, True, True, cfg.SOLVER.IMS_PER_BATCH // cfg.DATALOADER.NUM_INSTANCE, cfg.DATALOADER.NUM_INSTANCE)
    elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_cluster':
        triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
        cluster = ClusterLoss(cfg.SOLVER.CLUSTER_MARGIN, True, True, cfg.SOLVER.IMS_PER_BATCH // cfg.DATALOADER.NUM_INSTANCE, cfg.DATALOADER.NUM_INSTANCE)
    else:
        print('expected METRIC_LOSS_TYPE should be triplet, cluster, triplet_cluster'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)     # new add by luo
        print("label smooth on, numclasses:", num_classes)

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)
    elif cfg.DATALOADER.SAMPLER == 'triplet':
        def loss_func(score, feat, target):
            return triplet(feat, target)[0]
    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        def loss_func(score, feat, target):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    return xent(score, target) + triplet(feat, target)[0]  # new add by luo, open label smooth
                else:
                    return F.cross_entropy(score, target) + triplet(feat, target)[0]    # new add by luo, no label smooth

            elif cfg.MODEL.METRIC_LOSS_TYPE == 'cluster':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    return xent(score, target) + cluster(feat, target)[0]  # new add by luo, open label smooth
                else:
                    return F.cross_entropy(score, target) + cluster(feat, target)[0]    # new add by luo, no label smooth

            elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_cluster':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    return xent(score, target) + triplet(feat, target)[0] + cluster(feat, target)[0]  # new add by luo, open label smooth
                else:
                    return F.cross_entropy(score, target) + triplet(feat, target)[0] + cluster(feat, target)[0]    # new add by luo, no label smooth
            else:
                print('expected METRIC_LOSS_TYPE should be triplet, cluster, triplet_cluster，'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))
    else:
        print('expected sampler should be softmax, triplet or softmax_triplet, '
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func


def make_loss_with_center(cfg, num_classes):    # modified by gu
    if cfg.MODEL.NAME == 'resnet18' or cfg.MODEL.NAME == 'resnet34':
        feat_dim = 512
    else:
        feat_dim = 2048
    feat_dim = 128

    if cfg.MODEL.METRIC_LOSS_TYPE == 'center':
        center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss

    elif cfg.MODEL.METRIC_LOSS_TYPE == 'range_center':
        center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center_range loss
        range_criterion = RangeLoss(k=cfg.SOLVER.RANGE_K, margin=cfg.SOLVER.RANGE_MARGIN, alpha=cfg.SOLVER.RANGE_ALPHA,
                                    beta=cfg.SOLVER.RANGE_BETA, ordered=True, use_gpu=True,
                                    ids_per_batch=cfg.SOLVER.IMS_PER_BATCH // cfg.DATALOADER.NUM_INSTANCE,
                                    imgs_per_id=cfg.DATALOADER.NUM_INSTANCE)

    elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_center':
        triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
        center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss

    elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_range_center':
        triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
        center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center_range loss
        range_criterion = RangeLoss(k=cfg.SOLVER.RANGE_K, margin=cfg.SOLVER.RANGE_MARGIN, alpha=cfg.SOLVER.RANGE_ALPHA,
                                    beta=cfg.SOLVER.RANGE_BETA, ordered=True, use_gpu=True,
                                    ids_per_batch=cfg.SOLVER.IMS_PER_BATCH // cfg.DATALOADER.NUM_INSTANCE,
                                    imgs_per_id=cfg.DATALOADER.NUM_INSTANCE)
    else:
        print('expected METRIC_LOSS_TYPE with center should be center, '
              'range_center,triplet_center, triplet_range_center '
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)     # new add by luo
        print("label smooth on, numclasses:", num_classes)

    lowfeat_dim = 512
    fcfeat_dim = 128
#    norm_m = AddMarginProduct(fcfeat_dim, num_classes, s=64, m=0.7)
#    norm_m_low = ArcMarginProduct(lowfeat_dim, num_classes, s=64, m=0.35, easy_margin=True)
    norm_m = ArcMarginProduct(fcfeat_dim, num_classes, s=64, m=0.35, easy_margin=False)
#    norm_m = A2MarginProduct(fcfeat_dim, num_classes, s=64, m=0.35, addm=0.35, easy_margin=False)
#    norm_m1 = ArcMarginProduct(fcfeat_dim, num_classes, s=64, m=0.35, easy_margin=True)
#    norm_m2 = ArcMarginProduct(fcfeat_dim, num_classes, s=64, m=0.35, easy_margin=True)

    def loss_func(feat_tpl, feat_sm, target):
#        low_feat = feat[:,:lowfeat_dim]
#        global_feat = feat[:,lowfeat_dim:]
#        score_low = norm_m_low(low_feat, target)
#        score = norm_m(feat, target)
#        feat = torch.cat((feat_tpl, feat_sm), 1)
        score = norm_m(feat_sm, target)
#        score1 = norm_m1(feat_tpl, target)
#        score2 = norm_m2(feat_sm, target)
        if cfg.MODEL.METRIC_LOSS_TYPE == 'center':
            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                return xent(score, target) + \
                        cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)  # new add by luo, open label smooth
            else:
                return F.cross_entropy(score, target) + \
                        cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)    # new add by luo, no label smooth

        elif cfg.MODEL.METRIC_LOSS_TYPE == 'range_center':
            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                return xent(score, target) + \
                        cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target) + \
                        cfg.SOLVER.RANGE_LOSS_WEIGHT * range_criterion(feat, target)[0] # new add by luo, open label smooth
            else:
                return F.cross_entropy(score, target) + \
                        cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target) + \
                        cfg.SOLVER.RANGE_LOSS_WEIGHT * range_criterion(feat, target)[0]     # new add by luo, no label smooth

        elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_center':
            return F.cross_entropy(score, target), score
#            return xent(feat_sm, target) + triplet(feat_tpl, target)[0], feat_sm

#            if cfg.MODEL.IF_LABELSMOOTH == 'on':
#                return xent(score, target) + \
#                        triplet(feat, target)[0] + \
#                        cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)  # new add by luo, open label smooth
#            else:
#                return F.cross_entropy(score, target) + \
#                        triplet(feat, target)[0] + \
#                        cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target)    # new add by luo, no label smooth


        elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_range_center':
            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                return xent(score, target) + \
                       triplet(feat, target)[0] + \
                       cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target) + \
                       cfg.SOLVER.RANGE_LOSS_WEIGHT * range_criterion(feat, target)[0]  # new add by luo, open label smooth
            else:
                return F.cross_entropy(score, target) + \
                       triplet(feat, target)[0] + \
                       cfg.SOLVER.CENTER_LOSS_WEIGHT * center_criterion(feat, target) + \
                       cfg.SOLVER.RANGE_LOSS_WEIGHT * range_criterion(feat, target)[0]  # new add by luo, no label smooth

        else:
            print('expected METRIC_LOSS_TYPE with center should be center,'
                  ' range_center, triplet_center, triplet_range_center '
                  'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))
    return loss_func, center_criterion, norm_m
