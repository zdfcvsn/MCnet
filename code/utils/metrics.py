from __future__ import division
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = np.multiply(val, weight)
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum = np.add(self.sum, np.multiply(val, weight))
        self.count = self.count + weight
        self.avg = self.sum / self.count

    @property
    def value(self):
        return self.val

    @property
    def average(self):
        return np.round(self.avg, 5)

def batch_pix_accuracy(output, target):
    _, predict = torch.max(output, 1)
    pixel_labeled = (target > 0).sum()
    pixel_correct = ((predict == target)*(target > 0)).sum()
    pixel_acc = np.divide(pixel_correct.cpu().numpy(), pixel_labeled.cpu().numpy())
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct.cpu().numpy(), pixel_labeled.cpu().numpy(), pixel_acc

def batch_intersection_union(output, target, num_class):
    _, predict = torch.max(output, 1)

    # predict = predict * (target > 0).long()
    intersection = predict * (predict == target).long()
    area_inter = torch.histc(intersection.float(), bins=num_class-1, max=num_class-0.9, min=0.1)
    # print(area_inter[0])
    area_pred = torch.histc(predict.float(), bins=num_class-1, max=num_class-0.9, min=0.1)
    area_lab = torch.histc(target.float(), bins=num_class-1, max=num_class-0.9, min=0.1)
    area_union = area_pred + area_lab - area_inter
    # print(area_union.float())
    IoU = area_inter.float()/(np.spacing(1) + area_union.float())
    mIoU = IoU.sum() / torch.nonzero(area_lab).size(0)

    assert (area_inter <= area_union).all(), "Intersection area should be smaller than Union area"

    return mIoU.cpu().numpy()



