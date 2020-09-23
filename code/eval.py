import numpy as np
import torch
from torch.autograd import Function
from utils import batch_intersection_union, batch_pix_accuracy
import logging
def valid_epoch(net, dataset, criterion):
    net.eval()
    epoch_loss = 0
    epoch_pix = 0
    epoch_mIou = 0
    for i, (image, label) in enumerate(dataset):
        imgs = image.cuda()
        true_masks = label.cuda().long()
        masks_pred = net(imgs)
        masks_probs_flat = masks_pred
        true_masks_flat = true_masks
        loss = criterion(masks_probs_flat, true_masks_flat)
        epoch_loss += loss.item()
        batch_pix_acc = batch_pix_accuracy(masks_pred, true_masks)
        batch_mIou = batch_intersection_union(masks_pred, true_masks, 2)
        epoch_pix = epoch_pix + batch_pix_acc[2]
        epoch_mIou = epoch_mIou + batch_mIou
        if i % 20 == 0:
            print('val_loss: {:.4f}, val_batch_pix_acc:{:.4f}, val batch batch_mIou:{:.4f}'.format(
                loss.item(), batch_pix_acc[2], batch_mIou))
        # if i >0:
        #     break
    batches = len(dataset)
    epoch_loss = epoch_loss / batches
    epoch_pix = epoch_pix / batches
    epoch_mIou = epoch_mIou / batches
    print('-val Loss: {:.4f}, -epoch_pix: {:.4f}, -epoch_mIou: {:.4f}'.format(epoch_loss, epoch_pix, epoch_mIou))
    logging.info('-val Loss: {:.4f}, -epoch_pix: {:.4f}, -epoch_mIou: {:.4f}'.format(epoch_loss, epoch_pix, epoch_mIou))
    return epoch_pix, epoch_mIou
