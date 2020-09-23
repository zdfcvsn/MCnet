from dataloaders import RAILDataset
from torch.utils.data import DataLoader
from models import MCNet
from utils import CrossEntropyLoss2d
from torch import optim
from utils import batch_pix_accuracy, batch_intersection_union
from eval import valid_epoch
import torch
import os
import logging
import datetime

epochs = 50
batch_size = 4
val_batch = 2
train_set = RAILDataset(root='../../data/RAIL', split='training')
val_set = RAILDataset(root='../../data/RAIL', split='validation')
print(len(train_set))
train_data = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=0,drop_last=True)
val_data = DataLoader(dataset=val_set, batch_size=val_batch, shuffle=True, num_workers=0)

# model
net = MCNet(num_classes=2)
model_name = net.modelName()

dir_checkpoint = './save/checkpoints/'+model_name+'/'
if not os.path.exists(dir_checkpoint):
    os.mkdir(dir_checkpoint)
log = './save/log/'+model_name+'.log'
logging.basicConfig(level=logging.INFO, filename=log, filemode='a')
logging.info(datetime.datetime.now())
net.cuda()
# loss
criterion = CrossEntropyLoss2d()
# optimizer
optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=0.0005)
lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, gamma=0.5, step_size=5)

n_train = len(train_data)
best_mIou = 0
best_pix_acc = 0
for epoch in range(epochs):
    print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
    net.train()
    epoch_loss = 0
    print(datetime.datetime.now())
    for i, (image, label) in enumerate(train_data):
        imgs = image.cuda()
        true_masks = label.cuda().long()
        masks_pred = net(imgs)
        loss = criterion(masks_pred, true_masks)
        epoch_loss += loss.item()
        batch_pix_acc = batch_pix_accuracy(masks_pred, true_masks)
        batch_inter_union = batch_intersection_union(masks_pred, true_masks, 2)
        if i % 20 == 0:
            print('{:.4f} -Loss: {:4f}, -Pix_acc:{:4f}, -mIou:{:4f}'.format(
                   i/n_train, loss.item(), batch_pix_acc[2], batch_inter_union))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if i>2:
        #     break
    lr_schedule.step()
    print('Epoch finished ! Loss: {:.4f}'.format(epoch_loss / n_train))
    logging.basicConfig(level=logging.INFO, filename=log, filemode='a')
    logging.info('epoch {}_loss: {:.4f}\n'.format(epoch, epoch_loss / n_train))
    # val
    print('-------val---------')
    torch.cuda.empty_cache()
    pix_acc, mIou = valid_epoch(net, val_data, criterion)
    print(datetime.datetime.now())
    if mIou > best_mIou:
        best_mIou = mIou
        torch.save(net.state_dict(), dir_checkpoint + 'rail_mIou.pth'.format(epoch + 1))
        print('Checkpoint rail_mIou{} saved !'.format(epoch + 1))
        logging.info('epoch {} mIou save successful!\n'.format(epoch))
    torch.cuda.empty_cache()