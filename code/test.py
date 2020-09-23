import torch
import numpy as np
import cv2
from models import MCNet
from PIL import Image
from utils import batch_pix_accuracy, batch_intersection_union
from torchvision import transforms
import os
import os.path as osp

avg_pix_acc = 0
avg_mIou = 0

defect_type = 'real'
img_dir = '../../data/RAIL/test/'+defect_type+'/images'
label_dir = '../../data/RAIL/test/'+defect_type+'/annotations'
result_path = './result/FCN8/'+defect_type+'/'

num_test = 0

net = MCNet(num_classes=2)
result_path = './result/'+net.modelName()+'/'+defect_type+'/'
model_pth = './save/checkpoints/'+net.modelName()+'/rail_mIou.pth'
if not os.path.exists(result_path):
    os.makedirs(result_path)
net.load_state_dict(torch.load(model_pth))
net.cuda()
print("Model loaded !")

net.eval()
mean = [0.48897059, 0.46548275, 0.4294]
std = [0.22861765, 0.22948039, 0.24054667]
normalize = transforms.Normalize(mean, std)
to_tensor = transforms.ToTensor()

for num_test, imgs in enumerate(os.listdir(img_dir)):
    # model
    image_path = osp.join(img_dir, imgs)
    label_path = osp.join(label_dir, imgs.split('.')[0]+'.jpg')

    img = np.asarray(Image.open(image_path).convert('RGB'), dtype=np.float32)
    label = np.asarray(Image.open(label_path), dtype=np.int32)  # - 1 # from -1 to 149
    label = label / 255
    height, weight = label.shape
    h, w = 400, 400
    image = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)
    image = normalize(to_tensor(image))
    image = image.unsqueeze(0)
    image = image.cuda()
    true_masks = torch.from_numpy(label).cuda().long()


    with torch.no_grad():
        output = net(image)
        index, predict = torch.max(output, 1)

        pix_acc = batch_pix_accuracy(output, true_masks)
        inter_union = batch_intersection_union(output, true_masks, 2)
        avg_pix_acc = avg_pix_acc + pix_acc[2]
        avg_mIou = avg_mIou + inter_union

        ###
        predict = predict.reshape((h, w))
        predict = predict.cpu().numpy()
        predict = predict * 255
        predict = cv2.resize(predict, (weight, height), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(result_path+imgs.split('.')[0]+'.jpg', predict)
        ###
avg_pix_acc = avg_pix_acc / (num_test+1)
avg_mIou = avg_mIou / (num_test+1)
print('--test -avg_pix_acc:{}, -avg_mIou:{}'.format(avg_pix_acc, avg_mIou))


