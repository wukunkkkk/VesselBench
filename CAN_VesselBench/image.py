import random
import os
from PIL import Image
import numpy as np
import h5py
import cv2
import torch
from torchvision import datasets, transforms

def load_data(img_path,train = True):
    gt_path = img_path.replace('.tif','.h5').replace('images','ground_truth_h5')
    img_sar_path = img_path.replace('images','images_sar')
    img = Image.open(img_path).convert('RGB')
    img_sar = Image.open(img_sar_path).convert('RGB')
    gt_file = h5py.File(gt_path,'r')
    target = np.asarray(gt_file['density'])

    transform=transforms.Compose([
                        transforms.ToTensor(),transforms.Normalize(mean=[0.3312, 0.3555, 0.3373],
                                        std=[0.1395, 0.1273, 0.1225]),
                    ])
    transform_sar=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.3591, 0.3591, 0.3591],
                                     std=[0.2103, 0.2103, 0.2103]),
                    ])

    if train:
        ratio = 0.5
        crop_size = (int(img.size[0]*ratio),int(img.size[1]*ratio))
        rdn_value = random.random()
        if rdn_value<0.25:
            dx = 0
            dy = 0
        elif rdn_value<0.5:
            dx = int(img.size[0]*ratio)
            dy = 0
        elif rdn_value<0.75:
            dx = 0
            dy = int(img.size[1]*ratio)
        else:
            dx = int(img.size[0]*ratio)
            dy = int(img.size[1]*ratio)

        img = img.crop((dx,dy,crop_size[0]+dx,crop_size[1]+dy))
        img_sar = img_sar.crop((dx,dy,crop_size[0]+dx,crop_size[1]+dy))
        target = target[dy:(crop_size[1]+dy),dx:(crop_size[0]+dx)]
        if random.random()>0.8:
            target = np.fliplr(target)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img_sar = img_sar.transpose(Image.FLIP_LEFT_RIGHT)

        img = transform(img)
        img_sar = transform_sar(img_sar)
        img_sar = img_sar[0:1, :, :]
        img = torch.cat((img, img_sar), dim=0)
    else:
        img = transform(img)
        img_sar = transform_sar(img_sar)
        img_sar = img_sar[0:1, :, :]
        img = torch.cat((img, img_sar), dim=0)

    target = cv2.resize(target, 
                    (int(target.shape[1] / 8), int(target.shape[0] / 8)), 
                    interpolation=cv2.INTER_CUBIC) * 64


    return img,target
