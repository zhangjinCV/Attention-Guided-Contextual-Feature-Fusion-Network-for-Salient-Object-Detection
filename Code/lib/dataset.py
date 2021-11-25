#!/usr/bin/python3
#coding=utf-8

import os
import cv2
import numpy as np
try:
    from . import transform
except:
    import transform

from torch.utils.data import Dataset

class Config(object):
    def __init__(self, **kwargs):
        self.kwargs    = kwargs
        self.mean = np.array([[[0.485*255, 0.456*255, 0.406*255]]])
        self.std = np.array([[[0.229*255, 0.224*255, 0.225*255]]])

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        else:
            return None


class Data(Dataset):
    def __init__(self, cfg):

        with open(os.path.join(cfg.datapath , cfg.mode+'.txt'), 'r') as lines:
            self.samples = []
            for line in lines:
                imagepath = os.path.join(cfg.datapath + '/image/', line.strip() + '.jpg'
                                         )
                maskpath = os.path.join(cfg.datapath + '/mask/', line.strip()) + '.png'

                self.samples.append([imagepath, maskpath])

        if cfg.mode == 'train':
            self.transform = transform.Compose(
                                                transform.Normalize(mean=cfg.mean, std=cfg.std),
                                                transform.Resize(352, 352),
                                                transform.RandomHorizontalFlip(),
                                                transform.RandomCrop(320, 320),
                                                transform.ToTensor())
        elif cfg.mode == 'test':
            self.transform = transform.Compose( transform.Normalize(mean=cfg.mean, std=cfg.std),
                                                transform.Resize(320, 320),
                                                transform.ToTensor())
        else:
            raise ValueError

    def __getitem__(self, idx):
        imagepath, maskpath = self.samples[idx]
        image               = cv2.imread(imagepath).astype(np.float32)[:,:,::-1]
        mask                = cv2.imread(maskpath).astype(np.float32)[:,:,::-1]
        H, W, C             = mask.shape
        image, mask         = self.transform(image, mask)

        return image, mask, (H, W), maskpath.split('/')[-1]

    def __len__(self):
        return len(self.samples)


if __name__=='__main__':
    import matplotlib.pyplot as plt
    plt.ion()
    cfg  = Config(mode='train', datapath=r'F:\Dataset\DUST\DUTS-TR')
    data = Data(cfg)
    for i in range(100):
        image, mask, (H,W), maskpath = data[i]
       # print(image, mask)
        image = image.permute(1,2,0).numpy()*cfg.std + cfg.mean
        mask  = mask.numpy()
        plt.subplot(121)
        plt.imshow(np.uint8(image))
        plt.subplot(122)
        plt.imshow(mask.reshape((320,320,1)), cmap='binary')
        plt.show()

        input()
