#!/usr/bin/python3
#coding=utf-8

import os
import cv2
import matplotlib.pyplot as plt
plt.ion()
from warnings import filterwarnings
filterwarnings('ignore')
import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from lib import dataset
from saliency_toolbox import calculate_measures


TAG = "ACFFNet"
SAVE_PATH = TAG
GPU_ID=0
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)


class Test(object):
    def __init__(self, Dataset, datapath, Network, model_paths=None):

        self.datapath = datapath.split("/")[-1]
        self.cfg = Dataset.Config(datapath = datapath, mode='test')
        self.data   = Dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=bs, shuffle=True, num_workers=0)
        ## network
        self.net = Network(self.cfg)
        self.net.train(False)
       # self.net.cuda()
        self.net.eval()
        model_path = model_paths
        self.net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    def pre_pics(self, save_path=None):
        with torch.no_grad():
            stepss = 0
            for image, mask, (H, W), maskpath in tqdm.tqdm(self.loader):
                stepss += 1
                image, mask = image.float(), mask.float()
                out, _, _, _ = self.net(image)
                pred = torch.sigmoid(out)

                k_pred = pred
                for num in range(len(H)):
                    mae_pred = k_pred[num].unsqueeze(0)
                    mae_pred = F.interpolate(mae_pred, size=(H[num], W[num]), mode='bilinear', align_corners=True)
                    if save_path:
                        save_paths = os.path.join(save_path, self.cfg.datapath.split('\\')[-1])
                        if not os.path.exists(save_paths):
                            os.makedirs(save_paths)
                        mae_pred = mae_pred[0].permute(1, 2, 0) * 255
                        cv2.imwrite(save_paths + '\\' + maskpath[num][0:-4] + '.jpg', mae_pred.cpu().numpy())


def test_socre(path, save=None):
    sms_dir = [
       path + '/DUTS-TE/',
       path + '/ECSSD/',
       path + '/HKU-IS/',
       path + '/DUT-OMRON/',
       path + '/PASCAL-S/'
    ]
    gts_dir = [
        '../Dataset/DUST/DUTS-TE/mask/',
        '../Dataset/ECSSD/mask/',
        '../Dataset/HKU-IS/mask/',
        '../Dataset/DUT-OMRON/mask/',
        '../Dataset/PASCAL-S/mask/'

    ]
    measures = ['MAE', 'E-measure', 'Adp-F', 'Max-F', 'S-measure', 'Wgt-F']
    for i in range(len(gts_dir)):
        res = calculate_measures(gts_dir[i], sms_dir[i], measures, save=save)
        print(gts_dir[i].split('/')[-3], 'MAE:', res['MAE'], 'Fm:', res['Mean-F'], 'E-measure:', res['E-measure'],
              'S-measure:', res['S-measure'], 'Wgt-F:', res['Wgt-F'])


def save_pigs(model, model_path, save_path=None):
    DATASETS = [
       r'../Dataset/PASCAL-S',
       r'../Dataset/ECSSD',
       r'../Dataset/DUTS/DUTS-TE',
       r'../Dataset/DUT-OMRON',
       r'../Dataset/HKU-IS',
    ]
    for e in DATASETS:
        t = Test(dataset, e, model, model_path)
        t.pre_pics(save_path=save_path)


if __name__=='__main__':
    import os
    from net2 import ACFFNet
    model_path = r'../weight/ACFFNet.pth'
    bs = 1
    save_pigs(ACFFNet, model_path)






