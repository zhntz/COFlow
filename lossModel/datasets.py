# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data
import random

from torch import nn


def readFlow(flow):
    flow = flow[:,:,::-1].astype(np.float32)
    flow, valid = flow[:, :, :2], flow[:, :, 2]
    flow = (flow - 2**15) / 64.0
    return flow, valid

class FlowDataset(nn.Module):
    def __init__(self, aug_params=None,mv=None,res=None,reFLow=None,predFlow=None):
        super().__init__()
        from lossModel.augmentor import FlowAugmentor
        self.augmentor = FlowAugmentor(crop_size=(256,1024))

        self.img1 = mv
        self.img2 = res
        self.flow = reFLow
        self.valid = predFlow


        # self.is_test = False
        self.init_seed = False

    def forward(self):
        # if self.is_test:
        #     img1 = np.array(self.img1).astype(np.uint8)[..., :3]
        #     img2 = np.array(self.img2).astype(np.uint8)[..., :3]
        #     img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        #     img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        #     # return img1, img2, self.extra_info[index]
        #     return img1, img2

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        flow, valid = readFlow(self.flow)


        flow = np.array(flow).astype(np.float32)
        img1 = np.array(self.img1).astype(np.uint8)
        img2 = np.array(self.img2).astype(np.uint8)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)
        self.valid = valid
        if(valid is not None):
            print("nokong")
        else:
            print(" kong")
        # return img1, img2, flow, valid.float()
        return valid.float()
        






