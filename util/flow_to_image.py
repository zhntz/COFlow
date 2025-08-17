import numpy as np
import torch
from matplotlib.colors import hsv_to_rgb
import matplotlib.pyplot as plt
from util.frame_utils import *

def load_flow_to_numpy(path):
    with open(path, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        assert (202021.25 == magic), 'Magic number incorrect. Invalid .flo file'
        h = np.fromfile(f, np.int32, count=1)[0]
        w = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2 * w * h)
    data2D = np.resize(data, (w, h, 2))
    return data2D


def load_flow_to_png(path):
    flow = load_flow_to_numpy(path)
    print("load_flow",flow.shape)
    image = flow_to_image(flow)
    return image

# def cover_to_image(img):
#     if(img.shape == (1,2,256,1024)):
#         midFlow = img[0].data.cpu().numpy().transpose(1, 2, 0)
#         writeFlow("tmp.flo", midFlow)
#         t = load_flow_to_png("tmp.flo")
#         print("中级",type(t),"-",t.shape)
#         finalImg = torch.tensor(load_flow_to_png("tmp.flo").transpose(2, 0, 1)).unsqueeze(0)
#         return finalImg
#     return img

def flow_to_image(flow, max_flow=256):
    if max_flow is not None:
        max_flow = max(max_flow, 1.)
    else:
        max_flow = np.max(flow)

    n = 8
    u, v = flow[:, :, 0], flow[:, :, 1]
    mag = np.sqrt(np.square(u) + np.square(v))
    angle = np.arctan2(v, u)
    im_h = np.mod(angle / (2 * np.pi) + 1, 1)
    im_s = np.clip(mag * n / max_flow, a_min=0, a_max=1)
    im_v = np.clip(n - im_s, a_min=0, a_max=1)
    im = hsv_to_rgb(np.stack([im_h, im_s, im_v], 2))

    return (im * 255).astype(np.uint8)


if __name__ == '__main__':
    image = load_flow_to_png("/home/dell/flo-pytorch-CycleGAN-and-pix2pix-master/checkpoints/ambush_5/web/images/epoch010_flo.flo")
    print(image.shape)

    plt.imshow(image)
    plt.show()
