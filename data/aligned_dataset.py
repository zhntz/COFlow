import os
from data.base_dataset import BaseDataset, get_params, get_transform, get_transform_flo
from data.flo2png import flow_to_image
from data.image_folder import make_dataset
from PIL import Image
import numpy as np

from util.frame_utils import readFlow , readFlow_kitti , flowtransform


#aligned_dataset.py包含一个可以加载图像对的数据集类。它设置好了一个图像目录/path/to/data/train，
#其中包含 {A,B} 形式的图像对。在测试期间，您需要准备一个目录/path/to/data/test作为测试数据。

class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_ABC = os.path.join(opt.dataroot, opt.phase)  # get the image directory       获取数据路径
        self.dir_flo = os.path.join(opt.floroot, opt.phase)  #get the flo root
        self.dir_block = os.path.join(opt.mbroot, opt.phase)
        self.dir_occlusion = os.path.join(opt.occroot, opt.phase)

        self.ABC_paths = sorted(make_dataset(self.dir_ABC, opt.max_dataset_size))  # get image paths  返回图像列表
        self.flo_paths = sorted(make_dataset(self.dir_flo, opt.max_dataset_size))
        self.block_paths = sorted(make_dataset(self.dir_block, opt.max_dataset_size))
        self.occlusion_paths = sorted(make_dataset(self.dir_occlusion, opt.max_dataset_size))

        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image  确保裁剪大小小于图片本身大小
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        ABC_path = self.ABC_paths[index]
        flo_path = self.flo_paths[index]

        block_path = self.block_paths[index]
        occlusion_path = self.occlusion_paths[index]

        ABC = Image.open(ABC_path).convert('RGB')   #单独获取一张图片并且转换为RGB
        ############################
        # def load_flow_to_numpy(path):
        #     with open(path, 'rb') as f:
        #         magic = np.fromfile(f, np.float32, count=1)
        #         assert (202021.25 == magic), 'Magic number incorrect. Invalid .flo file'
        #         h = np.fromfile(f, np.int32, count=1)[0]
        #         w = np.fromfile(f, np.int32, count=1)[0]
        #         data = np.fromfile(f, np.float32, count=2 * w * h)
        #     data2D = np.resize(data, (w, h, 2))
        #     FLO = data2D[180:, :, :]
        #     return FLO
        ############################
        # print("oath",flo_path)

        FLO = readFlow(flo_path)
        ###FLO = readFlow_kitti(flo_path)

        BLOCK = Image.open(block_path).convert('RGB')
        OCCLUSION = Image.open(occlusion_path).convert('RGB')
        # FLO = FLO.reshape(2,1024,256)
        ############################
        # split AB image into A and B
        w, h = ABC.size  # 获取宽和高
        w2 = int(w / 3)
        w3 = 2 * w2
        A = ABC.crop((0, 0, w2, h))  # 对齐两幢图片    从左至右依次是A,C,B
        C = ABC.crop((w2, 0, w3, h))
        B = ABC.crop((w3, 0, w, h))

        # print("A",A.size)
        # print(type(A))
        # print("FLO",FLO.size)
        # print(type(FLO))
        # print("//////////////////////////////////////////////")
        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        C_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        BLOCK_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        OCCLUSION_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)
        C = C_transform(C)
        #A = A * 255
        #B = B * 255
        #C = C * 255
        BLOCK = BLOCK_transform(BLOCK)
        OCCLUSION = OCCLUSION_transform(OCCLUSION)

        # flo_transform_params = get_params(self.opt, FLO.size)

        ###FLO = flowtransform(FLO, transform_params, 256)

        FLO_transform = get_transform_flo(self.opt, transform_params, grayscale=(self.output_nc == 1))

        flo = FLO_transform(FLO)
        flo = flo.permute(2, 0, 1)
        flo = flo / 426.31052      #426.31052 spi               #225.375  fly
        # print("flo",flo.shape)
        # print("a", A.shape)
        return {'A': A, 'B': B, 'C': C, 'FLO':flo, 'mb':BLOCK, 'occ':OCCLUSION, 'A_paths': ABC_path, 'B_paths': ABC_path,'C_paths': ABC_path, 'flo_path':flo_path, 'mb_path':block_path, 'occ_path':occlusion_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.ABC_paths)
