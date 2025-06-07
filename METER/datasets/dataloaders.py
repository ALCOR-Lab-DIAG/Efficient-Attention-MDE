import matplotlib.pyplot as plt
import numpy as np
import random
import scipy
import skimage
import os
import torch
import torchvision.transforms
import torchvision.transforms.functional as TF

from PIL import Image
from scipy.sparse.linalg import spsolve
from torch.utils.data import DataLoader

from METER.datasets.data_augmentation import *
from METER.utils import *

def fill_depth_colorization(imgRgb=None, imgDepthInput=None, alpha=1):
    imgIsNoise = imgDepthInput == 0
    maxImgAbsDepth = np.max(imgDepthInput)
    imgDepth = imgDepthInput / maxImgAbsDepth
    imgDepth[imgDepth > 1] = 1
    (H, W) = imgDepth.shape
    numPix = H * W
    indsM = np.arange(numPix).reshape((W, H)).transpose()
    knownValMask = (imgIsNoise == False).astype(int)
    grayImg = skimage.color.rgb2gray(imgRgb)
    winRad = 1
    len_ = 0
    absImgNdx = 0
    len_window = (2 * winRad + 1) ** 2
    len_zeros = numPix * len_window

    cols = np.zeros(len_zeros) - 1
    rows = np.zeros(len_zeros) - 1
    vals = np.zeros(len_zeros) - 1
    gvals = np.zeros(len_window) - 1

    for j in range(W):
        for i in range(H):
            nWin = 0
            for ii in range(max(0, i - winRad), min(i + winRad + 1, H)):
                for jj in range(max(0, j - winRad), min(j + winRad + 1, W)):
                    if ii == i and jj == j:
                        continue
                    rows[len_] = absImgNdx
                    cols[len_] = indsM[ii, jj]
                    gvals[nWin] = grayImg[ii, jj]
                    len_ = len_ + 1
                    nWin = nWin + 1

            curVal = grayImg[i, j]
            gvals[nWin] = curVal
            c_var = np.mean((gvals[:nWin + 1] - np.mean(gvals[:nWin + 1])) ** 2)
            csig = c_var * 0.6
            mgv = np.min((gvals[:nWin] - curVal) ** 2)
            if csig < -mgv / np.log(0.01):
                csig = -mgv / np.log(0.01)
            if csig < 2e-06:
                csig = 2e-06
            gvals[:nWin] = np.exp(-(gvals[:nWin] - curVal) ** 2 / csig)
            gvals[:nWin] = gvals[:nWin] / sum(gvals[:nWin])
            vals[len_ - nWin:len_] = -gvals[:nWin]
            # Now the self-reference (along the diagonal).
            rows[len_] = absImgNdx
            cols[len_] = absImgNdx
            vals[len_] = 1  # sum(gvals(1:nWin))
            len_ = len_ + 1
            absImgNdx = absImgNdx + 1

    vals = vals[:len_]
    cols = cols[:len_]
    rows = rows[:len_]
    A = scipy.sparse.csr_matrix((vals, (rows, cols)), (numPix, numPix))
    rows = np.arange(0, numPix)
    cols = np.arange(0, numPix)
    vals = (knownValMask * alpha).transpose().reshape(numPix)
    G = scipy.sparse.csr_matrix((vals, (rows, cols)), (numPix, numPix))
    A = A + G
    b = np.multiply(vals.reshape(numPix), imgDepth.flatten('F'))
    # print ('Solving system..')
    new_vals = spsolve(A, b)
    new_vals = np.reshape(new_vals, (H, W), 'F')
    # print ('Done.')
    denoisedDepthImg = new_vals * maxImgAbsDepth
    output = denoisedDepthImg.reshape((H, W)).astype('float32')
    output = np.multiply(output, (1 - knownValMask)) + imgDepthInput

    return output


def custom_PIL_crop(im, new_width, new_height):
    width, height = im.size  # Get dimensions

    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2

    # Crop the center of the image
    im = im.crop((left, top, right, bottom))
    return im


def heat_map(depth):
    def lin_interp(shape, xyd):
        # taken from https://github.com/hunse/kitti
        m, n = shape
        ij, d = xyd[:, 1::-1], xyd[:, 2]
        f = LinearNDInterpolator(ij, d, fill_value=0)
        J, I = np.meshgrid(np.arange(n), np.arange(m))
        IJ = np.vstack([I.flatten(), J.flatten()]).T
        disparity = f(IJ).reshape(shape)
        return disparity

    y, x = np.where(depth > 0.0)
    d = depth[depth != 0]
    xyd = np.stack((x, y, d)).T
    gt = lin_interp(depth.shape, xyd)
    gt = np.expand_dims(gt, axis=-1)

    return gt


class KITTI_Dataset:
    """
      * Outdoor img (375, 1242, 3) depth (375, 1242, 1) both in png -> range between 0.5 to 80 meters
      * 697 Test and 23158 Train images (no calibration images)
      * Refers to kitti_multi_worker for filling images
    """

    def __init__(self, args, path, dts_type, aug, rgb_h_res, d_h_res, dts_size=0, scenarios='outdoor'):
        self.args = args
        self.dataset = path
        self.x = []
        self.y = []
        self.info = 0
        self.dts_type = dts_type
        self.aug = aug
        self.rgb_h_res = rgb_h_res
        self.d_h_res = d_h_res
        self.scenarios = scenarios

        with open(args.filenames_file, 'r') as f:
            self.filenames = f.readlines()
        with open(args.filenames_file_eval, 'r') as f:
            self.filenames_eval = f.readlines()
        
        if dts_type == 'test':
            for filename_eval in self.filenames_eval:
                file_name = filename_eval.split(' ')
                if file_name[1] != "None":
                    self.x.append(os.path.join(self.args.data_path_eval,file_name[0]))
                    self.y.append(os.path.join(self.args.gt_path_eval,file_name[1]))

        elif dts_type == 'train':
            for filename in self.filenames:
                file_name = filename.split(' ')
                self.x.append(os.path.join(self.args.data_path,file_name[0]))
                self.y.append(os.path.join(self.args.gt_path,file_name[1]))

        else:
            raise SystemError('Problem in the dts_path')

        if len(self.x) != len(self.y):
            raise SystemError('Problem with Img and Gt, no same train_size')

        self.x.sort()
        self.y.sort()

        if dts_size != 0:
            self.x = self.x[:dts_size]
            self.y = self.y[:dts_size]
        
        self.info = len(self.x)

    def __len__(self):
        return self.info

    def __getitem__(self, index=None, print_info_aug=False, do_heatmap=False):
        if index is None:
            index = np.random.randint(0, self.info)
        
        def depth_read(filename):
            depth_png = np.array(custom_PIL_crop(Image.open(filename), new_width=self.args.rgb_img_res[2], new_height=self.args.rgb_img_res[1]))
            depth = depth_png / 256
            depth[depth_png == 0] = -1.0
            return depth

        # Load Image
        img = Image.open(self.x[index]).convert('RGB')
        img = np.array(custom_PIL_crop(img, new_width=self.args.rgb_img_res[2], new_height=self.args.rgb_img_res[1]))

        # Load Depth Image
        depth_path = self.y[index]
        depth = depth_read(self.y[index])

        # Normalization
        img = np.clip(img / 255, 0.0, 1.0)
        depth = np.clip(depth, 0.0, 80.0)

        # Filling
        depth = np.expand_dims(depth, axis=-1)

        if self.aug:
            img, depth = augmentation2D(self.args, img, depth, print_info_aug)

        img_post_processing = torchvision.transforms.Compose([
            TT.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            torchvision.transforms.Resize((self.args.rgb_img_res[1], self.args.rgb_img_res[2]), antialias=True),
        ])
        depth_post_processing = torchvision.transforms.Compose([
            TT.ToTensor(),
            torchvision.transforms.Resize((self.args.d_img_res[1], self.args.d_img_res[2]), antialias=True),
        ])

        img = img_post_processing(img)
        depth = depth_post_processing(depth)
        
        return img.float(), depth.float()

class NYU2_Dataset:
    """
      * Indoor img (480, 640, 3) depth (480, 640, 1) both in png -> range between 0.5 to 10 meters
      * 654 Test and 50688 Train images
    """

    def __init__(self, args, path, dts_type, aug, rgb_h_res, d_h_res, dts_size=0, scenarios='indoor'):
        self.args = args        
        self.dataset = path
        self.x = []
        self.y = []
        self.info = 0
        self.dts_type = dts_type
        self.aug = aug
        self.rgb_h_res = rgb_h_res
        self.d_h_res = d_h_res
        self.scenarios = scenarios

        # Handle dataset
        if self.dts_type == 'test':
            img_path = self.dataset + self.dts_type + '/eigen_test_rgb.npy'
            depth_path = self.dataset + self.dts_type + '/eigen_test_depth.npy'

            rgb = np.load(img_path)
            depth = np.load(depth_path)

            self.x = rgb
            self.y = depth

            if dts_size != 0:
                self.x = rgb[:dts_size]
                self.y = depth[:dts_size]

            self.info = len(self.x)

        elif self.dts_type == 'train':
            scenarios = os.listdir(self.dataset + self.dts_type + '/')
            for scene in scenarios:
                elem = os.listdir(self.dataset + self.dts_type + '/' + scene)
                for el in elem:
                    if 'jpg' in el:
                        self.x.append(self.dts_type + '/' + scene + '/' + el)
                    elif 'png' in el:
                        self.y.append(self.dts_type + '/' + scene + '/' + el)
                    else:
                        raise SystemError('Type image error (train)')

            if len(self.x) != len(self.y):
                raise SystemError('Problem with Img and Gt, no same train_size')

            self.x.sort()
            self.y.sort()

            if dts_size != 0:
                self.x = self.x[:dts_size]
                self.y = self.y[:dts_size]

            self.info = len(self.x)

        else:
            raise SystemError('Problem in the path')

    def __len__(self):
        return self.info

    def __getitem__(self, index=None, print_info_aug=False):
        if index is None:
            index = np.random.randint(0, self.info)

        # Load Image
        if self.dts_type == 'test':
            img = self.x[index]
        else:
            img = Image.open(self.dataset + self.x[index]).convert('RGB')
            img = np.array(img)

        # Load Depth Image
        if self.dts_type == 'test':
            depth = np.expand_dims(self.y[index] * 100, axis=-1)
        else:
            depth = Image.open(self.dataset + self.y[index])
            depth = np.array(depth) / 255
            depth = np.clip(depth * 1000, 50, 1000)
            depth = np.expand_dims(depth, axis=-1)

        # Augmentation
        if self.aug:
            img, depth = augmentation2D(self.args, img, depth, print_info_aug)


        img_post_processing = TT.Compose([
            TT.ToTensor(),
            TT.Resize((self.args.rgb_img_res[1], self.args.rgb_img_res[2]), antialias=True),
            # TT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Imagenet
            TT.Normalize(mean=[0.481, 0.410, 0.392], std=[0.071, 0.075, 0.080]) # Best model
        ])
        depth_post_processing = TT.Compose([
            TT.ToTensor(),
            TT.Resize((self.args.d_img_res[1], self.args.d_img_res[2]), antialias=True),
        ])

        img = img_post_processing(img/255)
        depth = depth_post_processing(depth)

        return img.float(), depth.float()

#---------------------------------------------------- Dataloader -------------------------------------------------#

def init_train_test_loader(args, dts_type, dts_root_path, rgb_h_res, d_h_res, bs_train, bs_eval, num_workers, size_train=0, size_test=0):
    if dts_type == 'nyu':
        Dataset_class = NYU2_Dataset
        dts_root_path = dts_root_path + 'NYUv2/'
    elif dts_type == 'kitti':
        Dataset_class = KITTI_Dataset
        dts_root_path = dts_root_path + 'kitti/'
    else:
        print('OCCHIO AL DATASET')


    # Load Datasets
    test_Dataset = Dataset_class(
        args=args, path=dts_root_path, dts_type='test', aug=False, rgb_h_res=rgb_h_res, d_h_res=d_h_res, dts_size=size_test
    )
    training_Dataset = Dataset_class(
        args=args, path=dts_root_path, dts_type='train', aug=True, rgb_h_res=rgb_h_res, d_h_res=d_h_res, dts_size=size_train
    )
    # Create Dataloaders
    training_DataLoader = DataLoader(
        training_Dataset, batch_size=bs_train, shuffle=True, num_workers=num_workers
    )
    test_DataLoader = DataLoader(
        test_Dataset, batch_size=bs_eval, shuffle=False, num_workers=num_workers
    )

    return training_DataLoader, test_DataLoader, training_Dataset, test_Dataset