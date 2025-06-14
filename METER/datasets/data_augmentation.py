import numpy as np
import random

from itertools import product

def pixel_shift(depth_img, shift):
    depth_img = depth_img + shift
    return depth_img

def random_crop(x, y, crop_size=(192, 256)):
    assert x.shape[0] == y.shape[0]
    assert x.shape[1] == y.shape[1]
    h, w, _ = x.shape
    rangew = (w - crop_size[0]) // 2 if w > crop_size[0] else 0
    rangeh = (h - crop_size[1]) // 2 if h > crop_size[1] else 0
    offsetw = 0 if rangew == 0 else np.random.randint(rangew)
    offseth = 0 if rangeh == 0 else np.random.randint(rangeh)
    cropped_x = x[offseth:offseth + crop_size[0], offsetw:offsetw + crop_size[1], :]
    cropped_y = y[offseth:offseth + crop_size[0], offsetw:offsetw + crop_size[1], :]
    cropped_y = cropped_y[:, :, ~np.all(cropped_y == 0, axis=(0, 1))]
    if cropped_y.shape[-1] == 0:
        return x, y
    else:
        return cropped_x, cropped_y

def random_crop_kitti(x, y, crop_size=(288, 384)):
    assert x.shape[0] == y.shape[0]
    assert x.shape[1] == y.shape[1]
    h, w, _ = x.shape
    rangew = (w - crop_size[0]) // 2 if w > crop_size[0] else 0
    rangeh = (h - crop_size[1]) // 2 if h > crop_size[1] else 0
    offsetw = 0 if rangew == 0 else np.random.randint(rangew)
    offseth = 0 if rangeh == 0 else np.random.randint(rangeh)
    cropped_x = x[offseth:offseth + crop_size[0], offsetw:offsetw + crop_size[1], :]
    cropped_y = y[offseth:offseth + crop_size[0], offsetw:offsetw + crop_size[1], :]
    cropped_y = cropped_y[:, :, ~np.all(cropped_y == 0, axis=(0, 1))]
    if cropped_y.shape[-1] == 0:
        return x, y
    else:
        return cropped_x, cropped_y

def augmentation2D(args, img, depth, print_info_aug):
    if args.dts_type == "nyu":
        # Random flipping
        if random.uniform(0, 1) <= args.flip:
            img = (img[..., ::1, :, :]).copy()
            depth = (depth[..., ::1, :, :]).copy()
            if print_info_aug:
                print('--> Random flipped')
        # Random mirroring
        if random.uniform(0, 1) <= args.mirror:
            img = (img[..., ::-1, :]).copy()
            depth = (depth[..., ::-1, :]).copy()
            if print_info_aug:
                print('--> Random mirrored')
        # Augment image
        if random.uniform(0, 1) <= args.color_and_bright:
            # gamma augmentation
            gamma = random.uniform(0.9, 1.1)
            img = img ** gamma
            brightness = random.uniform(0.9, 1.1)
            img = img * brightness
            # color augmentation
            colors = np.random.uniform(0.9, 1.1, size=3)
            white = np.ones((img.shape[0], img.shape[1]))
            color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
            img *= color_image
            img = np.clip(img, 0, 255)  # Originally with 0 and 1
            if print_info_aug:
                print('--> Image randomly augmented')
        # Channel swap
        if random.uniform(0, 1) <= args.c_swap:
            indices = list(product([0, 1, 2], repeat=3))
            policy_idx = random.randint(0, len(indices) - 1)
            img = img[..., list(indices[policy_idx])]
            if print_info_aug:
                print('--> Channel swapped')
        # Random crop
        if random.random() <= args.random_crop:
            img, depth = random_crop(img, depth)
            if print_info_aug:
                print('--> Random cropped')
        # Depth Shift
        if random.random() <= args.random_d_shift:
            random_shift = random.randint(-10, 10)
            depth = pixel_shift(depth, shift=random_shift)
            if print_info_aug:
                print('--> Depth Shifted of {} cm'.format(random_shift))
    else:
        # Random flipping
        if random.uniform(0, 1) <= args.flip:
            img = (img[..., ::1, :, :]).copy()
            depth = (depth[..., ::1, :, :]).copy()
            
            if print_info_aug:
                print('--> Random flipped')
        # Random mirroring
        if random.uniform(0, 1) <= args.mirror:
            img = (img[..., ::-1, :]).copy()
            depth = (depth[..., ::-1, :]).copy()
            if print_info_aug:
                print('--> Random mirrored')
        # Augment image
        if random.random() <= args.color_and_bright:
            # gamma augmentation
            gamma = random.uniform(0.9, 1.1)
            img = img ** gamma
            brightness = random.uniform(0.9, 1.1)
            img = img * brightness
            # color augmentation
            colors = np.random.uniform(0.9, 1.1, size=3)
            white = np.ones((img.shape[0], img.shape[1]))
            color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
            img *= color_image
            img = np.clip(img, 0, 255)  # Originally with 0 and 1
            if print_info_aug:
                print('--> Image randomly augmented')
        # Channel swap
        if random.uniform(0, 1) <= args.c_swap:
            indices = list(product([0, 1, 2], repeat=3))
            policy_idx = random.randint(0, len(indices) - 1)
            img = img[..., list(indices[policy_idx])]
            if print_info_aug:
                print('--> Channel swapped')
        # Random crop
        if random.random() <= args.random_crop:
            img, depth = random_crop_kitti(img, depth)
            if print_info_aug:
                print('--> Random cropped')

    return img, depth