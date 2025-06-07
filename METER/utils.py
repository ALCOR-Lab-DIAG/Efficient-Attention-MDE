import argparse
import csv
import gc
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import pytz
import random
import smtplib
import shutil
import skimage.transform as st
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as TT
import torchvision.transforms.functional as TF
import warnings

from datetime import datetime
from einops import rearrange
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from itertools import product
from math import exp
from PIL import Image
from scipy.interpolate import LinearNDInterpolator
from time import perf_counter
from torch.utils.data import DataLoader
from torchsummaryX import summary
from torchvision import transforms
from tqdm import tqdm

def send_email(network, version, optimization, error=False, error_msg=None):
    rome_tz = pytz.timezone("Europe/Rome")
    current_time = datetime.now()
    print_date = current_time.strftime("%d/%m/%Y %H:%M:%S")

    YOUR_GOOGLE_EMAIL = 'sapienza.amr.2024@gmail.com'  # The email you setup to send the email using app password
    YOUR_GOOGLE_EMAIL_APP_PASSWORD = 'ytdo rxyn bocy ezap'  # The app password you generated

    # Setup the SMTP server
    smtpserver = smtplib.SMTP_SSL('smtp.gmail.com', 465)
    smtpserver.ehlo()
    smtpserver.login(YOUR_GOOGLE_EMAIL, YOUR_GOOGLE_EMAIL_APP_PASSWORD)

    # Create the email
    sent_from = YOUR_GOOGLE_EMAIL
    sent_to = sent_from  # Send it to self (as test)
    
    if not error:
        subject = '[RTX5000] Training ended'
        body = f"The train is correctly ended on {print_date}. The configuration was: \n\t- Network: {network} \n\t- Size: {version} \n\t- Optimization: {optimization}"
    else:
        subject = '!!! [RTX5000] Training ERROR !!!'
        body = f"The train of {network} {version} {optimization} is UNEXPLECTEDLY ended on {print_date}. The error was: \n\n {error_msg}"

    # Use MIMEMultipart to create an email with subject and body
    message = MIMEMultipart()
    message['From'] = sent_from
    message['To'] = sent_to
    message['Subject'] = subject

    # Attach the email body
    message.attach(MIMEText(body, 'plain'))

    # Send the email
    smtpserver.sendmail(sent_from, sent_to, message.as_string())

    # Close the connection
    smtpserver.close()

class TextFormatter:
    # ANSI escape codes for colors
    COLORS = {
        'black': '\033[30m',
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'purple': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
    }

    # ANSI escape codes for text styles
    STYLES = {
        'normal': '\033[0m',
        'bold': '\033[1m',
        'underline': '\033[4m',
        'blink': '\033[5m',
    }

    def __init__(self):
        self.terminal_width = shutil.get_terminal_size().columns
        self.max_length = 100

    def format(self, text, color=None, style=None, separator=False):
        color_code = self.COLORS.get(color, '')
        style_code = ''.join([self.STYLES.get(style, '') for style in style]) if style else ''
        formatted_text = f"{style_code}{color_code}{text}{self.STYLES['normal']}"

        if separator:
            separator_length = self.max_length - len(formatted_text) - 6
            separator_line = f"{self.STYLES['bold']}{self.COLORS.get(color, '')}{'*' * (separator_length // 2)}{self.STYLES['normal']}"
            formatted_text = f"{separator_line} {formatted_text} {separator_line}"

        return formatted_text

def hardware_check():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    print(f"[LOG] Current device: {device}")
    if 'cuda' in device:
        prop = torch.cuda.get_device_properties(device)
        print(f"[LOG] Device name: {prop.name}")
        print(f"[LOG] Device memory: {prop.total_memory/ (1024 ** 2)} MB")
        print(f"[LOG] Device processors: {prop.multi_processor_count}")
        
    return device

def format_string(input_str):
    # Separare la stringa in parti usando l'underscore come delimitatore
    parts = input_str.split('_')
    # Capitalizzare ogni parte
    formatted_parts = [parts[0].capitalize(),parts[1].upper()]
    # Unire le parti con uno spazio
    return ' '.join(formatted_parts)

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg

def none_or_int(value):
    if value == "None":
        return None
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid value: {value}. Must be an integer or 'None'.")

def str_to_tuple(value):
    try:
        return tuple(map(int, value.strip('()').split(',')))
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid tuple value: {value}")

def init_arg_parser():
    parser = argparse.ArgumentParser(description='Custom implementation.', fromfile_prefix_chars='@')
    parser.convert_arg_line_to_args = convert_arg_line_to_args
    
    # Resolutions
    parser.add_argument('--mode', type=str, help='Model name', default='meter')
    parser.add_argument('--model_name', type=str, help='Model name', default='meter')
    parser.add_argument('--rgb_img_res', type=str_to_tuple, help='Resolution of RGB images', default=(3, 192, 256))
    parser.add_argument('--d_img_res', type=str_to_tuple, help='Resolution of depth images', default=(1, 48, 64))
    
    # Operations
    parser.add_argument('--do_prints', action='store_true', help='Whether to print logs', default=False)
    parser.add_argument('--do_print_model', action='store_true', help='Whether to print the model structure', default=False)
    parser.add_argument('--do_pretrained', action='store_true', help='Whether to use pretrained weights', default=True)
    parser.add_argument('--do_train', action='store_true', help='Whether to train the model', default=False)
    parser.add_argument('--do_print_best_worst', action='store_true', help='Whether to print best and worst results', default=True)
        
    # Parameters
    parser.add_argument('--pretrained_w', type=str, help='Pretrained weights', default='')
    parser.add_argument('--dts_type', type=str, help='Dataset type (e.g., nyu)', default='nyu')
    parser.add_argument('--architecture_type', type=str, help='Type of architecture (e.g., s)', default='s')
    parser.add_argument('--seed', type=int, help='Random seed', default=42)
    parser.add_argument('--lr', type=float, help='Learning rate', default=1e-3)
    parser.add_argument('--lr_patience', type=int, help='Patience for learning rate scheduler', default=15)
    parser.add_argument('--epochs', type=int, help='Number of epochs', default=80)
    parser.add_argument('--batch_size', type=int, help='Batch size for training', default=64)
    parser.add_argument('--batch_size_eval', type=int, help='Batch size for evaluation', default=1)
    parser.add_argument('--n_workers', type=int, help='Number of workers for data loading', default=2)
    parser.add_argument('--e_stop_epochs', type=int, help='Early stopping epochs', default=30)
    parser.add_argument('--size_train', type=none_or_int, help='Size of training set (None for full size)', default=None)
    parser.add_argument('--size_test', type=none_or_int, help='Size of test set (None for full size)', default=None)
    
    # Augmentation
    parser.add_argument('--flip', type=float, help='Probability of horizontal flip', default=0.5)
    parser.add_argument('--mirror', type=float, help='Probability of vertical flip', default=0.5)
    parser.add_argument('--color_and_bright', type=float, help='Probability of color and brightness augmentation', default=0.5)
    parser.add_argument('--c_swap', type=float, help='Probability of channel swap', default=0.5)
    parser.add_argument('--random_crop', type=float, help='Probability of random cropping', default=0.5)
    parser.add_argument('--random_d_shift', type=float, help='Range of random depth shift in cm (+-)', default=0.5)
    
    # Network and Paths
    parser.add_argument('--network_type', type=str, help='Type of network', default="METER")
    parser.add_argument('--dataset_root', type=str, help='Root path to the dataset', default='/work/data/ssd_datasets/')
    parser.add_argument('--save_model_root', type=str, help='Root path to save the model', default='/work/data/ssd_results/')
    parser.add_argument('--imagenet_init', type=str, help='Path for ImageNet initialization', default='/work/imagenet/')  # Required to keep but unused by the user
    parser.add_argument('--checkpoint', type=str, help='Path for checkpoint', default='/work/project/')

    parser.add_argument('--data_path',                 type=str,   help='path to the data',  default="work/project/PixelFormer/datasets/nyu_depth_v2/official_splits/test/")
    parser.add_argument('--gt_path',                   type=str,   help='path to the groundtruth data', default="work/project/PixelFormer/datasets/nyu_depth_v2/official_splits/test/")
    parser.add_argument('--filenames_file',            type=str,   help='path to the filenames text file', default="work/project/PixelFormer/data_splits/nyudepthv2_test_files_with_gt.txt")
    parser.add_argument('--data_path_eval',            type=str,   help='path to the data for online evaluation', required=False, default = "work/project/PixelFormer/datasets/nyu_depth_v2/official_splits/test/")
    parser.add_argument('--gt_path_eval',              type=str,   help='path to the groundtruth data for online evaluation', required=False, default = "work/project/PixelFormer/datasets/datasets/nyu_depth_v2/official_splits/test/")
    parser.add_argument('--filenames_file_eval',       type=str,   help='path to the filenames text file for online evaluation', required=False, default = "work/project/PixelFormer/data_splits/nyudepthv2_test_files_with_gt.txt")
    parser.add_argument('--extra',                     type=str,   help='path to the filenames text file for online evaluation', required=False, default = "p")

    if len(sys.argv) == 2:
        arg_filename_with_prefix = '@' + sys.argv[1]
        args = parser.parse_args([arg_filename_with_prefix])
    else:
        args = parser.parse_args()
    
    args.distributed = False

    return args

def print_args(args):
    for key, value in args.__dict__.items():
        print(f"[LOG] {key}: {value}")

def get_net_weight(model):
    total_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
    total_size_mb = total_size_bytes / (1024 ** 2)
    return round(total_size_mb,2)

def plot_depth_map(dm):
    MIN_DEPTH = 0.0
    MAX_DEPTH = min(np.max(dm.numpy()), np.percentile(dm, 99))

    dm = np.clip(dm, MIN_DEPTH, MAX_DEPTH)
    cmap = plt.cm.plasma_r

    return dm, cmap, MIN_DEPTH, MAX_DEPTH


def resize_keeping_aspect_ratio(img, base):
    """
    Resize the image to a defined length manteining its proportions
    Scaling the shortest side of the image to a fixed 'base' length'
    """

    if img.shape[0] <= img.shape[1]:
        basewidth = int(base)
        wpercent = (basewidth / float(img.shape[0]))
        hsize = int((float(img.shape[1]) * float(wpercent)))
        img = st.resize(img, (basewidth, hsize), anti_aliasing=False, preserve_range=True)
    else:
        baseheight = int(base)
        wpercent = (baseheight / float(img.shape[1]))
        wsize = int((float(img.shape[0]) * float(wpercent)))
        img = st.resize(img, (wsize, baseheight), anti_aliasing=False, preserve_range=True)

    return img


def compute_rmse(predictions, depths):
    valid_mask = depths > 0.0
    valid_predictions = predictions[valid_mask]
    valid_depths = depths[valid_mask]
    mse = (torch.pow((valid_predictions - valid_depths).abs(), 2)).mean()
    return torch.sqrt(mse)

def compute_accuracy(y_pred, y_true, thr=0.05):
    valid_mask = y_true > 0.0
    valid_pred = y_pred[valid_mask]
    valid_true = y_true[valid_mask]
    correct = torch.max((valid_true / valid_pred), (valid_pred / valid_true)) < (1 + thr)
    return 100 * torch.mean(correct.float())

def print_model(model, device, save_model_root, input_shape):
    info = summary(model, torch.ones((1, input_shape[0], input_shape[1], input_shape[2])).to(device))
    info.to_csv(save_model_root + 'model_summary.csv')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def save_checkpoint(model, name, path_save_model):
    """
    Saves a model
    """
    if '_best' in name:
        folder = name.split("_best")[0]
    elif '_checkpoint' in name:
        folder = name.split("_checkpoint")[0]
    if not os.path.isdir(path_save_model):
        os.makedirs(path_save_model, exist_ok=True)
    torch.save(model.state_dict(), path_save_model + name)


def save_history(history, filepath):
    tmp_file = open(filepath + '.pkl', "wb")
    pickle.dump(history, tmp_file)
    tmp_file.close()


def save_csv_history(model_name, path):
    objects = []
    with (open(path + model_name + '_history.pkl', "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break
    df = pd.DataFrame(objects)
    df.to_csv(path + model_name + '_history.csv', header=False, index=False, sep=" ")


def load_pretrained_model(model, path_weigths, device, do_pretrained):
    model_name = model.__class__.__name__

    if do_pretrained:
        # print("\nloading checkpoint for entire {}..\n".format(model_name))
        model_dict = torch.load(path_weigths, map_location=torch.device(device))
        model.load_state_dict(model_dict, strict=False)
        # print("checkpoint loaded\n")

    return model, model_name

# def load_pretrained_encoder_only(target_model, pretrained_path, device):
#     """
#     Load only encoder weights from a complete pretrained model (encoder + decoder)
#     into a new model, ignoring decoder weights and handling different attention modules.
    
#     Args:
#         target_model: The new model to load encoder weights into
#         pretrained_path: Path to the complete pretrained model weights
#         device: Device to load the weights on
        
#     Returns:
#         model: The model with loaded encoder weights
#         loaded_layers: Dictionary with statistics about loaded layers
#     """
#     # Load complete pretrained weights
#     pretrained_weights = torch.load(pretrained_path, map_location=device)
    
#     if 'state_dict' in pretrained_weights:
#         pretrained_weights = pretrained_weights['state_dict']
    
#     # Get target model state dict
#     target_state = target_model.state_dict()
    
#     # Statistics tracking
#     stats = {
#         'encoder_loaded': 0,
#         'encoder_skipped': 0,
#         'decoder_skipped': 0,
#         'attention_skipped': 0
#     }
    
#     # Track which layers were not loaded
#     unmatched_keys = []
    
#     for name, param in pretrained_weights.items():
#         # Skip decoder layers completely
#         if 'decoder' in name:
#             stats['decoder_skipped'] += 1
#             continue
            
#         # Process only encoder layers
#         if 'encoder' in name:
#             # Skip attention layers which might be different
#             if 'attn' in name or 'attention' in name or 'transformer' in name:
#                 stats['attention_skipped'] += 1
#                 continue
            
#             try:
#                 # Clean the layer name to match target model
#                 target_name = name
                
#                 if target_name in target_state and target_state[target_name].shape == param.shape:
#                     target_state[target_name].copy_(param)
#                     stats['encoder_loaded'] += 1
#                 else:
#                     stats['encoder_skipped'] += 1
#                     unmatched_keys.append(name)
#             except Exception as e:
#                 print(f"Error loading layer {name}: {str(e)}")
#                 unmatched_keys.append(name)
    
#     # Load the updated state dict
#     target_model.load_state_dict(target_state, strict=False)
    
#     # Print statistics
#     print(f"[Pretrained W] Weight Loading Statistics:")
#     print(f"[Pretrained W] Successfully loaded {stats['encoder_loaded']} encoder layers")
#     print(f"[Pretrained W] Skipped {stats['encoder_skipped']} incompatible encoder layers")
#     print(f"[Pretrained W] Skipped {stats['attention_skipped']} attention layers")
#     print(f"[Pretrained W] Skipped {stats['decoder_skipped']} decoder layers")
    
#     if unmatched_keys:
#         print("[Pretrained W] Details of unmatched encoder layers:")
#         for key in unmatched_keys[:5]:  # Show first 5 unmatched keys
#             print(f" {key}")
#         if len(unmatched_keys) > 5:
#             print(f"... and {len(unmatched_keys) - 5} more")
            
#     return target_model, stats

def load_pretrained_encoder_only(target_model, pretrained_path, device):
    """
    Load only encoder weights from a complete pretrained model (encoder + decoder)
    into a new model, ignoring decoder weights and handling different attention modules.
    
    Args:
        target_model: The new model to load encoder weights into
        pretrained_path: Path to the complete pretrained model weights
        device: Device to load the weights on
        
    Returns:
        model: The model with loaded encoder weights
        loaded_layers: Dictionary with statistics about loaded layers
    """
    # Load complete pretrained weights
    pretrained_weights = torch.load(pretrained_path, map_location=device)
    
    if 'state_dict' in pretrained_weights:
        pretrained_weights = pretrained_weights['state_dict']
    
    # Get target model state dict
    target_state = target_model.state_dict()
    
    # Statistics tracking
    stats = {
        'encoder_loaded': 0,
        'encoder_skipped': 0,
        'decoder_skipped': 0,
        'attention_skipped': 0
    }
    
    # Track which layers were not loaded
    unmatched_keys = []
    
    for name, param in pretrained_weights.items():
        try:
            target_name = name
            target_state[target_name].copy_(param)
        except Exception as e:
            continue

    # Load the updated state dict
    target_model.load_state_dict(target_state, strict=False)
            
    return target_model, stats

def plot_graph(f, g, f_label, g_label, title, path):
    epochs = range(0, len(f))
    plt.plot(epochs, f, 'b', label=f_label)
    plt.plot(epochs, g, 'orange', label=g_label)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid('on', color='#cfcfcf')
    plt.tight_layout()
    plt.savefig(path + title + '.pdf')
    plt.close()

def plot_history(history, path):
    plot_graph(history['train_loss'], history['val_loss'], 'Train Loss', 'Val. Loss', 'TrainVal_loss', path)
    plot_graph(history['train_acc'], history['val_acc'], 'Train Acc.', 'Val. Acc.', 'TrainVal_acc', path)

def plot_loss_parts(history, path, title):
    l_mae_list = history['l_mae']
    l_norm_list = history['l_norm']
    l_grad_list = history['l_grad']
    l_ssim_list = history['l_ssim']
    epochs = range(0, len(l_mae_list))
    plt.plot(epochs, l_mae_list, 'r', label='l_mae')
    plt.plot(epochs, l_norm_list, 'g', label='l_norm')
    plt.plot(epochs, l_grad_list, 'b', label='l_grad')
    plt.plot(epochs, l_ssim_list, 'orange', label='l_ssim')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.grid('on', color='#cfcfcf')
    plt.legend()
    plt.tight_layout()
    plt.savefig(path + title + '.pdf')
    plt.close()

def print_img(dts_type, dataset, label, save_model_root, index=None, quantity=1, print_info_aug=False):
    for i in range(quantity):
        img, depth = dataset.__getitem__(index, print_info_aug)
        
        if dts_type == "nyu":
            fig = plt.figure(figsize=(15, 3))
        else:
            fig = plt.figure(figsize=(30, 3))
            
        plt.subplot(1, 3, 1)
        plt.title('Input image')
        img_normalized = (img - img.min()) / (img.max() - img.min())
        plt.imshow(torch.moveaxis(img_normalized, 0, -1).numpy(), cmap="grey")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title('Grayscale DepthMap')
        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min())
        plt.imshow(torch.moveaxis(depth_normalized, 0, -1), cmap='gray')
        plt.colorbar()
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title('Colored DepthMap')
        depth, cmap_dm, vmin, vmax = plot_depth_map(depth)
        colorized_depth_normalized = (depth - depth.min()) / (depth.max() - depth.min())
        plt.imshow(torch.moveaxis(depth, 0, -1), cmap=cmap_dm)
        plt.colorbar()
        plt.axis('off')
        
        save_path = save_model_root + 'example&augment_img/'
        if not os.path.exists(save_model_root):
            os.mkdir(save_model_root)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        plt.tight_layout()
        plt.savefig(save_path + 'img_' + str(i) + '_' + label + '.pdf')
        plt.close(fig=fig)

def save_prediction_examples(model, dataset, device, indices, save_path, ep):
    """
    Shows prediction example
    """
    fig = plt.figure(figsize=(20, 3)) # 20 NYU # 40 KITTI
    for i, index in zip(range(len(indices)), indices):
        img, depth = dataset.__getitem__(index)
        img = np.expand_dims(img, axis=0)
        # Predict
        model.eval()
        with torch.no_grad():
            pred = model(torch.from_numpy(img).to(device))
            # Build plot
            _, cmap_dm, vmin, vmax = plot_depth_map(depth)
            plt.subplot(1, len(indices), i+1)
            plt.imshow(np.squeeze(pred.cpu()), cmap=cmap_dm, vmin=vmin, vmax=vmax)
            cbar = plt.colorbar()
            cbar.ax.set_xlabel('cm', size=13, rotation=0)
            if False:
                plt.axis('off')

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    plt.tight_layout()
    plt.savefig(save_path + 'img_ep_' + str(ep) + '.pdf')
    plt.close(fig=fig)


def save_best_worst(list_type, type, model, dataset, device, save_model_root):
    save_path = save_model_root + type + '_predictions/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i in range(len(list_type)):
        index_image = list_type[i][0]
        rmse_value = list_type[i][1]

        img, depth = dataset.__getitem__(index=index_image)

        fig = plt.figure(figsize=(18, 3)) # 18 NYU # 40 KITTI
        plt.subplot(1, 4, 1)
        plt.title(f'Original image {index_image}')
        plt.imshow(torch.moveaxis(img, 0, -1), cmap='gray', vmin=0.0, vmax=1.0)
        plt.axis('off')

        plt.subplot(1, 4, 2)
        plt.title('Ground Truth')
        depth, cmap_dm, vmin, vmax = plot_depth_map(depth)
        plt.imshow(torch.moveaxis(depth, 0, -1), cmap=cmap_dm, vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.axis('off')

        # Predict
        model.eval()
        with torch.no_grad():
            pred = model(torch.unsqueeze(img, dim=0).to(device))

        plt.subplot(1, 4, 3)
        plt.title('Predicted DepthMap')
        pred, cmap_dm, _, _ = plot_depth_map(torch.squeeze(pred.cpu(), dim=0))
        plt.imshow(torch.moveaxis(pred, 0, -1), cmap=cmap_dm, vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.axis('off')

        plt.subplot(1, 4, 4)
        plt.title('Disparity Map, RMSE = {:.2f}'.format(rmse_value))
        intensity_img = torch.moveaxis(torch.abs(depth - pred), 0, -1)
        plt.imshow(intensity_img, cmap=plt.cm.magma, vmin=0)
        plt.colorbar()
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(save_path + '/seq_' + str(i) + '.pdf')
        plt.close(fig=fig)

def compute_MeanVar(dataset):
    r_mean, g_mean, b_mean = [], [], []
    r_var, g_var, b_var = [], [], []
    for i in range(dataset.__len__()):
        img, _ = dataset.__getitem__(index=i)
        r = np.array(img[0, :, :])
        g = np.array(img[1, :, :])
        b = np.array(img[2, :, :])

        r_mean.append(np.mean(r))
        g_mean.append(np.mean(g))
        b_mean.append(np.mean(b))

        r_var.append(np.var(r))
        g_var.append(np.var(g))
        b_var.append(np.var(b))

    print(f"The MEAN are: R - {np.mean(r_mean)}, G - {np.mean(g_mean)}, B - {np.mean(b_mean)}\n"
          f"The VAR are: R - {np.mean(r_var)}, G - {np.mean(g_var)}, B - {np.mean(b_var)}")


def compute_MeanImg(dataset, save_model_root):
    r, g, b = [], [], []
    for i in range(dataset.__len__()):
        img, _ = dataset.__getitem__(index=i)
        r.append(np.array(img[0, :, :]))
        g.append(np.array(img[1, :, :]))
        b.append(np.array(img[2, :, :]))

    r_sum = np.mean(np.stack(r, axis=-1), axis=-1)
    g_sum = np.mean(np.stack(g, axis=-1), axis=-1)
    b_sum = np.mean(np.stack(b, axis=-1), axis=-1)
    mean_img = torch.moveaxis(torch.from_numpy(np.stack([r_sum, g_sum, b_sum], axis=-1)), -1, 0)
    np.save(save_model_root + 'nyu_Mimg.npy', mean_img)

    print("Process Completed")

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, val_range, window_size=11, window=None, size_average=True, full=False):
    L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs

    return ret

class Sobel(nn.Module):
    def __init__(self):
        super(Sobel, self).__init__()
        self.edge_conv = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, bias=False)
        edge_kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        edge_ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        edge_k = np.stack((edge_kx, edge_ky))

        edge_k = torch.from_numpy(edge_k).float().view(2, 1, 3, 3)
        self.edge_conv.weight = nn.Parameter(edge_k)

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = self.edge_conv(x)
        out = out.contiguous().view(-1, 2, x.size(2), x.size(3))

        return out

class balanced_loss_function(nn.Module):
    def __init__(self, device):
        super(balanced_loss_function, self).__init__()
        self.cos = nn.CosineSimilarity(dim=1, eps=0)
        self.get_gradient = Sobel().to(device)
        self.device = device

    def forward(self, output, depth):
        with torch.no_grad():
            ones = torch.ones(depth.size(0), 1, depth.size(2), depth.size(3)).float().to(self.device)

        depth_grad = self.get_gradient(depth)
        output_grad = self.get_gradient(output)

        depth_grad_dx = depth_grad[:, 0, :, :].contiguous().view_as(depth)
        depth_grad_dy = depth_grad[:, 1, :, :].contiguous().view_as(depth)
        output_grad_dx = output_grad[:, 0, :, :].contiguous().view_as(depth)
        output_grad_dy = output_grad[:, 1, :, :].contiguous().view_as(depth)

        depth_normal = torch.cat((-depth_grad_dx, -depth_grad_dy, ones), 1)
        output_normal = torch.cat((-output_grad_dx, -output_grad_dy, ones), 1)

        loss_depth = torch.abs(output - depth).mean()
        loss_dx = torch.abs(output_grad_dx - depth_grad_dx).mean()
        loss_dy = torch.abs(output_grad_dy - depth_grad_dy).mean()
        loss_normal = 100 * torch.abs(1 - self.cos(output_normal, depth_normal)).mean()

        loss_ssim = (1 - ssim(output, depth, val_range=1000.0)) * 100

        loss_grad = (loss_dx + loss_dy) / 2

        return loss_depth, loss_ssim, loss_normal, loss_grad

def log10(x):
    return torch.log(x) / math.log(10)

class Result(object):
    def __init__(self,dts_type):
        self.dts_type = dts_type
        self.irmse, self.imae = 0, 0
        self.mse, self.rmse, self.mae = 0, 0, 0
        self.absrel, self.lg10 = 0, 0
        self.delta1, self.delta2, self.delta3 = 0, 0, 0

    def set_to_worst(self):
        self.irmse, self.imae = np.inf, np.inf
        self.mse, self.rmse, self.mae = np.inf, np.inf, np.inf
        self.absrel, self.lg10 = np.inf, np.inf
        self.delta1, self.delta2, self.delta3 = 0, 0, 0

    def update(self, irmse, imae, mse, rmse, mae, absrel, lg10, delta1, delta2, delta3):
        self.irmse, self.imae = irmse, imae
        self.mse, self.rmse, self.mae = mse, rmse, mae
        self.absrel, self.lg10 = absrel, lg10
        self.delta1, self.delta2, self.delta3 = delta1, delta2, delta3

    def evaluate(self, output, target):
        valid_mask = target > 0

        output = output[valid_mask]
        target = target[valid_mask]

        # if 'kitti' in self.dts_type:
        #     output = output[2080:] # remove first 13pixels lines
        #     target = target[2080:]

        abs_diff = (output - target).abs()

        self.mse = float((torch.pow(abs_diff, 2)).mean())
        self.rmse = math.sqrt(self.mse)
        self.mae = float(abs_diff.mean())
        self.lg10 = float((log10(output) - log10(target)).abs().mean())
        self.absrel = float((abs_diff / target).mean())

        maxRatio = torch.max(output / target, target / output)
        self.delta1 = float((maxRatio < 1.25).float().mean())
        self.delta2 = float((maxRatio < 1.25 ** 2).float().mean())
        self.delta3 = float((maxRatio < 1.25 ** 3).float().mean())

        inv_output = 1 / output
        inv_target = 1 / target
        abs_inv_diff = (inv_output - inv_target).abs()
        self.irmse = math.sqrt((torch.pow(abs_inv_diff, 2)).mean())
        self.imae = float(abs_inv_diff.mean())

class AverageMeter(object):
    def __init__(self,dts_type):
        self.dts_type = dts_type
        self.reset()

    def reset(self):
        self.count = 0.0
        self.sum_irmse, self.sum_imae = 0, 0
        self.sum_mse, self.sum_rmse, self.sum_mae = 0, 0, 0
        self.sum_absrel, self.sum_lg10 = 0, 0
        self.sum_delta1, self.sum_delta2, self.sum_delta3 = 0, 0, 0

    def update(self, result, n=1):
        self.count += n

        self.sum_irmse += n * result.irmse
        self.sum_imae += n * result.imae
        self.sum_mse += n * result.mse
        self.sum_rmse += n * result.rmse
        self.sum_mae += n * result.mae
        self.sum_absrel += n * result.absrel
        self.sum_lg10 += n * result.lg10
        self.sum_delta1 += n * result.delta1
        self.sum_delta2 += n * result.delta2
        self.sum_delta3 += n * result.delta3

    def average(self):
        avg = Result(self.dts_type)
        avg.update(
            self.sum_irmse / self.count, self.sum_imae / self.count,
            self.sum_mse / self.count, self.sum_rmse / self.count, self.sum_mae / self.count,
            self.sum_absrel / self.count, self.sum_lg10 / self.count,
            self.sum_delta1 / self.count, self.sum_delta2 / self.count, self.sum_delta3 / self.count)
        return avg

def compute_evaluation(test_dataloader, model, model_type, path_save_csv_results,dts_type,quant):
    best_worst_dict = {}
    result = Result(dts_type)
    result.set_to_worst()
    average_meter = AverageMeter(dts_type)
    test_images = {
        "images": [],
        "gt_depths": [],
        "pred_depths": []
    }
    if not quant:
        model.eval()  # switch to evaluate mode
    
    for i, (inputs, depths) in enumerate(tqdm(test_dataloader, desc="Computing evaluation metrics")):
        # inputs, depths = inputs.cuda(), depths.cuda()
        
        # compute output
        with torch.no_grad():
            predictions = model(inputs)
        
        if i == 0 or i == 10 or i == 100:
            test_images["images"].append(inputs)
            test_images["gt_depths"].append(depths)
            test_images["pred_depths"].append(predictions)

        result.evaluate(predictions, depths)
        average_meter.update(result)  # (result, inputs.size(0))
        
        # best_worst_dict[i] = result.rmse
        if math.isnan(result.rmse):
            best_worst_dict[i] = 0
        else:
            best_worst_dict[i] = result.rmse

    avg = average_meter.average()
    save_test_images(
        save_path = path_save_csv_results,
        test_images = test_images 
    )

    start_time = perf_counter()
    for _, (inputs, _) in enumerate(tqdm(test_dataloader, desc="Inference speed test")):
        inputs = inputs#.cuda()
        with torch.no_grad():
            predictions = model(inputs)
    # torch.cuda.synchronize()
    end_time = perf_counter()
    elapsed_time = end_time - start_time

    return best_worst_dict, avg, elapsed_time

def save_test_images(save_path,test_images,bar=False):
    dict_zip = zip(test_images["images"], test_images["gt_depths"], test_images["pred_depths"])
    save_cnt = 1
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for image, gt_depth, pred_depth in dict_zip:
        image_np = image.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
        gt_depth_resized = F.interpolate(gt_depth, size=(192, 256), mode="bilinear", align_corners=False)
        pred_depth_resized = F.interpolate(pred_depth, size=(192, 256), mode="bilinear", align_corners=False)

        gt_depth_np = gt_depth_resized.detach().cpu().squeeze(0).squeeze(0).numpy()
        pred_depth_np = pred_depth_resized.detach().cpu().squeeze(0).squeeze(0).numpy()
        
        image_np = image.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
        gt_depth_np = np.squeeze(gt_depth_np)
        pred_depth_np = np.squeeze(pred_depth_np)
        gt_depth_np = (gt_depth_np - gt_depth_np.min()) / (gt_depth_np.max() - gt_depth_np.min())
        pred_depth_np = (pred_depth_np - pred_depth_np.min()) / (pred_depth_np.max() - pred_depth_np.min())

        gt_min, gt_max = gt_depth_np.min(), gt_depth_np.max()
        pred_min, pred_max = pred_depth_np.min(), pred_depth_np.max()

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(image_np)
        axes[0].set_title("Image")
        axes[0].axis("off")

        im1 = axes[1].imshow(gt_depth_np, cmap="plasma")
        axes[1].set_title("Ground Truth Depth")
        axes[1].axis("off")
        if bar:
            cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.02)
            cbar1.set_label("Depth Value")
            cbar1.set_ticks([0, 1])
            cbar1.ax.set_yticklabels([f'{gt_min:.2f}', f'{gt_max:.2f}'])

        im2 = axes[2].imshow(pred_depth_np, cmap="plasma")
        axes[2].set_title("Predicted Depth")
        axes[2].axis("off")
        if bar:
            cbar2 = fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.02)
            cbar2.set_label("Depth Value")
            cbar2.set_ticks([0, 1])
            cbar2.ax.set_yticklabels([f'{pred_min:.2f}', f'{pred_max:.2f}'])

        plt.tight_layout()
        plt.savefig(save_path + "/" + str(save_cnt) + "_test_image.png")
        save_cnt *= 10

def capitalize_first_last(s):
    if len(s) < 2:
        return s.upper()
    else:
        return s[0].upper() + s[1:-1] + s[-1].upper()

def write_results(args, eval_measures, elapsed_time):
    if "_" in args.network_type:
        opt_type = args.network_type.split("_")[0]
        opt_name = capitalize_first_last(opt_type) if "h" in opt_type else opt_type.capitalize()
        net_name = opt_name + " METER"
    else:
        net_name = "METER"

    if args.extra:
        net_name += "_" + args.extra

    rmse = eval_measures[0]
    abs_rel = eval_measures[1]
    delta_1 = eval_measures[2]
    delta_2 = eval_measures[3]
    delta_3 = eval_measures[4]

    results_line = (
        f"{args.dts_type} - "
        f"{net_name} & "
        f"{rmse:.3f} & "
        f"{abs_rel:.3f} & "
        f"{delta_1:.3f} & "
        f"{delta_2:.3f} & "
        f"{delta_3:.3f} & "
        f"{elapsed_time:.2f} \\\\"
    )

    with open("work/project/results.txt", "a") as file:
        file.write(results_line + "\n")