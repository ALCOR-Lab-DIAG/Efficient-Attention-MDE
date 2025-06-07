import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pytz
import shutil
import sys
import torch
import torch.nn as nn
import smtplib

from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from torchvision import transforms

eval_metrics = ['silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'd1', 'd2', 'd3']

inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)

def send_email(network, version, optimization, where, error=False, error_msg=""):
    rome_tz = pytz.timezone("Europe/Rome")
    current_time = datetime.now(rome_tz)
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
        body = f"The train is correctly ended on {print_date}. The configuration was: \n\t- Network: {network} \n\t- Size: {version} \n\t- Optimization: {optimization} \n\t- Optimized module: {where}"
    else:
        subject = '!!! [RTX5000] Training ERROR !!!'
        body = f"The train of {network} {version} {optimization}-{where} is UNEXPLECTEDLY ended on {print_date}. The error was: \n\n {error_msg}"

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

def hardware_check(mode):
    if mode == "train" or mode == "analysis":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = "cpu"
    
    print(f"[LOG] Current device: {device}")
    if 'cuda' in device:
        prop = torch.cuda.get_device_properties(device)
        print(f"[LOG] Device name: {prop.name}")
        print(f"[LOG] Device memory: {prop.total_memory/ (1024 ** 2)} MB")
        print(f"[LOG] Device processors: {prop.multi_processor_count}")
        
    return device

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg

def init_arg_parser():
    parser = argparse.ArgumentParser(description='PixelFormer PyTorch implementation.', fromfile_prefix_chars='@')
    parser.convert_arg_line_to_args = convert_arg_line_to_args

    if "train" in sys.argv[1]:
        parser.add_argument('--mode',                      type=str,   help='train or test', default='train')
        parser.add_argument('--model_name',                type=str,   help='model name', default='pixelformer')
        parser.add_argument('--encoder',                   type=str,   help='type of encoder, base07, large07', default='large07')
        parser.add_argument('--pretrain',                  type=str,   help='path of pretrained encoder', default=None)

        # Dataset
        parser.add_argument('--dataset',                   type=str,   help='dataset to train on, kitti or nyu', default='nyu')
        parser.add_argument('--data_path',                 type=str,   help='path to the data',  default="work/project/PixelFormer/datasets/nyu_depth_v2/official_splits/test/")
        parser.add_argument('--gt_path',                   type=str,   help='path to the groundtruth data', default="work/project/PixelFormer/datasets/nyu_depth_v2/official_splits/test/")
        parser.add_argument('--filenames_file',            type=str,   help='path to the filenames text file', default="work/project/PixelFormer/data_splits/nyudepthv2_test_files_with_gt.txt")
        parser.add_argument('--input_height',              type=int,   help='input height', default=480)
        parser.add_argument('--input_width',               type=int,   help='input width',  default=640)
        parser.add_argument('--max_depth',                 type=float, help='maximum depth in estimation', default=10)

        # Log and save
        parser.add_argument('--log_directory',             type=str,   help='directory to save checkpoints and summaries', default='')
        parser.add_argument('--checkpoint_path',           type=str,   help='path to a checkpoint to load', default='')
        parser.add_argument('--log_freq',                  type=int,   help='Logging frequency in global steps', default=100)
        parser.add_argument('--save_freq',                 type=int,   help='Checkpoint saving frequency in global steps', default=5000)

        # Training
        parser.add_argument('--weight_decay',              type=float, help='weight decay factor for optimization', default=1e-2)
        parser.add_argument('--retrain',                               help='if used with checkpoint_path, will restart training from step zero', action='store_true')
        parser.add_argument('--adam_eps',                  type=float, help='epsilon in Adam optimizer', default=1e-6)
        parser.add_argument('--batch_size',                type=int,   help='batch size', default=4)
        parser.add_argument('--num_epochs',                type=int,   help='number of epochs', default=50)
        parser.add_argument('--learning_rate',             type=float, help='initial learning rate', default=1e-4)
        parser.add_argument('--end_learning_rate',         type=float, help='end learning rate', default=-1)
        parser.add_argument('--variance_focus',            type=float, help='lambda in paper: [0, 1], higher value more focus on minimizing variance of error', default=0.85)
        parser.add_argument('--opt_type',                  type=str,   help='optimization type', default=None),
        parser.add_argument('--opt_where',                 type=str,   help='where optimization is applied', default=None),
        
        # Preprocessing
        parser.add_argument('--do_random_rotate',                      help='if set, will perform random rotation for augmentation', action='store_true')
        parser.add_argument('--degree',                    type=float, help='random rotation maximum degree', default=2.5)
        parser.add_argument('--do_kb_crop',                            help='if set, crop input images as kitti benchmark images', action='store_true')
        parser.add_argument('--use_right',                             help='if set, will randomly use right images when train on KITTI', action='store_true')

        # Multi-gpu training
        parser.add_argument('--num_threads',               type=int,   help='number of threads to use for data loading', default=1)
        parser.add_argument('--world_size',                type=int,   help='number of nodes for distributed training', default=1)
        parser.add_argument('--rank',                      type=int,   help='node rank for distributed training', default=0)
        parser.add_argument('--dist_url',                  type=str,   help='url used to set up distributed training', default=0) #'tcp://127.0.0.1:1234'
        parser.add_argument('--dist_backend',              type=str,   help='distributed backend', default='nccl')
        parser.add_argument('--gpu',                       type=int,   help='GPU id to use.', default=None)
        parser.add_argument('--multiprocessing_distributed',           help='Use multi-processing distributed training to launch '
                                                                            'N processes per node, which has N GPUs. This is the '
                                                                            'fastest way to use PyTorch for either single node or '
                                                                            'multi node data parallel training', action='store_true')
        # Online eval
        parser.add_argument('--do_online_eval',                        help='if set, perform online eval in every eval_freq steps', action='store_true')
        parser.add_argument('--eval_batch_size',                       help='evaluation dataset batch size', default=1)
        parser.add_argument('--data_path_eval',            type=str,   help='path to the data for online evaluation', required=False, default = "work/project/PixelFormer/datasets/nyu_depth_v2/official_splits/test/")
        parser.add_argument('--gt_path_eval',              type=str,   help='path to the groundtruth data for online evaluation', required=False, default = "work/project/PixelFormer/datasets/datasets/nyu_depth_v2/official_splits/test/")
        parser.add_argument('--filenames_file_eval',       type=str,   help='path to the filenames text file for online evaluation', required=False, default = "work/project/PixelFormer/data_splits/nyudepthv2_test_files_with_gt.txt")
        parser.add_argument('--min_depth_eval',            type=float, help='minimum depth for evaluation', default=1e-3)
        parser.add_argument('--max_depth_eval',            type=float, help='maximum depth for evaluation', default=80)
        parser.add_argument('--eigen_crop',                            help='if set, crops according to Eigen NIPS14', action='store_true')
        parser.add_argument('--garg_crop',                             help='if set, crops according to Garg  ECCV16', action='store_true')
        parser.add_argument('--eval_freq',                 type=int,   help='Online evaluation frequency in global steps', default=500)
        parser.add_argument('--eval_summary_directory',    type=str,   help='output directory for eval summary,'
                                                                            'if empty outputs to checkpoint folder', default='')
    else:
        # General arguments
        parser.add_argument('--mode', type=str,   help='train or test', default='train')
        parser.add_argument('--model_name', type=str, help='model name', default='newcrfs_kittieigen')
        parser.add_argument('--encoder', type=str, help='type of encoder, base07, large07', default='large07')
        parser.add_argument('--dataset', type=str, help='dataset to train on', default='kitti')
        parser.add_argument('--save_path', type=str,   help='directory to save results', default='')
        parser.add_argument('--input_height', type=int, help='input height', default=352)
        parser.add_argument('--input_width', type=int, help='input width', default=1216)
        parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=80)
        parser.add_argument('--do_kb_crop', help='if set, crop input images as KITTI benchmark images', action='store_true')

        # Evaluation-specific arguments
        parser.add_argument('--data_path_eval', type=str, help='path to evaluation data', required=True)
        parser.add_argument('--gt_path_eval', type=str, help='path to ground truth data for evaluation', required=True)
        parser.add_argument('--filenames_file_eval', type=str, help='path to the filenames text file for evaluation', required=True)
        parser.add_argument('--min_depth_eval', type=float, help='minimum depth for evaluation', default=1e-3)
        parser.add_argument('--max_depth_eval', type=float, help='maximum depth for evaluation', default=80)
        parser.add_argument('--garg_crop', help='if set, apply Garg crop for evaluation', action='store_true')
        parser.add_argument('--eigen_crop', help='if set, apply Eigen crop for evaluation', action='store_true')

        # Checkpoint and optimization arguments
        parser.add_argument('--checkpoint_path', type=str, help='path to a specific checkpoint to load', default='model_zoo/model_kittieigen.ckpt')
        parser.add_argument('--opt_type', type=str, help='optimization type', default='')
        parser.add_argument('--opt_where', type=str, help='where optimization is applied', default='')
        parser.add_argument('--extra',                     type=str,   help='path to the filenames text file for online evaluation', required=False, default = "p")

    if sys.argv.__len__() == 2:
        arg_filename_with_prefix = '@' + sys.argv[1]
        args = parser.parse_args([arg_filename_with_prefix])
    else:
        args = parser.parse_args()
    
    args.distributed = False

    return args

def test_arg_parser():
    parser = argparse.ArgumentParser(description='PixelFormer PyTorch implementation.', fromfile_prefix_chars='@')
    parser.convert_arg_line_to_args = convert_arg_line_to_args

    parser.add_argument('--model_name', type=str, help='model name', default='pixelformer')
    parser.add_argument('--encoder', type=str, help='type of encoder, base07, large07', default='large07')
    parser.add_argument('--data_path', type=str, help='path to the data', required=True)
    parser.add_argument('--filenames_file', type=str, help='path to the filenames text file', required=True)
    parser.add_argument('--input_height', type=int, help='input height', default=480)
    parser.add_argument('--input_width', type=int, help='input width', default=640)
    parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=10)
    parser.add_argument('--checkpoint_path', type=str, help='path to a specific checkpoint to load', default='')
    parser.add_argument('--dataset', type=str, help='dataset to train on', default='nyu')
    parser.add_argument('--do_kb_crop', help='if set, crop input images as kitti benchmark images', action='store_true')
    parser.add_argument('--save_viz', help='if set, save visulization of the outputs', action='store_true')
    parser.add_argument('--save_results', type=str, help='path to save the results', required=True)
    parser.add_argument('--opt_type',                  type=str,   help='optimization type', default=None)
    parser.add_argument('--opt_where',                  type=str,   help='where optimization is applied', default=None),
    parser.add_argument('--mode',                      type=str,   help='train or test', default='train')
    
    if sys.argv.__len__() == 2:
        arg_filename_with_prefix = '@' + sys.argv[1]
        args = parser.parse_args([arg_filename_with_prefix])
    else:
        args = parser.parse_args()

    args.distributed = False
    return args

def print_args(args):
    for key, value in args.__dict__.items():
        print(f"[LOG] {key}: {value}")

def print_examples(dataloader):
    # sample_batched = next(iter(dataloader.data))
    sample_batched = next(iter(dataloader.data))
    image_tensor = sample_batched['image'][0]
    depth_tensor = sample_batched['depth'][0]
        
    image_np = image_tensor.permute(1, 2, 0).numpy()
    image_np = np.clip(image_np, 0, 1)
    depth_np = depth_tensor.permute(1, 2, 0).numpy()
    depth_np = np.clip(depth_np, 0, 1)
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    ax[0].imshow(image_np)
    ax[0].set_title("Image sample with data augmentation")
    ax[0].axis('off')

    ax[1].imshow(depth_np)
    ax[1].set_title("Depth ground truth")
    ax[1].axis('off')

    plt.savefig(
        "work/data/ssd_results/random_data_sample.pdf",
        format = "pdf", 
        bbox_inches = "tight"
    )

def get_net_weight(model):
    total_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
    total_size_mb = total_size_bytes / (1024 ** 2)
    return round(total_size_mb,2)

def normalize_result(value, vmin=None, vmax=None):
    value = value.cpu().numpy()[0, :, :]

    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax

    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)
    else:
        value = value * 0.

    return np.expand_dims(value, 0)

def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()
    
    rms = (gt - pred) ** 2
    rms = np.sqrt(rms.mean())

    log_rms = (np.log(gt) - np.log(pred)) ** 2
    log_rms = np.sqrt(log_rms.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    return [silog, abs_rel, log10, rms, sq_rel, log_rms, d1, d2, d3]

class silog_loss(nn.Module):
    def __init__(self, variance_focus):
        super(silog_loss, self).__init__()
        self.variance_focus = variance_focus

    def forward(self, depth_est, depth_gt, mask):
        depth_gt = depth_gt.permute(0,3,1,2) ##########################################################################################################
        d = torch.log(depth_est[mask]) - torch.log(depth_gt[mask])
        return torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0

def flip_lr(image):
    """
    Flip image horizontally

    Parameters
    ----------
    image : torch.Tensor [B,3,H,W]
        Image to be flipped

    Returns
    -------
    image_flipped : torch.Tensor [B,3,H,W]
        Flipped image
    """
    assert image.dim() == 4, 'You need to provide a [B,C,H,W] image to flip'
    return torch.flip(image, [3])

def post_process_depth(depth, depth_flipped, method='mean'):
    """
    Post-process an inverse and flipped inverse depth map

    Parameters
    ----------
    inv_depth : torch.Tensor [B,1,H,W]
        Inverse depth map
    inv_depth_flipped : torch.Tensor [B,1,H,W]
        Inverse depth map produced from a flipped image
    method : str
        Method that will be used to fuse the inverse depth maps

    Returns
    -------
    inv_depth_pp : torch.Tensor [B,1,H,W]
        Post-processed inverse depth map
    """
    B, C, H, W = depth.shape
    inv_depth_hat = flip_lr(depth_flipped)
    inv_depth_fused = fuse_inv_depth(depth, inv_depth_hat, method=method)
    xs = torch.linspace(0., 1., W, device=depth.device,
                        dtype=depth.dtype).repeat(B, C, H, 1)
    mask = 1.0 - torch.clamp(20. * (xs - 0.05), 0., 1.)
    mask_hat = flip_lr(mask)
    return mask_hat * depth + mask * inv_depth_hat + \
           (1.0 - mask - mask_hat) * inv_depth_fused

def fuse_inv_depth(inv_depth, inv_depth_hat, method='mean'):
    """
    Fuse inverse depth and flipped inverse depth maps

    Parameters
    ----------
    inv_depth : torch.Tensor [B,1,H,W]
        Inverse depth map
    inv_depth_hat : torch.Tensor [B,1,H,W]
        Flipped inverse depth map produced from a flipped image
    method : str
        Method that will be used to fuse the inverse depth maps

    Returns
    -------
    fused_inv_depth : torch.Tensor [B,1,H,W]
        Fused inverse depth map
    """
    if method == 'mean':
        return 0.5 * (inv_depth + inv_depth_hat)
    elif method == 'max':
        return torch.max(inv_depth, inv_depth_hat)
    elif method == 'min':
        return torch.min(inv_depth, inv_depth_hat)
    else:
        raise ValueError('Unknown post-process method {}'.format(method))

def save_test_images(save_path,test_images,bar=False):
    dict_zip = zip(test_images["images"], test_images["gt_depths"], test_images["pred_depths"])
    save_cnt = 1
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for image, gt_depth, pred_depth in dict_zip:
        image_np = image.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
        gt_depth_np = gt_depth.detach().cpu().squeeze(0).squeeze(-1).numpy()
        pred_depth_np = pred_depth.detach().cpu().squeeze(0).squeeze(0).numpy()
        
        image_np = image.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
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

        plt.tight_layout()
        plt.savefig(save_path + str(save_cnt) + "_test_image.png")
        save_cnt *= 10

def capitalize_first_last(s):
    if len(s) < 2:
        return s.upper()
    else:
        return s[0].upper() + s[1:-1] + s[-1].upper()

def write_results(args, eval_measures, elapsed_time):
    if args.opt_type != "":
        opt_name = capitalize_first_last(args.opt_type) if "h" in args.opt_type else args.opt_type.capitalize()
        if args.opt_where == "enc":
            net_name = opt_name + "-Base PXF"
        elif args.opt_where == "dec":
            net_name = "Base-" + opt_name + " PXF"
        else:
            net_name = opt_name + " PXF"
    else:
        net_name = "PXF"

    if args.extra:
        net_name += "_" + args.extra

    rmse = eval_measures[3]
    abs_rel = eval_measures[1]
    delta_1 = eval_measures[6]
    delta_2 = eval_measures[7]
    delta_3 = eval_measures[8]

    results_line = (
        f"{args.dataset} - "
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