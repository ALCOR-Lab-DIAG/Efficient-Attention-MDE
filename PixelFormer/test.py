import torch
import torch.nn as nn
from torch.autograd import Variable

import os, sys, errno
import argparse
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import perf_counter
import warnings

from PixelFormer.utils import *

from PixelFormer.datasets.dataloader import *
from PixelFormer.networks.PixelFormer import *

def eval(args,model, dataloader_eval, post_process=False):
    eval_measures = torch.zeros(10)#.cuda(device=args.gpu)
    test_images = {
        "images": [],
        "gt_depths": [],
        "pred_depths": []
    }

    for idx, eval_sample_batched in enumerate(tqdm(dataloader_eval.data,desc="Computing evaluation metrics")):
        with torch.no_grad():
            image = torch.autograd.Variable(eval_sample_batched['image'])#.cuda(args.gpu, non_blocking=True))
            gt_depth = eval_sample_batched['depth']
            has_valid_depth = eval_sample_batched['has_valid_depth']
            if not has_valid_depth:
                # print('Invalid depth. continue.')
                continue

            pred_depth = model(image)
            
            if post_process:
                image_flipped = flip_lr(image)
                pred_depth_flipped = model(image_flipped)
                pred_depth = post_process_depth(pred_depth, pred_depth_flipped)

            if idx == 0 or idx == 10 or idx == 100:
                test_images["images"].append(image)
                test_images["gt_depths"].append(gt_depth)
                test_images["pred_depths"].append(pred_depth)

            pred_depth = pred_depth.cpu().numpy().squeeze()
            gt_depth = gt_depth.cpu().numpy().squeeze()
           
        if args.do_kb_crop:
            height, width = gt_depth.shape
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            pred_depth_uncropped = np.zeros((height, width), dtype=np.float32)
            pred_depth_uncropped[top_margin:top_margin + 352, left_margin:left_margin + 1216] = pred_depth
            pred_depth = pred_depth_uncropped

        pred_depth[pred_depth < args.min_depth_eval] = args.min_depth_eval
        pred_depth[pred_depth > args.max_depth_eval] = args.max_depth_eval
        pred_depth[np.isinf(pred_depth)] = args.max_depth_eval
        pred_depth[np.isnan(pred_depth)] = args.min_depth_eval

        valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)

        if args.garg_crop or args.eigen_crop:
            gt_height, gt_width = gt_depth.shape
            eval_mask = np.zeros(valid_mask.shape)

            if args.garg_crop:
                eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height), int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

            elif args.eigen_crop:
                if args.dataset == 'kitti':
                    eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height), int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
                elif args.dataset == 'nyu':
                    eval_mask[45:471, 41:601] = 1

            valid_mask = np.logical_and(valid_mask, eval_mask)

        measures = compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])

        eval_measures[:9] += torch.tensor(measures)#.cuda(device=args.gpu)
        eval_measures[9] += 1

    eval_measures_cpu = eval_measures.cpu()
    cnt = eval_measures_cpu[9].item()
    eval_measures_cpu /= cnt

    save_test_images(
        save_path = args.save_path,
        test_images = test_images 
    )
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        start_time = perf_counter()
        
        for _, eval_sample_batched in enumerate(tqdm(dataloader_eval.data,desc="Inference speed test")):
            with torch.no_grad():
                image = torch.autograd.Variable(eval_sample_batched['image'])#.cuda(args.gpu, non_blocking=True))
                pred_depth = model(image)

        # torch.cuda.synchronize()
        end_time = perf_counter()
        elapsed_time = end_time - start_time

    return eval_measures_cpu, elapsed_time

def test(args):
    formatter = TextFormatter()
    prep_str = formatter.format(
        f"Preparing testing of PixelFormer {args.encoder[:-2]}", 
        color = 'blue', 
        style = ['bold'], 
        separator = True
    )
    print(prep_str)

    torch.cuda.empty_cache()
    print(f"[LOG] Model will tested on {args.gpu} GPU")

    dataloader_eval = NewDataLoader(args, 'online_eval')

    print("[Data]",end=" ")
    data_str = formatter.format(
        f"Dataset {args.dataset.upper()} correctly instantiated \u2713", 
        color = 'green', 
        style = ['bold'], 
        separator = False
    )
    print(data_str)

    # Build the model
    model_name = "PixelFormer " + args.encoder[:-2]
    model_str = formatter.format(
        "Building " + model_name, 
        color = 'blue', 
        style = ['bold'], 
        separator = True
    )
    print(model_str)

    model = PixelFormer(
        version = args.encoder,
        inv_depth = False,
        max_depth = args.max_depth,
        pretrained = None,
        opt_type = args.opt_type,
        opt_where = args.opt_where
    )
    
    if args.extra == "q":
        model = torch.load(args.checkpoint_path, map_location='cpu')
    elif args.checkpoint_path != '':
        if os.path.isfile(args.checkpoint_path):
            checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint, strict=False)
            model.to('cpu')  # Sposta esplicitamente il modello su CPU
            del checkpoint
        else:
            print(f"No checkpoint found at {args.checkpoint_path}")

    model.eval()

    # Model characteristics
    print(f"[Model] Pretrained model builded and loaded!")
    print(f"[Model] PixelFormer version: {args.encoder[:-2]}")
    
    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print(f"[Model] Trainable parameters: {num_params:,}")

    model_weight = get_net_weight(model)
    print(f"[Model] Network memory weight: {model_weight} MB")
    
    if args.opt_type != "":
        opt_name = args.opt_type.capitalize()
        where = args.opt_where
        full_str = f"Starting {opt_name} {where} PixelFormer {args.encoder[:-2]} testing"
    else:
        opt_name = ""
        where = ""
        full_str = f"Starting PixelFormer {args.encoder[:-2]} testing"

    # Train loop
    train_str = formatter.format(
        full_str,
        color = 'blue', 
        style = ['bold'], 
        separator = True
    )
    print(train_str)

    with torch.no_grad():
        eval_measures, elapsed_time = eval(args, model, dataloader_eval, post_process=True)
    
    res_str = formatter.format(
        f"Results", 
        color = 'blue', 
        style = ['bold'], 
        separator = True
    )
    print(res_str)
    write_results(args,eval_measures,elapsed_time)
        
    print(f"[LOG] Network: PixelFormer")
    print(f"[LOG] Size: {args.encoder[:-2]}")
    print(f"[LOG] Trainable parameters: {num_params:,}")
    print(f"[LOG] Network memory weight: {model_weight} MB")
    print(f"[LOG] RMSE: {eval_measures[3]:.3f}")
    print(f"[LOG] AbsREL: {eval_measures[1]:.3f}")
    print(f"[LOG] Delta1: {eval_measures[6]:.3f}")
    print(f"[LOG] Delta2: {eval_measures[7]:.3f}")
    print(f"[LOG] Delta3: {eval_measures[8]:.3f}")
    print(f"[LOG] Test time: {elapsed_time:.2f}")

if __name__ == '__main__':
    test(args)