import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.utils as utils
import time
import warnings

from tensorboardX import SummaryWriter
from tqdm import tqdm

from PixelFormer.datasets.dataloader import *
from PixelFormer.globals import *
from PixelFormer.networks.PixelFormer import *
from PixelFormer.networks.moh_swin_transformer import SwinMoHWindowAttention
from PixelFormer.networks.moh_SAM import SAMMoHWindowAttention
from PixelFormer.utils import *

def online_eval(args, model, dataloader_eval, gpu, ngpus, post_process=False):
    eval_measures = torch.zeros(10).cuda(device=gpu)
    for _, eval_sample_batched in enumerate(tqdm(dataloader_eval.data)):
        with torch.no_grad():
            image = torch.autograd.Variable(eval_sample_batched['image'].cuda(gpu, non_blocking=True))
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

        eval_measures[:9] += torch.tensor(measures).cuda(device=gpu)
        eval_measures[9] += 1

    if not args.multiprocessing_distributed or gpu == 0:
        eval_measures_cpu = eval_measures.cpu()
        cnt = eval_measures_cpu[9].item()
        eval_measures_cpu /= cnt
        print('Computing errors for {} eval samples'.format(int(cnt)), ', post_process: ', post_process)
        print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format('silog', 'abs_rel', 'log10', 'rms',
                                                                                     'sq_rel', 'log_rms', 'd1', 'd2',
                                                                                     'd3'))
        for i in range(8):
            print('{:7.4f}, '.format(eval_measures_cpu[i]), end='')
        print('{:7.4f}'.format(eval_measures_cpu[8]))
        return eval_measures_cpu

    eval_labels = ['silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'd1', 'd2', 'd3']

    # Dizionario per associare etichette e valori
    eval_metrics = {eval_labels[i]: eval_measures[i].item() for i in range(len(eval_labels))}

    # Stampa formattata
    for labels, vals in eval_metrics.items():
        print(f"{labels}: {vals:.4f}")

    return eval_measures


def main_worker(gpu, ngpus_per_node, args):
    # Build dataloaders
    dataloader = NewDataLoader(args, 'train')
    dataloader_eval = NewDataLoader(args, 'online_eval')
    
    formatter = TextFormatter()
    print("[Data]",end=" ")
    data_str = formatter.format(
        f"Dataset {args.dataset.upper()} correctly instantiated \u2713", 
        color = 'green', 
        style = ['bold'], 
        separator = False
    )
    print(data_str)
    print_examples(dataloader)
    
    print(f"[Data] Dataset name: {args.dataset}")
    print(f"[Data] Train data: {dataloader.data.__len__()} samples")
    print(f"[Data] Train batch size: {args.batch_size} samples")
    print(f"[Data] Evaluation data: {dataloader_eval.data.__len__()} samples")
    print(f"[Data] Evaluation batch size: {args.eval_batch_size} samples")

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
        pretrained = args.pretrain,
        opt_type = args.opt_type,
        opt_where = args.opt_where
    )

    model.cuda()
    model = torch.nn.DataParallel(model)
    
    # Model characteristics
    print(f"[Model] Model builded!")
    print(f"[Model] PixelFormer version: {args.encoder[:-2]}")
    if args.opt_type == "moh":
        moh_str = formatter.format(
            f"[LOG] Using additive MoH term in loss", 
            color = 'cyan', 
            style = ['bold'], 
            separator = False
        )
        print(moh_str)
    
    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print(f"[Model] Trainable parameters: {num_params:,}")

    model_weight = get_net_weight(model)
    print(f"[Model] Network memory weight: {model_weight} MB")
    
    global_step = 0
    best_eval_measures_lower_better = torch.zeros(6).cpu() + 1e3
    best_eval_measures_higher_better = torch.zeros(3).cpu()
    best_eval_steps = np.zeros(9, dtype=np.int32)

    # Optimization
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr = args.learning_rate
    )

    print(f"[Optimization] Epochs: {args.num_epochs}")
    print(f"[Optimization] Optimizer: {optimizer.__class__.__name__}")
    print(f"[Optimization] Learning rate: {optimizer.param_groups[0]['lr']}")
    print(f"[Optimization] End scheduler learning rate: {args.end_learning_rate}")

    # Loss
    silog_criterion = silog_loss(variance_focus=args.variance_focus)

    # Forward test
    dummy_img = torch.rand(3,args.input_height,args.input_width).unsqueeze(0)
    print(f"[Input] Input RGB image shape: {tuple(dummy_img.squeeze(0).shape)}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        dummy_depth = model(dummy_img.to(args.gpu))
    print(f"[Output] PixelFormer depth shape: {tuple(dummy_depth.squeeze(0).shape)}")

    # Logging
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        writer = SummaryWriter(args.log_directory + '/' + args.model_name + '/summaries', flush_secs=30)
        if args.do_online_eval:
            if args.eval_summary_directory != '':
                eval_summary_path = os.path.join(args.eval_summary_directory, args.model_name)
            else:
                eval_summary_path = os.path.join(args.log_directory, args.model_name, 'eval')
            eval_summary_writer = SummaryWriter(eval_summary_path, flush_secs=30)

    # Initial parameters stats
    var_sum = [var.sum().item() for var in model.parameters() if var.requires_grad]
    var_cnt = len(var_sum)
    var_sum = np.sum(var_sum)

    print(f"[Stats] Initial parameter sum value: {var_sum:.3f}")
    print(f"[Stats] Initial paramenter mean value: {var_sum/var_cnt:.3f}")

    # Train loop
    train_str = formatter.format(
        f"Starting PixelFormer {args.encoder[:-2]} training", 
        color = 'blue', 
        style = ['bold'], 
        separator = True
    )
    print(train_str)
    
    model.train()
    start_time = time.time()
    duration = 0
    num_log_images = args.batch_size
    end_learning_rate = args.end_learning_rate if args.end_learning_rate != -1 else 0.1 * args.learning_rate
    steps_per_epoch = len(dataloader.data)
    num_total_steps = args.num_epochs * steps_per_epoch
    epoch = global_step // steps_per_epoch
    accumulation_steps = 10000
    model_just_loaded = True

    while epoch < args.num_epochs:
        if args.distributed:
            dataloader.train_sampler.set_epoch(epoch)

        for step, sample_batched in enumerate(dataloader.data):
            if args.opt_type == "moh":
                if args.opt_where == "enc":
                    SwinMoHWindowAttention.SWIN_LOAD_BALANCING_LOSSES.clear()
                elif args.opt_where == "dec":
                    SAMMoHWindowAttention.SAM_LOAD_BALANCING_LOSSES.clear()
                else:
                    SwinMoHWindowAttention.SWIN_LOAD_BALANCING_LOSSES.clear()
                    SAMMoHWindowAttention.SAM_LOAD_BALANCING_LOSSES.clear()
                
            optimizer.zero_grad()
            before_op_time = time.time()

            image = torch.autograd.Variable(sample_batched['image'].cuda(args.gpu, non_blocking=True))
            depth_gt = torch.autograd.Variable(sample_batched['depth'].cuda(args.gpu, non_blocking=True))

            depth_est = model(image)

            if args.dataset == 'nyu':
                mask = depth_gt > 0.1
            else:
                mask = depth_gt > 1.0
            print(mask.shape)

            loss = silog_criterion.forward(depth_est, depth_gt, mask.to(torch.bool))
            # Aggiungi la Moh loss se specificato
            if args.opt_type == "moh":
                               
                # Trasferisci i tensori di LOAD_BALANCING_LOSSES sul dispositivo corretto
                if args.opt_where == "enc":
                    moh_losses_on_device = [tensor.to(args.gpu) for tensor in SwinMoHWindowAttention.SWIN_LOAD_BALANCING_LOSSES]
                elif args.opt_where == "dec":
                    moh_losses_on_device = [tensor.to(args.gpu) for tensor in SAMMoHWindowAttention.SAM_LOAD_BALANCING_LOSSES]
                else:
                    # Unisci entrambe le liste
                    moh_losses_on_device = (
                        [tensor.to(args.gpu) for tensor in SwinMoHWindowAttention.SWIN_LOAD_BALANCING_LOSSES] +
                        [tensor.to(args.gpu) for tensor in SAMMoHWindowAttention.SAM_LOAD_BALANCING_LOSSES]
                    )
                
                # Calcola la Moh loss
                moh_loss = 0.01 * sum(moh_losses_on_device) / max(len(moh_losses_on_device), 1)
                moh_loss = moh_loss.detach()  # Previene la creazione di un secondo backward pass
                # Aggiungi la Moh loss alla loss principale
                loss = loss + moh_loss

            loss.backward()
            for param_group in optimizer.param_groups:
                current_lr = (args.learning_rate - end_learning_rate) * (1 - global_step / num_total_steps) ** 0.9 + end_learning_rate
                param_group['lr'] = current_lr
            
            optimizer.step()

            if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                print('[epoch][s/s_per_e/gs]: [{}][{}/{}/{}], lr: {:.12f}, loss: {:.12f}'.format(epoch, step, steps_per_epoch, global_step, current_lr, loss))
                if np.isnan(loss.cpu().item()):
                    nanloss_str = formatter.format(
                        "[!!!] NaN in loss occurred. Aborting training",  
                        color = 'red', 
                        style = ['bold'], 
                        separator = False
                    )
                    print(nanloss_str)
                    return -1

            duration += time.time() - before_op_time
            if global_step and global_step % args.log_freq == 0 and not model_just_loaded:
                var_sum = [var.sum().item() for var in model.parameters() if var.requires_grad]
                var_cnt = len(var_sum)
                var_sum = np.sum(var_sum)
                examples_per_sec = args.batch_size / duration * args.log_freq
                duration = 0
                time_sofar = (time.time() - start_time) / 3600
                training_time_left = (num_total_steps / global_step - 1.0) * time_sofar
                if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                    print("{}".format(args.model_name))
                print_string = 'GPU: {} | examples/s: {:4.2f} | loss: {:.5f} | var sum: {:.3f} avg: {:.3f} | time elapsed: {:.2f}h | time left: {:.2f}h'
                print(print_string.format(args.gpu, examples_per_sec, loss, var_sum.item(), var_sum.item()/var_cnt, time_sofar, training_time_left))

                if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                            and args.rank % ngpus_per_node == 0):
                    writer.add_scalar('silog_loss', loss, global_step)
                    writer.add_scalar('learning_rate', current_lr, global_step)
                    writer.add_scalar('var average', var_sum.item()/var_cnt, global_step)
                    depth_gt = torch.where(depth_gt < 1e-3, depth_gt * 0 + 1e3, depth_gt)
                    for i in range(num_log_images):
                        writer.add_image('depth_gt/image/{}'.format(i), normalize_result(1/depth_gt[i, :, :, :].data), global_step)
                        writer.add_image('depth_est/image/{}'.format(i), normalize_result(1/depth_est[i, :, :, :].data), global_step)
                        writer.add_image('image/image/{}'.format(i), inv_normalize(image[i, :, :, :]).data, global_step)
                    writer.flush()

            if args.do_online_eval and global_step and global_step % args.eval_freq == 0 and not model_just_loaded:
                model.eval()
                with torch.no_grad():
                    eval_measures = online_eval(args, model, dataloader_eval, gpu, ngpus_per_node, post_process=True)
                    print(eval_measures)
                if eval_measures is not None:
                    for i in range(9):
                        eval_summary_writer.add_scalar(eval_metrics[i], eval_measures[i].cpu(), int(global_step))
                        measure = eval_measures[i]
                        is_best = False
                        if i < 6 and measure < best_eval_measures_lower_better[i]:
                            old_best = best_eval_measures_lower_better[i].item()
                            best_eval_measures_lower_better[i] = measure.item()
                            is_best = True
                        elif i >= 6 and measure > best_eval_measures_higher_better[i-6]:
                            old_best = best_eval_measures_higher_better[i-6].item()
                            best_eval_measures_higher_better[i-6] = measure.item()
                            is_best = True
                        if is_best:
                            old_best_step = best_eval_steps[i]
                            old_best_name = '/model-{}-best_{}_{:.5f}'.format(old_best_step, eval_metrics[i], old_best)
                            model_path = args.log_directory + '/' + args.model_name + old_best_name
                            if os.path.exists(model_path):
                                command = 'rm {}'.format(model_path)
                                os.system(command)
                            best_eval_steps[i] = global_step
                            model_save_name = '/model-{}-best_{}_{:.5f}'.format(global_step, eval_metrics[i], measure)
                            print('New best for {}. Saving model: {}'.format(eval_metrics[i], model_save_name))
                            checkpoint = {'model': model.state_dict()}
                            torch.save(checkpoint, args.log_directory + '/' + args.model_name + model_save_name)
                    eval_summary_writer.flush()
                model.train()
                # block_print()
                # enable_print()

            model_just_loaded = False
            global_step += 1

        epoch += 1
       
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        writer.close()
        if args.do_online_eval:
            eval_summary_writer.close()

    end_str = formatter.format(
        "Training ended successfully \u2713", 
        color = 'green', 
        style = ['bold','blink'], 
        separator = False
    )
    print(end_str)
   
    version = args.encoder[:-2]
    send_email("PixelFormer", version, args.opt_type, args.opt_where)

def train(args):
    formatter = TextFormatter()
    prep_str = formatter.format(
        f"Preparing training of Pixelformer {args.encoder[:-2]}", 
        color = 'blue', 
        style = ['bold'], 
        separator = True
    )
    print(prep_str)
    
    torch.cuda.empty_cache()
    ngpus_per_node = torch.cuda.device_count()
    if ngpus_per_node > 1 and not args.multiprocessing_distributed:
        nanloss_str = formatter.format(
            "[!!!] This machine has more than 1 GPU\n[!!!] Please specify --multiprocessing_distributed, or set \'CUDA_VISIBLE_DEVICES=0\'",  
            color = 'red', 
            style = ['bold'], 
            separator = False
        )
        print(nanloss_str)
        return -1

    if args.do_online_eval:
        print("[LOG] ", end = "")
        onl_str = formatter.format(
            "Online evaluation enabled \u2713", 
            color = 'green', 
            style = ['bold'], 
            separator = False
        )
        print(onl_str)
        
        print(f"[LOG] Model will be evaluated every {args.eval_freq} training steps")
        print(f"[LOG] Best models will be saved for individual eval metrics")
              

    if args.multiprocessing_distributed:
        print(f"[LOG] Model will trained on {ngpus_per_node} GPUs")
    else:
        print(f"[LOG] Model will trained on 1 GPUs")
    main_worker(args.gpu, ngpus_per_node, args)
