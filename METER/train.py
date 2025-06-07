import numpy as np
import os
import torch

from METER.datasets.dataloaders import *
from METER.globals import *
from METER.utils import *

def train(args):
    # Init training
    formatter = TextFormatter()
    prep_str = formatter.format(
        f"Preparing training of METER {args.architecture_type}", 
        color = 'blue', 
        style = ['bold'], 
        separator = True
    )
    print(prep_str)

    ngpus_per_node = torch.cuda.device_count()
    if ngpus_per_node > 0:
        print(f"[LOG] Model will trained on {ngpus_per_node} GPUs")
    else:
        print(f"[LOG] Model will trained on 1 GPUs")

    # Set-seed
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    # Datasets loading
    training_DataLoader, test_DataLoader, training_Dataset, test_Dataset = init_train_test_loader(
        args = args,
        dts_type = args.dts_type,
        dts_root_path = args.dataset_root,
        rgb_h_res = args.rgb_img_res[1],
        d_h_res = args.d_img_res[1],
        bs_train = args.batch_size,
        bs_eval = args.batch_size_eval,
        num_workers = args.n_workers,
        size_train = args.size_train,
        size_test = args.size_test
    )

    print("[Data]",end=" ")
    data_str = formatter.format(
        f"Dataset {args.dts_type.upper()} correctly instantiated \u2713", 
        color = 'green', 
        style = ['bold'], 
        separator = False
    )
    print(data_str)
    
    if args.do_prints:
        print_img(args.dts_type, test_Dataset, label='rgb_sample', quantity=2, save_model_root=args.save_model_root)
        print_img(args.dts_type, training_Dataset, label='aug_sample', quantity=5, print_info_aug=False, save_model_root=args.save_model_root)
    
    print(f"[Data] Dataset name: {args.dts_type}")
    print(f"[Data] Train data: {training_Dataset.__len__()} samples")
    print(f"[Data] Train batch size: {args.batch_size} samples")
    print(f"[Data] Evaluation data: {test_Dataset.__len__()} samples")
    print(f"[Data] Evaluation batch size: {args.batch_size_eval} samples")
    
    if "_" in args.network_type:
        opt_type = args.network_type.split("_")[0]
        network_type_name = f"{opt_type.capitalize()}-METER"
    else:
        opt_type = None
        network_type_name = "METER"
    
    model_str = formatter.format(
        f"Building {network_type_name}", 
        color = 'blue', 
        style = ['bold'], 
        separator = True
    )
    print(model_str)
        
    if opt_type == "meta":
        from METER.networks.meta_METER import meta_build_model, Attention
        print(f"[LOG] Building {opt_type.capitalize()}-METER blocks")
        model = meta_build_model(
            device = args.gpu, 
            arch_type = args.architecture_type,
            rgb_img_res = args.rgb_img_res,
        ).to(device = args.gpu)        
    elif opt_type == "pyra":
        if args.dts_type == "nyu":
            from METER.networks.pyra_METER_nyu import pyra_build_model, Attention
        else:
            from METER.networks.pyra_METER_kitti import pyra_build_model, Attention
        print(f"[LOG] Building {opt_type.capitalize()}-METER blocks")
        model = pyra_build_model(
            device = args.gpu, 
            arch_type = args.architecture_type,
            rgb_img_res = args.rgb_img_res,
        ).to(device = args.gpu)        
    elif opt_type == "moh":
        from METER.networks.moh_METER import moh_build_model, Attention
        print(f"[LOG] Building {opt_type.capitalize()}-METER blocks")
        model = moh_build_model(
            device = args.gpu, 
            arch_type = args.architecture_type,
            rgb_img_res = args.rgb_img_res,
        ).to(device = args.gpu)        
    else:
        from METER.networks.METER import build_model, Attention
        print(f"[LOG] Building STANDARD blocks")
        model = build_model(
            device = args.gpu, 
            arch_type = args.architecture_type,
            rgb_img_res = args.rgb_img_res,
        ).to(device = args.gpu)        

    # Model characteristics
    print(f"[Model] Model builded!")
    print(f"[Model] METER version: {args.architecture_type}")
    
    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print(f"[Model] Trainable parameters: {num_params:,}")

    model_weight = get_net_weight(model)
    print(f"[Model] Network memory weight: {model_weight} MB")

    # Evaluation measures
    torch.cuda.empty_cache()
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'lrs': [], 'test_rmse': [], 'l_mae': [], 'l_norm': [], 'l_grad': [], 'l_ssim': []}
    min_rmse = float('inf')
    min_acc = 0
    train_loss_list = []
    test_loss_list = []
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr = args.lr, 
        betas = (0.9, 0.999), 
        eps = 1e-08, 
        weight_decay = 0.01, 
        amsgrad = False
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode = 'min', 
        factor = 0.1, 
        patience = args.lr_patience, 
        threshold = 1e-4, 
        threshold_mode = 'rel',
        cooldown = 0, 
        min_lr = 1e-8, 
        eps = 1e-08, 
        verbose = False
    )

    # Early stopping
    trigger_times, early_stopping_epochs = 0, args.e_stop_epochs

    print(f"[Optimization] Epochs: {args.epochs}")
    print(f"[Optimization] Optimizer: {optimizer.__class__.__name__}")
    print(f"[Optimization] Learning rate: {optimizer.param_groups[0]['lr']}")

    # Loss
    criterion = balanced_loss_function(device = args.gpu)

    print(f"[LOG] Loading pretrained encoder weights from full model")
    model, _ = load_pretrained_encoder_only(
        target_model = model,
        pretrained_path = args.pretrained_w,
        device = args.gpu
    )

    pre_str = formatter.format(
        f"[LOG] Pretrained encoder successfully loaded", 
        color = 'green', 
        style = ['bold'], 
        separator = False
    )
    print(pre_str)    

    if opt_type == "moh":
        moh_str = formatter.format(
            f"[LOG] Using additive MoH term in loss", 
            color = 'cyan', 
            style = ['bold'], 
            separator = False
        )
        print(moh_str)    

    # Forward test
    dummy_img = torch.rand(args.rgb_img_res[0],args.rgb_img_res[1],args.rgb_img_res[2]).unsqueeze(0)
    print(f"[Input] Input RGB image shape: {tuple(dummy_img.squeeze(0).shape)}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        dummy_depth = model(dummy_img.to(args.gpu))
    print(f"[Output] PixelFormer depth shape: {tuple(dummy_depth.squeeze(0).shape)}")
        
    # Initial parameters stats
    var_sum = [var.sum().item() for var in model.parameters() if var.requires_grad]
    var_cnt = len(var_sum)
    var_sum = np.sum(var_sum)

    print(f"[Stats] Initial parameter sum value: {var_sum:.3f}")
    print(f"[Stats] Initial paramenter mean value: {var_sum/var_cnt:.3f}")

    # Train loop
    train_str = formatter.format(
        f"Starting {network_type_name} {args.architecture_type} training", 
        color = 'blue', 
        style = ['bold'], 
        separator = True
    )
    print(train_str)
    
    # Train
    for epoch in range(args.epochs):
        iter = 1
        model.train()
        running_loss, accuracy = 0, 0
        running_l_mae, running_l_grad, running_l_norm, running_l_ssim = 0, 0, 0, 0
        with tqdm(training_DataLoader, unit="step", position=0, leave=True) as tepoch:
            for batch in tepoch:
                if opt_type == "moh":
                    Attention.LOAD_BALANCING_LOSSES.clear()
                    
                tepoch.set_description(f"Epoch {epoch + 1}/{args.epochs} - Training")
                # Load data
                inputs, depths = batch[0].to(device=args.gpu), batch[1].to(device=args.gpu)
                
                # Forward
                optimizer.zero_grad()
                outputs = model(inputs)
                
                # Compute loss
                loss_depth, loss_ssim, loss_normal, loss_grad = criterion(outputs, depths)
                loss = loss_depth + loss_normal + loss_grad + loss_ssim
                if opt_type == "moh":
                    moh_losses_on_device = [tensor.to(args.gpu) for tensor in Attention.LOAD_BALANCING_LOSSES]
                    # Calcola la Moh loss
                    moh_loss = 0.01 * sum(moh_losses_on_device) / max(len(moh_losses_on_device), 1)
                    moh_loss = moh_loss.detach()  # Previene la creazione di un secondo backward pass
                    # Aggiungi la Moh loss alla loss principale
                    loss = loss + moh_loss

                # Backward
                loss.backward()
                optimizer.step()
                
                # Evaluation and Stats
                running_loss += loss.item()
                running_l_mae += loss_depth.item()
                running_l_norm += loss_normal.item()
                running_l_grad += loss_grad.item()
                running_l_ssim += loss_ssim.item()

                train_loss_support = [loss_depth.item(), loss_normal.item(), loss_grad.item(), loss.item()]
                train_loss_list.append(train_loss_support)

                accuracy += compute_accuracy(outputs, depths)
                tepoch.set_postfix(
                    {
                        'Loss': running_loss / iter,
                        'Acc': accuracy.item() / iter,
                        'Lr': args.lr if not history['lrs'] else history['lrs'][-1],
                        'L_mae': running_l_mae / iter,
                        'L_norm': running_l_norm / iter,
                        'L_grad': running_l_grad / iter,
                        'L_ssim': running_l_ssim / iter
                    }
                )
                iter += 1

        # Validation
        iter = 1
        model.eval()
        test_loss, test_accuracy, test_rmse = 0, 0, 0
        with tqdm(test_DataLoader, unit="step", position=0, leave=True) as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}/{args.epochs} - Validation")
                inputs, depths = batch[0].to(device=args.gpu), batch[1].to(device=args.gpu)
                # Validation loop
                with torch.no_grad():
                    outputs = model(inputs)
                    
                    # Evaluation metrics
                    test_accuracy += compute_accuracy(outputs, depths)
                    
                    # Loss
                    loss_depth, loss_ssim, loss_normal, loss_grad = criterion(outputs, depths)
                    loss = loss_depth + loss_normal + loss_grad + loss_ssim
                    test_loss += loss.item()
                    test_loss_support = [loss_depth.item(), loss_normal.item(), loss_grad.item(), loss.item()]
                    test_loss_list.append(test_loss_support)

                    # RMSE
                    test_rmse += compute_rmse(outputs, depths)
                    tepoch.set_postfix(
                        {
                            'Loss': test_loss / iter, 
                            'Acc': test_accuracy.item() / iter,
                            'RMSE': test_rmse.item() / iter
                        }
                    )
                    iter += 1

        # Update history infos
        history['lrs'].append(get_lr(optimizer))
        history['train_loss'].append(running_loss / len(training_DataLoader))
        history['val_loss'].append(test_loss / len(test_DataLoader))
        history['train_acc'].append(accuracy.item() / len(training_DataLoader))
        history['val_acc'].append(test_accuracy.item() / len(test_DataLoader))
        history['test_rmse'].append(test_rmse.item() / len(test_DataLoader))
        
        # Update history losses infos
        history['l_mae'].append(running_l_mae / len(training_DataLoader))
        history['l_norm'].append(running_l_norm / len(training_DataLoader))
        history['l_grad'].append(running_l_grad / len(training_DataLoader))
        history['l_ssim'].append(running_l_ssim / len(training_DataLoader))
        
        # Update scheduler LR
        scheduler.step(history['test_rmse'][-1])
        
        # Save model by best RMSE
        if min_rmse >= (test_rmse / len(test_DataLoader)):
            trigger_times = 0
            min_rmse = test_rmse / len(test_DataLoader)
            save_checkpoint(model, args.network_type + '_best', args.save_model_root)
            print('[LOG] New best RMSE: {:.3f} at epoch {}'.format(min_rmse, epoch + 1))
        else:
            trigger_times += 1
            print('[LOG] RMSE did not improved, EarlyStopping from {} epochs'.format(early_stopping_epochs - trigger_times))
        
        # Save model by best ACCURACY
        if min_acc <= (test_accuracy / len(test_DataLoader)):
            min_acc = test_accuracy / len(test_DataLoader)
            save_checkpoint(model, args.network_type + '_best_acc', args.save_model_root)
            print('[LOG] New best ACCURACY: {:.3f} at epoch {}'.format(min_acc, epoch + 1))
            if trigger_times > 4:
                trigger_times = trigger_times - 2
                print(f"[LOG] EarlyStopping increased due to Accuracy, stop in {early_stopping_epochs - trigger_times} epochs")

        save_prediction_examples(
            model, 
            dataset = test_Dataset,
            device = args.gpu,
            indices = [0, 216, 432, 639], 
            ep = epoch,
            save_path = args.save_model_root + 'evolution_img/'
        )
        save_history(history, args.save_model_root + args.network_type + '_history')
        
        # Empty CUDA cache
        torch.cuda.empty_cache()

        if trigger_times == early_stopping_epochs:
            print('[LOG] Val Loss did not imporved for {} epochs, training stopped'.format(early_stopping_epochs + 1))
            break

        # Save loss for graphs
        np.save(args.save_model_root + 'train.npy', np.array(train_loss_list))
        np.save(args.save_model_root + 'test.npy', np.array(test_loss_list))

    end_str = formatter.format(
        "Training ended successfully \u2713", 
        color = 'green', 
        style = ['bold','blink'], 
        separator = False
    )
    print(end_str)
    
    save_csv_history(model_name=args.network_type, path=args.save_model_root)
    plot_history(history, path=args.save_model_root)
    plot_loss_parts(history, path=args.save_model_root, title='Loss Components')

    if "_" in args.network_type:
        opt_type = args.network_type.split("_")[0]
    else:
        opt_type = "None"

    send_email("METER", args.architecture_type, opt_type)
