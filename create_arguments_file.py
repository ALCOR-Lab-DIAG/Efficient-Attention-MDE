import argparse
import os

def main():
    # Configurazione del parser per gli argomenti
    parser = argparse.ArgumentParser(description="Crea un file di configurazione per il modello.")
    parser.add_argument("main_script", type=str, help="Il percorso allo script principale (MAIN_SCRIPT).")
    parser.add_argument("config_file", type=str, help="Il percorso al file di configurazione (CONFIG_FILE).")
    parser.add_argument("mode", type=str, choices=["train","test","analysis", "stats","compress"], help="Modalità di esecuzione: 'analysis' o 'stats'")
    parser.add_argument("extra", type=str, help="Run tests for quantized or pruned models")
    
    # Parsing degli argomenti
    args = parser.parse_args()
    
    # Recupero degli argomenti
    main_script = args.main_script
    config_file = args.config_file
    
    # Stampa per debugging (opzionale)
    print(f"Main script: {main_script}")
    print(f"Config file: {config_file}")
    
    list_main_script = [x for x in args.main_script.split("/") if x != '']
    list_config_file = [x for x in args.config_file.split("/") if x != '']

    network = os.path.splitext(list_main_script[-1])[0].split("_")[-1]    
    if network == "pxf":
        network_full_name = "PixelFormer"
    elif network == "newcrfs":
        network_full_name = "NeWCRFs"
    else:
        network_full_name = "METER"

    config_file_name = list_config_file[-1].strip(".txt")
    config_file_path = config_file.replace("/work/project/", "./")
    config_file_dir_path = os.path.dirname(config_file_path)
    # Controlla il contenuto della cartella
    for file_name in os.listdir(config_file_dir_path):
        # Verifica se è un file e se il nome inizia con "arguments"
        if file_name.startswith("arguments") and os.path.isfile(os.path.join(config_file_dir_path, file_name)):
            file_path = os.path.join(config_file_dir_path, file_name)
            os.remove(file_path)  # Elimina il file
            print(f"Removed precedent arguments file")

    config_file_name_list = config_file_name.split("_")
    
    if args.extra:
        extra = args.extra
    else:
        extra = ""
    
    if args.mode == "analysis" or args.mode == "stats" or args.mode == "compress":
        mode = args.mode
    else:
        mode = config_file_name_list[1]

    size = config_file_name_list[-2]
    dataset = config_file_name_list[-1]

    if len(config_file_name_list) > 4:
        is_optimized = True    
        optimization_type = config_file_name_list[2]
        optimization_where = config_file_name_list[3]
        if optimization_where[0] != "f":
            optimization_where = optimization_where[:3]
    else:
        is_optimized = False
        optimization_type = None
        optimization_where = None

    if mode == "train" and network != "meter":
        mode_str = f"--mode {mode}\n"
        
        if is_optimized:
            model_str = f"--model_name {network.lower()}_{optimization_type}_{optimization_where}_{size}_{dataset}\n"
        else:
            model_str = f"--model_name {network.lower()}_{size}_{dataset}\n"

        size_str = f"--encoder {size}07\n"
        pretrain_str = f"--pretrain work/data/ssd_pretrained/swin_{size}_patch4_window7_224_22k.pth\n"
        sep_str_1 = "\n"
        dataset_str = f"--dataset {dataset}\n"
        if dataset == "nyu":
            data_path_str = "--data_path work/data/ssd_datasets/nyu_depth_v2/sync/\n"
            gt_path_str = "--gt_path work/data/ssd_datasets/nyu_depth_v2/sync/\n"
            filenames_str = f"--filenames_file work/project/{network_full_name}/data_splits/nyudepthv2_train_files_with_gt_dense.txt\n"
            heigt_str = "--input_height 480\n"
            width_str = "--input_width 640\n"
            max_depth_str = "--max_depth 10\n"
        else:
            data_path_str = "--data_path work/data/ssd_datasets/kitti/kitti_train\n"
            gt_path_str = "--gt_path work/data/ssd_datasets/kitti/kitti_dataset\n"
            filenames_str = f"--filenames_file work/project/{network_full_name}/data_splits/eigen_train_files_with_gt.txt\n"
            heigt_str = "--input_height 352\n"
            width_str = "--input_width 1120\n"
            max_depth_str = "--max_depth 80\n"

        sep_str_2 = "\n"
        log_directory_str = "--log_directory work/data/ssd_results/\n"
        log_freq_str = "--log_freq 100\n"
        sep_str_3 = "\n"
        weight_decay_str = "--weight_decay 1e-2\n"
        adam_eps_str = "--adam_eps 1e-3\n"
        batch_size_str = "--batch_size 8\n"
        num_epochs_str = "--num_epochs 10\n"
        learning_rate_str = "--learning_rate 4e-5\n"
        end_learning_rate_str = "--end_learning_rate 4e-6\n"
        variance_focus_str = "--variance_focus 0.85\n"
        multiprocessing_distributed_str = "--multiprocessing_distributed\n"
        sep_str_4 = "\n"
        do_random_rotate_str = "--do_random_rotate\n"
        
        if dataset == "nyu":
            degree_str = "--degree 2.5\n"
            do_kb_crop_str = ""
        else:
            degree_str = "--degree 1.0\n"
            do_kb_crop_str = "--do_kb_crop\n"
        
        sep_str_5 = "\n"
        num_threads_str = "--num_threads 7\n"
        dist_url_str = "--dist_url 0\n"
        sep_str_6 = "\n"
        do_online_eval_str = "--do_online_eval\n"
        
        if dataset == "nyu":
            data_path_eval_str = "--data_path_eval work/data/ssd_datasets/nyu_depth_v2/official_splits/test/\n"
            gt_path_eval_str = "--gt_path_eval work/data/ssd_datasets/nyu_depth_v2/official_splits/test/\n"
            filenames_file_eval_str = f"--filenames_file_eval work/project/{network_full_name}/data_splits/nyudepthv2_test_files_with_gt.txt\n"
            min_depth_eval_str = "--min_depth_eval 1e-3\n"
            max_depth_eval_str = "--max_depth_eval 10\n"
            crop_str = "--eigen_crop\n"
        else:
            data_path_eval_str = "--data_path_eval work/data/ssd_datasets/kitti/kitti_train\n"
            gt_path_eval_str = "--gt_path_eval work/data/ssd_datasets/kitti/kitti_dataset\n"
            filenames_file_eval_str = f"--filenames_file_eval work/project/{network_full_name}/data_splits/eigen_test_files_with_gt.txt\n"
            min_depth_eval_str = "--min_depth_eval 1e-3\n"
            max_depth_eval_str = "--max_depth_eval 80\n"
            crop_str = "--garg_crop\n"
            
        eval_freq_str = "--eval_freq 1000\n\n"
        
        if is_optimized:
            opt_type_str = f"--opt_type {optimization_type}\n"
            opt_where_str = f"--opt_where {optimization_where}"
        else:
            opt_type_str = ""
            opt_where_str = ""

        config_lines = [        
            mode_str,
            model_str,
            size_str,
            pretrain_str,
            sep_str_1,
            dataset_str,
            data_path_str,
            gt_path_str,
            filenames_str,
            heigt_str,
            width_str,
            max_depth_str,
            sep_str_2,
            log_directory_str,
            log_freq_str,
            sep_str_3,
            weight_decay_str,
            adam_eps_str,
            batch_size_str,
            num_epochs_str,
            learning_rate_str,
            end_learning_rate_str,
            variance_focus_str,
            multiprocessing_distributed_str,
            sep_str_4,
            do_random_rotate_str,
            degree_str,
            do_kb_crop_str,
            sep_str_5,
            num_threads_str,
            dist_url_str,
            sep_str_6,
            do_online_eval_str,
            data_path_eval_str,
            gt_path_eval_str,
            filenames_file_eval_str,
            min_depth_eval_str,
            max_depth_eval_str,
            crop_str,
            eval_freq_str,
            opt_type_str,
            opt_where_str
        ]

    elif (mode == "test" or mode == "analysis" or mode == "stats" or mode == "compress") and network != "meter":
        mode_str = f"--mode {mode}\n"
        model_str = f"--model_name {network.lower()}_{dataset}\n"
        size_str = f"--encoder {size}07\n"
        dataset_str = f"--dataset {dataset}\n"

        if extra == "p":
            save_path_str = f"--save_path /work/data/ssd_results/{network.lower()}_{size}_{dataset}_pruned/\n"
        elif extra == "q":
            save_path_str = f"--save_path /work/data/ssd_results/{network.lower()}_{size}_{dataset}_quantized/\n"
        elif is_optimized:
            save_path_str = f"--save_path /work/data/ssd_results/{network.lower()}_{optimization_type}_{optimization_where}_{size}_{dataset}/\n"
        else:
            save_path_str = f"--save_path /work/data/ssd_results/{network.lower()}_{size}_{dataset}/\n"
        
        if dataset == "nyu":
            heigt_str = "--input_height 480\n"
            width_str = "--input_width 640\n"
            max_depth_str = "--max_depth 10\n"
            do_kb_crop_str = "\n"

            data_path_eval_str = "--data_path_eval work/data/ssd_datasets/nyu_depth_v2/official_splits/test/\n"
            gt_path_eval_str = "--gt_path_eval work/data/ssd_datasets/nyu_depth_v2/official_splits/test/\n"
            filenames_file_eval_str = f"--filenames_file_eval work/project/{network_full_name}/data_splits/nyudepthv2_test_files_with_gt.txt\n"
            min_depth_eval_str = "--min_depth_eval 1e-3\n"
            max_depth_eval_str = "--max_depth_eval 10\n"
            eigen_str = "--eigen_crop\n"
        else:
            heigt_str = "--input_height 352\n"
            width_str = "--input_width 1216\n"
            max_depth_str = "--max_depth 80\n"
            do_kb_crop_str = "--do_kb_crop\n\n"

            data_path_eval_str = "--data_path_eval work/data/ssd_datasets/kitti/kitti_train\n"
            gt_path_eval_str = "--gt_path_eval work/data/ssd_datasets/kitti/kitti_dataset\n"
            filenames_file_eval_str = f"--filenames_file_eval work/project/{network_full_name}/data_splits/eigen_test_files_with_gt.txt\n"
            min_depth_eval_str = "--min_depth_eval 1e-3\n"
            max_depth_eval_str = "--max_depth_eval 80\n"
            eigen_str = "--garg_crop\n"

        if extra == "p":
            checkpint_str = f"--checkpoint_path work/data/ssd_pretrained/checkpoints/{network}_{size}_{dataset}_pruned\n"
        elif extra == "q":
            checkpint_str = f"--checkpoint_path work/data/ssd_pretrained/checkpoints/{network}_{size}_{dataset}_quantized\n"
        elif is_optimized:
            checkpint_str = f"--checkpoint_path work/data/ssd_pretrained/checkpoints/{network}_{optimization_type}_{optimization_where}_{size}_{dataset}\n"
        else:
            checkpint_str = f"--checkpoint_path work/data/ssd_pretrained/checkpoints/{network}_{size}_{dataset}\n"
 
        if is_optimized:
            opt_type_str = f"\n--opt_type {optimization_type}\n"
            opt_where_str = f"--opt_where {optimization_where}"
        else:
            opt_type_str = ""
            opt_where_str = ""

        if extra:
            extra_str = f"--extra {extra}\n"

        config_lines = [
            mode_str,
            model_str,
            size_str,
            dataset_str,
            save_path_str,
            heigt_str,
            width_str,
            max_depth_str,
            do_kb_crop_str,
            data_path_eval_str,
            gt_path_eval_str,
            filenames_file_eval_str,
            min_depth_eval_str,
            max_depth_eval_str,
            eigen_str,
            checkpint_str,
            opt_type_str,
            opt_where_str,
            extra_str if extra != "" else ""
        ]     
    
    elif (mode == "train" or mode == "test" or mode == "stats" or mode == "compress") and network == "meter":
        mode_str = f"--mode {mode}\n"
        
        if size == "tiny":
            compressed_size = "xxs"
        elif size == "base":
            compressed_size = "xs"
        else:
            compressed_size = "s"

        model_name_str = "--model_name meter\n"
        if dataset == "nyu":
            rgb_img_res_str = "--rgb_img_res (3,192,256)\n"
            d_img_res_str = "--d_img_res (1,48,64)\n"    
            data_path_str = ""
            gt_path_str = ""
            filenames_file_str = ""
            data_path_eval_str = ""
            gt_path_eval_str = ""
            filenames_file_eval_str = ""
        else:
            rgb_img_res_str = "--rgb_img_res (3,192,636)\n"
            d_img_res_str = "--d_img_res (1,48,160)\n"   
            data_path_str = "--data_path work/data/ssd_datasets/kitti/kitti_train\n"
            gt_path_str = "--gt_path work/data/ssd_datasets/kitti/kitti_dataset\n"
            filenames_file_str = "--filenames_file work/project/METER/data_splits/eigen_train_files_with_gt.txt\n"
            data_path_eval_str = "--data_path_eval work/data/ssd_datasets/kitti/kitti_train\n"
            gt_path_eval_str = "--gt_path_eval work/data/ssd_datasets/kitti/kitti_dataset\n"
            filenames_file_eval_str = "--filenames_file_eval work/project/METER/data_splits/eigen_test_files_with_gt.txt\n"
            
        if dataset == "nyu":
            if is_optimized and optimization_type != "moh":
                prentrained_str = f"--pretrained_w work/data/ssd_pretrained/build_model_best_{optimization_type}_{dataset}_{compressed_size}\n"        
            else:
                prentrained_str = f"--pretrained_w work/data/ssd_pretrained/build_model_best_{dataset}_{compressed_size}\n"
        else:
            prentrained_str = f"--pretrained_w work/data/ssd_pretrained/build_model_best_{dataset}_{compressed_size}\n"

        do_prints_str = "--do_prints\n"
        do_print_model_str = "--do_print_model\n"
        if (mode == "test" or mode == "stats" or mode == "compress"):
            do_pretrained_str = "--do_pretrained\n"
        else:
            do_pretrained_str = ""
        
        if mode == "train":
            do_train_str = "--do_train\n"
            do_print_best_worst_str = "--do_print_best_worst\n"
        else:
            do_train_str = ""
            do_print_best_worst_str = ""
        
        dts_type_str = f"--dts_type {dataset}\n"
        architecture_type_str = f"--architecture_type {compressed_size}\n"
        
        seed_str = "--seed 42\n"
        lr_str = "--lr 1e-3\n"

        if dataset == "nyu":
            lr_patience_str = "--lr_patience 30\n"
            epochs_str = "--epochs 250\n"
        else:
            lr_patience_str = "--lr_patience 40\n"
            epochs_str = "--epochs 800\n"

        batch_size_str = "--batch_size 64\n"
        batch_size_eval_str = "--batch_size_eval 1\n"
        if mode == "train":
            n_workers_str = "--n_workers 8\n"
        else:
            n_workers_str = "--n_workers 1\n"
        e_stop_epochs_str = "--e_stop_epochs 150\n"
        size_train_str = "--size_train None\n"
        size_test_str = "--size_test None\n"

        flip_str = "--flip 0.5\n"
        mirror_str = "--mirror 0.5\n"
        color_and_bright_str = "--color_and_bright 0.5\n"
        c_swap_str = "--c_swap 0.5\n"
        random_crop_str = "--random_crop 0.5\n"
        random_d_shift_str = "--random_d_shift 0.5\n"
        
        if not is_optimized:
            network_type_str = "--network_type METER\n"
        else:
            network_type_str = f"--network_type {optimization_type}_METER\n"

        dataset_root_str = "--dataset_root /work/data/ssd_datasets/\n"
        save_model_root_str = f"--save_model_root /work/data/ssd_results/meter_{dataset}/\n"
        
        if mode == "test" or mode == "stats" or mode == "compress":
            if extra == "p":
                checkpoint_str = f"--checkpoint work/data/ssd_pretrained/checkpoints/meter_{size}_{dataset}_pruned\n"
            elif extra == "q":
                checkpoint_str = f"--checkpoint work/data/ssd_pretrained/checkpoints/meter_{size}_{dataset}_quantized\n"
            elif is_optimized:
                checkpoint_str = f"--checkpoint work/data/ssd_pretrained/checkpoints/meter_{optimization_type}_{size}_{dataset}\n"
            else:
                checkpoint_str = f"--checkpoint work/data/ssd_pretrained/checkpoints/meter_{size}_{dataset}\n"
                # checkpoint_str = f"--checkpoint work/data//ssd_results/meter_tiny_nyu/METER_best\n"
        else:
            checkpoint_str = "\n"
        
        if extra == "p":
            save_model_root_str = f"--save_model_root /work/data/ssd_results/meter_{size}_{dataset}_pruned/\n"
        elif extra == "q":
            save_model_root_str = f"--save_model_root /work/data/ssd_results/meter_{size}_{dataset}_quantized/\n"
        elif is_optimized:
            save_model_root_str = f"--save_model_root /work/data/ssd_results/meter_{optimization_type}_{size}_{dataset}/\n"
        else:
            save_model_root_str = f"--save_model_root /work/data/ssd_results/meter_{size}_{dataset}/\n"

        if extra:
            extra_str = f"--extra {extra}\n"

        config_lines = [
            mode_str,
            model_name_str,
            rgb_img_res_str,
            d_img_res_str,
            data_path_str,
            gt_path_str,
            filenames_file_str,
            data_path_eval_str,
            gt_path_eval_str,
            filenames_file_eval_str,
            prentrained_str,
            do_prints_str,
            do_prints_str,
            do_print_model_str,
            do_pretrained_str,
            do_train_str,
            do_print_best_worst_str,
            dts_type_str,
            architecture_type_str,
            seed_str,
            lr_str,
            lr_patience_str,
            epochs_str,
            batch_size_str,
            batch_size_eval_str,
            n_workers_str,
            e_stop_epochs_str,
            size_train_str,
            size_test_str,
            flip_str,
            mirror_str,
            color_and_bright_str,
            c_swap_str,
            random_crop_str,
            random_d_shift_str,
            network_type_str,
            dataset_root_str,
            save_model_root_str,
            checkpoint_str,
            extra_str if extra != "" else ""
        ]
    # Scrittura del file di configurazione
    try:
        with open(config_file_path, "w") as file:
            for line in config_lines:
                file.write(line)
        print(f"Configuration file created successfully at {config_file}")
    except Exception as e:
        print(f"Error while creating configuration file: {e}")
        exit(1)
    
if __name__ == "__main__":
    main()
