from METER.datasets.dataloaders import *
from METER.globals import *
from METER.utils import *

def inference_time(args):
    # Init training
    formatter = TextFormatter()
    prep_str = formatter.format(
        f"Preparing testing of METER {args.architecture_type}", 
        color = 'blue', 
        style = ['bold'], 
        separator = True
    )
    print(prep_str)

    torch.cuda.empty_cache()
    print(f"[LOG] Model will trained on {args.gpu} GPU")

    # Set-seed
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    # Datasets loading
    _, test_DataLoader, _, _ = init_train_test_loader(
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

    if args.extra == "q":
        model = torch.load(args.checkpoint, map_location='cpu')
    elif opt_type == "meta":
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

    if args.extra != "q":
        model, _ = load_pretrained_model(
            model=model,
            path_weigths=args.checkpoint,
            device=args.gpu,
            do_pretrained=args.do_pretrained
        )

    # Model characteristics
    print(f"[Model] Model builded!")
    print(f"[Model] METER version: {args.architecture_type}")
    
    if args.extra != "q":
        num_params = sum([np.prod(p.size()) for p in model.parameters()])
        print(f"[Model] Trainable parameters: {num_params:,}")

        model_weight = get_net_weight(model)
        print(f"[Model] Network memory weight: {model_weight} MB")

    # Train loop
    test_str = formatter.format(
        f"Starting {network_type_name} {args.architecture_type} testing", 
        color = 'blue', 
        style = ['bold'], 
        separator = True
    )
    print(test_str)

    model_name = args.model_name
    size = "tiny" if "xxs" in args.architecture_type else "base" if "xs" in args.architecture_type else "large" if "s" in args.architecture_type else "unknown"
    dataset = args.dts_type
    opt_name = opt_type

    if opt_name is not None:
        full_name = model_name + "_" + size + "_" + opt_name + "_" + dataset
    else:
        full_name = model_name + "_" + size + "_" + dataset

    # Evaluate
    # model.cuda()   
    if args.extra == "q":
        quant = True
    else:
        quant = False

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        _, avg, elapsed_time = compute_evaluation(
            test_dataloader = test_DataLoader,
            model = model,
            model_type = '_',
            path_save_csv_results = args.save_model_root,
            dts_type = args.dts_type,
            quant = quant
        )
    
    res_str = formatter.format(
        f"Results", 
        color = 'blue', 
        style = ['bold'], 
        separator = True
    )
    print(res_str)
    # write_results(args,[avg.rmse,avg.absrel,avg.delta1,avg.delta2,avg.delta3],elapsed_time)
    
    print(f"[LOG] Network: {network_type_name}")
    print(f"[LOG] Size: {args.architecture_type}")
    print(f"[LOG] Trainable parameters: {num_params:,}")
    print(f"[LOG] Network memory weight: {model_weight} MB")
    print(f"[LOG] RMSE: {avg.rmse:.3f}")
    print(f"[LOG] AbsREL: {avg.absrel:.3f}")
    print(f"[LOG] Delta1: {avg.delta1:.3f}")
    print(f"[LOG] Delta2: {avg.delta2:.3f}")
    print(f"[LOG] Delta3: {avg.delta3:.3f}")
    print(f"[LOG] Test time: {elapsed_time:.2f}")

def test(args):
    inference_time(args)