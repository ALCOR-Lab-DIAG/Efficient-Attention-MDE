from fvcore.nn import FlopCountAnalysis
from ptflops import get_model_complexity_info

from METER.globals import *
from METER.utils import *

def stats(args):
    formatter = TextFormatter()
    prep_str = formatter.format(
        f"Preparing getting stats of METER {args.architecture_type}", 
        color = 'blue', 
        style = ['bold'], 
        separator = True
    )
    print(prep_str)

    print(f"[LOG] Model will trained on {args.gpu} GPU")

    # Set-seed
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    np.random.seed(args.seed)

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

    model, _ = load_pretrained_model(
        model=model,
        path_weigths=args.checkpoint,
        device=args.gpu,
        do_pretrained=args.do_pretrained
    )

    # Model characteristics
    print(f"[Model] Model builded!")
    print(f"[Model] METER version: {args.architecture_type}")
    
    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print(f"[Model] Trainable parameters: {num_params:,}")

    model_weight = get_net_weight(model)
    print(f"[Model] Network memory weight: {model_weight} MB")

    input_tensor = torch.randn(1,3,args.rgb_img_res[1],args.rgb_img_res[2])
    flops = FlopCountAnalysis(model, input_tensor)
    flops.uncalled_modules_warnings(False)
    print(f"[Model] Networks FLOPs: {flops.total() / 1e9:.2f} G")

    macs, _ = get_model_complexity_info(
        model, 
        (3, args.rgb_img_res[1], args.rgb_img_res[2]),
        as_strings=False, 
        print_per_layer_stat=False,
        verbose=False
    )

    print(f"[Model] Networks MACs: {macs / 1e9:.2f} G")

    model_name = args.model_name
    size = "tiny" if "xxs" in args.architecture_type else "base" if "xs" in args.architecture_type else "large" if "s" in args.architecture_type else "unknown"
    dataset = args.dts_type
    opt_name = opt_type

    if opt_name is not None:
        full_name = model_name + "_" + size + "_" + opt_name + "_" + dataset
    else:
        full_name = model_name + "_" + size + "_" + dataset

    info_str = f"{full_name},{num_params},{flops.total()},{macs},{model_weight}\n"

    # Save to file
    with open("work/project/stats.txt", "a") as f:
        f.write(info_str)

def test(args):
    stats(args)