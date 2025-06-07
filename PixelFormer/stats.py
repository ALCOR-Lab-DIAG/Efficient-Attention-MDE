import os
import torch

from fvcore.nn import FlopCountAnalysis
from ptflops import get_model_complexity_info

from PixelFormer.networks.PixelFormer import *
from PixelFormer.utils import *

def stats(args):
    formatter = TextFormatter()
    prep_str = formatter.format(
        f"Preparing getting stats of PixelFormer {args.encoder[:-2]}", 
        color = 'blue', 
        style = ['bold'], 
        separator = True
    )
    print(prep_str)

    torch.cuda.empty_cache()
    print(f"[LOG] Model will tested on {args.gpu} GPU")

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
    ).to(device=args.gpu)
    
    if args.checkpoint_path != '':
        if os.path.isfile(args.checkpoint_path):
            checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
            state_dict = {k.replace('module.', ''): v.cpu() for k, v in checkpoint['model'].items()}  # Rimuove DataParallel prefix
            model.load_state_dict(state_dict, strict=False)
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

    input_tensor = torch.randn(1,3,args.input_height,args.input_width)
    flops = FlopCountAnalysis(model, input_tensor)
    flops.uncalled_modules_warnings(False)
    print(f"[Model] Networks FLOPs: {flops.total() / 1e9:.2f} G")

    macs, _ = get_model_complexity_info(
        model, 
        (3, args.input_height, args.input_width),
        as_strings=False, 
        print_per_layer_stat=False,
        verbose=False
    )

    print(f"[Model] Networks MACs: {macs / 1e9:.2f} G")

    model_name = args.model_name.split("_")[0]
    size = args.encoder[:-2]
    dataset = args.dataset
    opt_name, where = (args.opt_type, args.opt_where) if args.opt_type != "" else ("", "")

    if opt_name != "":
        full_name = model_name + "_" + size + "_" + opt_name + "_" + where  + "_" + dataset
    else:
        full_name = model_name + "_" + size + "_" + dataset

    info_str = f"{full_name},{num_params},{flops.total()},{macs},{model_weight}\n"

    # Save to file
    with open("work/project/stats.txt", "a") as f:
        f.write(info_str)

if __name__ == '__main__':
    stats(args)