import copy
import torch

from pathlib import Path
from torch.nn.utils import prune
from torch.quantization import quantize_dynamic

from NeWCRFs.globals import *
from NeWCRFs.networks.NewCRFDepth import *
from NeWCRFs.utils import *

def apply_pruning(model: torch.nn.Module, amount: float):
    to_prune = [
        (m, 'weight')
        for m in model.modules()
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))
    ]
    prune.global_unstructured(to_prune,
                              pruning_method=prune.L1Unstructured,
                              amount=amount)
    for m, _ in to_prune:
        prune.remove(m, 'weight')
    return model

def apply_quantization(model: torch.nn.Module):
    return quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

def compressor(args):
    model = NewCRFDepth(
        version = args.encoder,
        inv_depth = False,
        max_depth = args.max_depth,
        pretrained = None,
        opt_type = args.opt_type,
        opt_where = args.opt_where
    )
    
    if args.checkpoint_path != '':
        if os.path.isfile(args.checkpoint_path):
            checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
            state_dict = {k.replace('module.', ''): v.cpu() for k, v in checkpoint['model'].items()}  # Rimuove DataParallel prefix
            model.load_state_dict(state_dict, strict=False)
            model.to('cpu')  # Sposta esplicitamente il modello su CPU
            del checkpoint
        else:
            print(f"No checkpoint found at {args.checkpoint_path}")

    model_name = args.model_name.split("_")[0]
    size = args.encoder[:-2]
    dataset = args.dataset
    full_name = model_name + "_" + size + "_" + dataset

    # PRUNING
    pruned = copy.deepcopy(model)
    pruned = apply_pruning(pruned, 0.7)
    pruned_name = full_name + "_pruned"
    pruned_path = "work/data/ssd_pretrained/checkpoints/" + pruned_name
    torch.save(pruned.state_dict(), pruned_path)
    print(f"[NewCRFs] saved pruned model → {pruned_path}")

    # QUANTIZATION
    quant = copy.deepcopy(model)
    quant = apply_quantization(quant)
    quant_name = full_name + "_quantized"
    quant_path = "work/data/ssd_pretrained/checkpoints/" + quant_name
    torch.save(quant.state_dict(), quant_path)
    print(f"[NewCRFs] saved quantized model → {quant_path}")