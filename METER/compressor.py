import copy
import torch

from pathlib import Path
from torch.nn.utils import prune
from torch.quantization import quantize_dynamic

from METER.globals import *
from METER.networks.METER import *
from METER.utils import *

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
    # ricrea il modello su CPU per compressione
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

    model_name = args.model_name
    size = "tiny" if "xxs" in args.architecture_type else "base" if "xs" in args.architecture_type else "large" if "s" in args.architecture_type else "unknown"
    dataset = args.dts_type
    full_name = model_name + "_" + size + "_" + dataset

    # PRUNING
    pruned = copy.deepcopy(model)
    pruned = apply_pruning(pruned, 0.7)
    pruned_name = full_name + "_pruned"
    pruned_path = "work/data/ssd_pretrained/checkpoints/" + pruned_name
    torch.save(pruned.state_dict(), pruned_path)
    print(f"[METER] saved pruned model → {pruned_path}")

    # QUANTIZATION
    quant = copy.deepcopy(model)
    quant = apply_quantization(quant)
    quant_name = full_name + "_quantized"
    quant_path = "work/data/ssd_pretrained/checkpoints/" + quant_name
    torch.save(quant.state_dict(), quant_path)
    print(f"[METER] saved quantized model → {quant_path}")
