import argparse
import numpy as np
import os
import pickle
import torch

from torch import nn
from tqdm import tqdm

from PixelFormer.datasets.dataloader import *
from PixelFormer.networks.PixelFormer import *
from PixelFormer.utils import *

os.environ['MPLCONFIGDIR'] = "/work/project"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def extract_embeddings(model, dataloader, stage=3, max_samples=1000, device='cuda'):
    pool = nn.AdaptiveAvgPool2d((1,1))
    embeddings, count = [], 0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader.data,desc="Extract embeddings"):
            imgs = batch['image'].to(device)
            feats = model.backbone(imgs)[stage]
            vecs = pool(feats).view(feats.size(0), -1)
            embeddings.append(vecs.cpu().numpy())
            count += vecs.size(0)
            if count >= max_samples:
                break
    X = np.concatenate(embeddings, axis=0)[:max_samples]
    return X

def extractor(args):
    formatter = TextFormatter()
    prep_str = formatter.format(
        f"Preparing embedding extraction of PixelFormer {args.encoder[:-2]}", 
        color = 'blue', 
        style = ['bold'], 
        separator = True
    )
    print(prep_str)

    # 1) DataLoader
    val_loader = NewDataLoader(args, 'online_eval')
    
    # 2) Modello
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
    model.train()

    if args.checkpoint_path != '':
        if os.path.isfile(args.checkpoint_path):
            checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model'], strict=False)
            del checkpoint
        else:
            print(f"No checkpoint found at {args.checkpoint_path}")

    model.cuda()
    model.eval()

    # 3) Estrazione e salvataggio
    X = extract_embeddings(
        model, 
        val_loader, 
        stage=3,
        max_samples=len(val_loader.data),
        device=args.gpu
    )
    out_pickle = args.save_path + "emb.pkl"
    with open(out_pickle, 'wb') as f:
        pickle.dump({'embeddings': X}, f)
    print(f"[✔] Saved {X.shape[0]} embeddings in {out_pickle}")

if __name__=='__main__':
    main()
