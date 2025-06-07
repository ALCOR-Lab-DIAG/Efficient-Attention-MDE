import torch
import torch.nn as nn

from PixelFormer.networks.BCP import *
from PixelFormer.networks.PQI import PSP
from PixelFormer.networks.SAM import SAM
from PixelFormer.networks.meta_SAM import MetaSAM
from PixelFormer.networks.moh_SAM import MoHSAM
from PixelFormer.networks.pyra_SAM import PyraSAM
from PixelFormer.networks.swin_transformer import SwinTransformer
from PixelFormer.networks.meta_swin_transformer import MetaSwinTransformer
from PixelFormer.networks.moh_swin_transformer import MoHSwinTransformer
from PixelFormer.networks.pyra_swin_transformer import PyraSwinTransformer
from PixelFormer.networks.upsampler import *
from PixelFormer.networks.caformer import caformer_s18, convformer_s18, identityformer_s12, randformer_s12

class PixelFormer(nn.Module):

    def __init__(self, version=None, inv_depth=False, pretrained=None, 
                    frozen_stages=-1, min_depth=0.1, max_depth=100.0, opt_type = None, opt_where = None, **kwargs):
        super().__init__()

        self.inv_depth = inv_depth
        self.with_auxiliary_head = False
        self.with_neck = False
        self.opt_type = opt_type
        self.opt_where = opt_where

        norm_cfg = dict(type='BN', requires_grad=True)
        # norm_cfg = dict(type='GN', requires_grad=True, num_groups=8)

        window_size = int(version[-2:])

        if version[:-2] == 'base':
            embed_dim = 128
            depths = [2, 2, 18, 2]
            num_heads = [4, 8, 16, 32]
            in_channels = [128, 256, 512, 1024]
            if self.opt_type == "moh":
                shared_head = [4, 8, 4, 5] 
                routed_head = [0, 0, 8, 16]
        elif version[:-2] == 'large':
            embed_dim = 192
            depths = [2, 2, 18, 2]
            num_heads = [6, 12, 24, 48]
            in_channels = [192, 384, 768, 1536]
            if self.opt_type == "moh":
                shared_head = [6, 12, 6, 8] 
                routed_head = [0, 0, 12, 24]
        elif version[:-2] == 'tiny':
            embed_dim = 96
            depths = [2, 2, 6, 2]
            num_heads = [3, 6, 12, 24]
            in_channels = [96, 192, 384, 768]
            if self.opt_type == "moh":
                shared_head = [3, 6, 3, 4] 
                routed_head = [0, 0, 6, 12]

        if self.opt_type == "moh" and (self.opt_where == "enc" or self.opt_where == "full"):
            backbone_cfg = dict(
                embed_dim=embed_dim,
                depths=depths,
                num_heads=num_heads,
                window_size=window_size,
                ape=False,
                drop_path_rate=0.3,
                patch_norm=True,
                use_checkpoint=False,
                frozen_stages=frozen_stages,
                shared_head=shared_head,
                routed_head=routed_head
            )
        else:
            backbone_cfg = dict(
                embed_dim=embed_dim,
                depths=depths,
                num_heads=num_heads,
                window_size=window_size,
                ape=False,
                drop_path_rate=0.3,
                patch_norm=True,
                use_checkpoint=False,
                frozen_stages=frozen_stages
            )

        embed_dim = 512
        decoder_cfg = dict(
            in_channels=in_channels,
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=embed_dim,
            dropout_ratio=0.0,
            num_classes=32,
            norm_cfg=norm_cfg,
            align_corners=False
        )

        if self.opt_type == "meta" and (self.opt_where == "enc" or self.opt_where == "full"):
            print("[LOG] Building META-Swin encoder")
            self.backbone = MetaSwinTransformer(**backbone_cfg)
        elif self.opt_type == "moh" and (self.opt_where == "enc" or self.opt_where == "full"):
            print("[LOG] Building MoH-Swin encoder")
            self.backbone = MoHSwinTransformer(**backbone_cfg)
        elif self.opt_type == "ca" and (self.opt_where == "enc" or self.opt_where == "full"):
            print("[LOG] Building CAformer encoder")
            self.backbone = caformer_s18(
                depths = depths,
                dims = in_channels
            )
        elif self.opt_type == "id" and (self.opt_where == "enc" or self.opt_where == "full"):
            print("[LOG] Building Identityformer encoder")
            self.backbone = identityformer_s12(
                depths = depths,
                dims = in_channels
            )
        elif self.opt_type == "rand" and (self.opt_where == "enc" or self.opt_where == "full"):
            print("[LOG] Building Randformer encoder")
            self.backbone = randformer_s12(
                depths = depths,
                dims = in_channels
            )
        elif self.opt_type == "conv" and (self.opt_where == "enc" or self.opt_where == "full"):
            print("[LOG] Building Convformer encoder")
            self.backbone = convformer_s18(
                depths = depths,
                dims = in_channels
            )
        elif self.opt_type == 'pyra' and (self.opt_where == "enc" or self.opt_where == "full"):
            print("[LOG] Building PYRA-Swin encoder")
            self.backbone = PyraSwinTransformer(**backbone_cfg)
        else:
            print("[LOG] Building standard encoder")
            self.backbone = SwinTransformer(**backbone_cfg)
        
        # self.backbone = SwinTransformer(**backbone_cfg)
        
        v_dim = decoder_cfg['num_classes']*4
        win = 7
        sam_dims = [128, 256, 512, 1024]
        v_dims = [64, 128, 256, embed_dim]

        if self.opt_type == "meta" and (self.opt_where == "dec" or self.opt_where == "full"):
            print("[LOG] Building META-SAM decoder")
            self.sam4 = MetaSAM(input_dim=in_channels[3], embed_dim=sam_dims[3], window_size=win, v_dim=v_dims[3], num_heads=32)
            self.sam3 = MetaSAM(input_dim=in_channels[2], embed_dim=sam_dims[2], window_size=win, v_dim=v_dims[2], num_heads=16)
            self.sam2 = MetaSAM(input_dim=in_channels[1], embed_dim=sam_dims[1], window_size=win, v_dim=v_dims[1], num_heads=8)
            self.sam1 = MetaSAM(input_dim=in_channels[0], embed_dim=sam_dims[0], window_size=win, v_dim=v_dims[0], num_heads=4)
        elif self.opt_type == "pyra" and (self.opt_where == "dec" or self.opt_where == "full"):
            print("[LOG] Building PYRA-SAM decoder")
            self.sam4 = PyraSAM(input_dim=in_channels[3], embed_dim=sam_dims[3], window_size=win, v_dim=v_dims[3], num_heads=32)
            self.sam3 = PyraSAM(input_dim=in_channels[2], embed_dim=sam_dims[2], window_size=win, v_dim=v_dims[2], num_heads=16)
            self.sam2 = PyraSAM(input_dim=in_channels[1], embed_dim=sam_dims[1], window_size=win, v_dim=v_dims[1], num_heads=8)
            self.sam1 = PyraSAM(input_dim=in_channels[0], embed_dim=sam_dims[0], window_size=win, v_dim=v_dims[0], num_heads=4)
        elif self.opt_type == "moh" and (self.opt_where == "dec" or self.opt_where == "full"):
            print("[LOG] Building MoH-SAM decoder")
            self.sam4 = MoHSAM(input_dim=in_channels[3], embed_dim=sam_dims[3], window_size=win, v_dim=v_dims[3], num_heads=32, shared_head=5, routed_head=16)
            self.sam3 = MoHSAM(input_dim=in_channels[2], embed_dim=sam_dims[2], window_size=win, v_dim=v_dims[2], num_heads=16, shared_head=4, routed_head=8)
            self.sam2 = MoHSAM(input_dim=in_channels[1], embed_dim=sam_dims[1], window_size=win, v_dim=v_dims[1], num_heads=8, shared_head=8, routed_head=0)
            self.sam1 = MoHSAM(input_dim=in_channels[0], embed_dim=sam_dims[0], window_size=win, v_dim=v_dims[0], num_heads=4, shared_head=4, routed_head=0)
        else:
            print("[LOG] Building standard decoder")
            self.sam4 = SAM(input_dim=in_channels[3], embed_dim=sam_dims[3], window_size=win, v_dim=v_dims[3], num_heads=32)
            self.sam3 = SAM(input_dim=in_channels[2], embed_dim=sam_dims[2], window_size=win, v_dim=v_dims[2], num_heads=16)
            self.sam2 = SAM(input_dim=in_channels[1], embed_dim=sam_dims[1], window_size=win, v_dim=v_dims[1], num_heads=8)
            self.sam1 = SAM(input_dim=in_channels[0], embed_dim=sam_dims[0], window_size=win, v_dim=v_dims[0], num_heads=4)
        
        self.decoder = PSP(**decoder_cfg)
        self.disp_head1 = DispHead(input_dim=sam_dims[0])

        self.bcp = BCP(max_depth=max_depth, min_depth=min_depth)

        if self.opt_type == "meta" or self.opt_type == "pyra" or self.opt_type is None:
            self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        print(f"[LOG] Load pretrained SwinTX encoder from: {pretrained}")
        self.backbone.init_weights(pretrained=pretrained)
        self.decoder.init_weights()
        if self.with_auxiliary_head:
            if isinstance(self.auxiliary_head, nn.ModuleList):
                for aux_head in self.auxiliary_head:
                    aux_head.init_weights()
            else:
                self.auxiliary_head.init_weights()

    def forward(self, imgs):

        enc_feats = self.backbone(imgs)
        if self.with_neck:
            enc_feats = self.neck(enc_feats)

        q4 = self.decoder(enc_feats)

        q3 = self.sam4(enc_feats[3], q4)
        q3 = nn.PixelShuffle(2)(q3)
        q2 = self.sam3(enc_feats[2], q3)
        q2 = nn.PixelShuffle(2)(q2)
        q1 = self.sam2(enc_feats[1], q2)
        q1 = nn.PixelShuffle(2)(q1)
        q0 = self.sam1(enc_feats[0], q1)
        bin_centers = self.bcp(q4)
        f = self.disp_head1(q0, bin_centers, 4)

        return f
