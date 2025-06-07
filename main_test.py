import torch

from PixelFormer.globals import *
from PixelFormer.kitti_test import kitti_test
from PixelFormer.test import test
from PixelFormer.utils import *

def main_test():
    # Argument parser
    args = test_arg_parser()
    formatter = TextFormatter()
    if args.opt_type is not None:
        intro = f"PixelFormer {str(args.opt_type)}"
    else:
        intro = "PixelFormer Baseline"
    
    intro_str = formatter.format(
        intro,
        color = 'blue', 
        style = ['bold'], 
        separator = True
    )
    print(intro_str)
    print_args(args)

    # Device setup
    device_str = formatter.format(
        "Setting device", 
        color = 'blue', 
        style = ['bold'], 
        separator = True
    )
    print(device_str)
    device = hardware_check(args.mode)
    args.gpu = device

    # kitti_test(args)
    test(args)

if __name__ == '__main__':
    main_test()