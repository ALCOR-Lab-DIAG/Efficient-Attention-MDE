import torch

from PixelFormer.compressor import compressor
from PixelFormer.extractor import extractor
from PixelFormer.globals import *
from PixelFormer.stats import *
from PixelFormer.test import test
from PixelFormer.train import train
from PixelFormer.utils import *

def main():
    # Argument parser
    args = init_arg_parser()
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
        
    # Starting train
    if args.mode == "train":
        try:
            train(args)
        except KeyboardInterrupt:
            print("Training interrupted by user")
            torch.cuda.empty_cache()
        except Exception as e:
            print(e)
            send_email("PixelFormer", args.encoder[:-2], args.opt_type, args.opt_where, error=True, error_msg=e)
            torch.cuda.empty_cache()
            raise  # Rilancia l'eccezione per debugging o ulteriori gestioni

    # Starting test
    elif args.mode == "test":
        test(args)
    elif args.mode == "analysis":
        extractor(args)
    elif args.mode == "stats":
        stats(args)
    elif args.mode == "compress":
        compressor(args)
    else:
        device_str = formatter.format(
            "[!!!] Choose between train or test mode for the PixelFormer network.\n[!!!] Adjust the --mode flag in the arguments file located in the /config directory.",  
            color = 'red', 
            style = ['bold'], 
            separator = False
        )
        print(device_str)
    
if __name__ == '__main__':
    main()