import torch

from METER.compressor import compressor
from METER.globals import *
from METER.utils import *
from METER.train import train
from METER.test import test
from METER.stats import stats

def main():
    # Argument parser
    args = init_arg_parser()
    formatter = TextFormatter()
    if "_" in args.network_type:
        intro = f"{format_string(args.network_type)}"
    else:
        intro = f"{args.network_type} Baseline"
    
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
    device = hardware_check()
    args.gpu = device
            
    # Starting train
    if args.do_train:
        try:
            train(args)
        except KeyboardInterrupt:
            print("Training interrupted by user")
            torch.cuda.empty_cache()
        except Exception as e:
            print(e)
            if "_" in args.network_type:
                opt_type = args.network_type.split("_")[0]
            else:
                opt_type = ""
                
            send_email("METER", args.architecture_type, opt_type, error=True, error_msg=e)
            torch.cuda.empty_cache()
            raise  # Rilancia l'eccezione per debugging o ulteriori gestioni
    # Starting test
    elif args.mode == "stats":
        stats(args)
    elif args.mode == "compress":
        compressor(args)
    else:
        try:
            test(args)
        except KeyboardInterrupt:
            print("Test interrupted by user")
            torch.cuda.empty_cache()
        except Exception as e:
            print(e)
            torch.cuda.empty_cache()
            raise  # Rilancia l'eccezione per debugging o ulteriori gestioni
    
if __name__ == '__main__':
    main()