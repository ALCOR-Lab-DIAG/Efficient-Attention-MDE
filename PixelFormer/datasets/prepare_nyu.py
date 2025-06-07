from PixelFormer.datasets.data_utils import *
from PixelFormer.globals import global_var

def prepare_nyu():
    nyu_file = download_nyu_eval(
        nyu_file = global_var['nyu_test_file']
    )

    extract_nyu_eval(
        dataset_file = nyu_file,
        split_reference = global_var['nyu_split_ref'],
        output_dir = global_var['nyu_test_dir']
    )

    download_nyu_train(
        url = global_var['nyu_train_link'],
        output_dir = global_var['nyu_dir']
    )