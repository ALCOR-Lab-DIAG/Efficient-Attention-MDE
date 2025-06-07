import os

os.environ['MPLCONFIGDIR'] = "/work/project"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTHONASHSEED"] = "42"

global_var = {
    # Dataset
    'dts_name': "nyu",
    'nyu_dir' : "work/project/PixelFormer/datasets/nyu_depth_v2/",
    
    'nyu_test_link' : "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat",
    'nyu_test_file' : "work/project/PixelFormer/datasets/nyu_depth_v2_labeled.mat",
    'nyu_split_ref' : "work/project/PixelFormer/datasets/splits.mat",
    'nyu_test_dir' : "work/project/PixelFormer/datasets/nyu_depth_v2/official_splits/",
    
    'nyu_train_link' : "https://drive.google.com/file/d/1AysroWpfISmm-yRFGBgFTrLy6FjQwvwP/view?usp=sharing",
    'nyu_train_dir' : "work/project/PixelFormer/datasets/nyu_depth_v2/sync/",
}