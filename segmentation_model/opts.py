import argparse


def parse_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--manual_seed', default=1234, type=int, help='Mannual seed')    
    parser.add_argument('--proj_dir', default='/mnt/aertslab/USERS/Zezhong/HN_OUTCOME', type=str, help='Root path')
    parser.add_argument('--model_name', default='1wce_dice-loss-160augment-hn-petct-pmhcon_gtvp_LR0.001opc_contrast20220406-0936.h5', type=str, help='test model name')
    parser.add_argument('--image_type', default='ct', type=str, help='(ct|pet')
    parser.add_argument('--hausdorff_percent', default=95, type=int, help='hausdorff_percent')
    parser.add_argument('--overlap_tolerance', default=5, type=int, help='overlap_tolerance')
    parser.add_argument('--surface_dice_tolerance', default=5, type=str, help='surface_dice_tolerance')
     
    args = parser.parse_args()

    return args
