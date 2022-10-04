import argparse


def parse_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--manual_seed', default=1234, type=int, help='Mannual seed')    
    parser.add_argument('--proj_dir', default='/mnt/aertslab/USERS/Zezhong/HN_OUTCOME', type=str, help='Root path')

    parser.add_argument('--model_name', default='1_wce-dice-loss-0.001-160augment-hn-petct-pmhmdchuschum_gtvn.h5', type=str, help='test model name')
    parser.add_argument('--image_type', default='ct', type=str, help='(ct|pet')
    parser.add_argument('--hausdorff_percent', default=95, type=int, help='hausdorff_percent')
    parser.add_argument('--overlap_tolerance', default=5, type=int, help='overlap_tolerance')
    parser.add_argument('--surface_dice_tolerance', default=5, type=str, help='surface_dice_tolerance')
    parser.add_argument('--image_shape', default=(64, 160, 160), type=tuple, help='image shape')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--epochs', default=300, type=int, help='epochs')
    parser.add_argument('--initial_lr', default=0.001, type=int, help='initial learning rate')
    args = parser.parse_args()

    return args
