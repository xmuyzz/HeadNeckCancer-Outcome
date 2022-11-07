import argparse


def parse_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--manual_seed', default=1234, type=int, help='Mannual seed')    
    parser.add_argument('--proj_dir', default='/mnt/aertslab/USERS/Zezhong/HN_OUTCOME', type=str, help='Root path')
    parser.add_argument('--model_name', default='best_dice_model.h5', type=str, help='saved model')
    parser.add_argument('--image_type', default='ct', type=str, help='ct|pet')
    parser.add_argument('--image_shape', default=(64, 160, 160), type=str, help='image shape')
    parser.add_argument('--input_shape', default=(1, 64, 160, 160), type=str, help='input shape')
    parser.add_argument('--hausdorff_percent', default=95, type=int, help='hausdorff_percent')
    parser.add_argument('--overlap_tolerance', default=5, type=int, help='overlap_tolerance')
    parser.add_argument('--surface_dice_tolerance', default=5, type=str, help='surface_dice_tolerance')
    parser.add_argument('--batch_size', default=6, type=int, help='batch size')
    parser.add_argument('--epochs', default=300, type=int, help='epochs')
    parser.add_argument('--initial_lr', default=0.001, type=int, help='initial learning rate')
    parser.add_argument('--test_set', default='test2', type=str, help='test1 | test2')
    parser.add_argument('--n_labels', default=2, type=int, help='n_labels')
    parser.add_argument('--plot_img', action='store_true', help='If true, plot_img is performed.')
    parser.set_defaults(plot_img=False)
    
    args = parser.parse_args()
    return args
