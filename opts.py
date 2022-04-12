import argparse


def parse_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--manual_seed', default=1234, type=int, help='Mannual seed')    
    # path
    parser.add_argument('--proj_dir', default='/mnt/aertslab/USERS/Zezhong/HN_OUTCOME', type=str, help='Root path')
    parser.add_argument('--data_dir', default='/mnt/aertslab/DATA/HeadNeck/HN_PETSEG/curated', type=str, help='data path')
    parser.add_argument('--prepro_data', default='prepro_data', type=str, help='Preprocessed image path')
    parser.add_argument('--label', default='label', type=str, help='Label path')
    parser.add_argument('--output', default='output', type=str, help='Results output path')
    parser.add_argument('--pro_data', default='pro_data', type=str, help='Processed data path')
    parser.add_argument('--model', default='model', type=str, help='Results output path')
    parser.add_argument('--log', default='log', type=str, help='Log data path')
    parser.add_argument('--train_folder', default='train', type=str, help='Train results path')
    parser.add_argument('--val_folder', default='val', type=str, help='Validation results path')
    parser.add_argument('--test_folder', default='test', type=str, help='Test results path')
    
    # data preprocessing
    parser.add_argument('--_outcome_model', default='overall_survival', type=str, 
                        help='outcome model (overall_survival|local_control|distant_control')
    parser.add_argument('--new_spacing', default=(1, 1, 3), type=float, help='new spacing size')
    parser.add_argument('--data_exclude', default=None, type=str, help='Exclude data')
    parser.add_argument('--crop_shape', default=[192, 192, 100], type=float, help='Crop image shape')
    parser.add_argument('--run_type', default=None, type=str, help='Used run type (train|val|test|tune)')
    parser.add_argument('--input_channel', default=3, type=int, help='Input channel (1|3)')
    parser.add_argument('--norm_type', default='np_clip', type=str, help='image normalization (np_clip|np_linear')
    parser.add_argument('--slice_range', default=range(17, 83), type=int, help='Axial slice range')
    parser.add_argument('--interp', default='linear', type=str, help='Interpolation for respacing')

    # train model
    parser.add_argument('--tumor_type', default='primary_node', type=str, help='(primary_node|primary|node')
    parser.add_argument('--input_data_type', default='masked_img', type=str, help='(masked_img|raw_img')
    parser.add_argument('--i_kfold', default=0, type=int, help='(0|1|2|3|4)')
    parser.add_argument('--cnn_name', default='resnet101', type=str, help='resnet (18|34|50|152|200)')
    parser.add_argument('--cox_model_name', default='LogisticHazard', type=str, help='cox model')
    parser.add_argument('--batch_size', default=8, type=int, help='Batch size')
    parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
    parser.add_argument('--epoch', default=3, type=int, help='Epoch')
    parser.add_argument('--in_channels', default=1, type=int, help='Input channels')
    parser.add_argument('--num_durations', default=20, type=int, help='Number of durations.')
    parser.add_argument('--activation', default='sigmoid', type=str, help='Activation function')
    parser.add_argument('--loss_function', default='binary_crossentropy', type=str, help='loss function')
    parser.add_argument('--optimizer_function', default='adam', type=str, help='optmizer function')
    parser.add_argument('--run_model', default='EffNetB4', type=str, help='run model')
    parser.add_argument('--input_shape', default=(192, 192, 3), type=int, help='Input shape')
    parser.add_argument('--freeze_layer', default=None, type=str, help='Freeze layer to train')

    # evalute model                        
    parser.add_argument('--thr_img', default=0.5, type=float, help='threshold to decide positive class')
    parser.add_argument('--n_bootstrap', default=1000, type=int, help='bootstrap to calcualte 95% CI of AUC')
    parser.add_argument('--score_type', default='os_surv', type=str, help='(median|3yr_surv|5yr_surv|os_surv)')    
    parser.add_argument('--load_model', default='model', type=str, help='(model|weights')
    parser.add_argument('--saved_model', default='EffNetB4', type=str, help='saved model name')

    # others 
    parser.add_argument('--augmentation', action='store_true', help='If true, augmentation is performed.')
    parser.set_defaults(augmentation=True)
    parser.add_argument('--train', action='store_true', help='If true, training is performed.')
    parser.set_defaults(train=True)
    parser.add_argument('val', action='store_true', help='If true, validation is performed.')
    parser.set_defaults(val=False)
    parser.add_argument('--test', action='store_true', help='If true, test is performed.')
    parser.set_defaults(test=False)
    
    args = parser.parse_args()

    return args
