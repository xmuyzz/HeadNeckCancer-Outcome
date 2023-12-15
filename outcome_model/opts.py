import argparse

def parse_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--manual_seed', default=1234, type=int, help='Mannual seed')    
    parser.add_argument('--proj_dir', default='/mnt/kannlab_rfa/Zezhong/HeadNeck/outcome', type=str, help='project path')
    parser.add_argument('--data_dir', default='/home/xmuyzz/data/HNSCC/outcome', type=str, help='data path')

    # data preprocessing
    parser.add_argument('--clinical_data_file', default='HN_clinical_meta_data.csv', type=str, help='label')
    parser.add_argument('--save_label', default='label.csv', type=str, help='save label')
    parser.add_argument('--input_channel', default=1, type=int, help='Input channel (1|3)')
    
    # model 
    parser.add_argument('--cnn_name', default='DenseNet', type=str, help='resnet (18|34|50|152|200)')
    parser.add_argument('--model_depth', default=169, type=str, help='resnet (18|34|50|152|200)')
    parser.add_argument('--cox', default='LogisticHazard', type=str, help='CoxPH | PCHazard | DeepHit | LogisticHazard | MTLR')
    parser.add_argument('--batch_size', default=4, type=int, help='batch size')
    parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
    parser.add_argument('--epoch', default=100, type=int, help='Epoch')
    parser.add_argument('--in_channels', default=1, type=int, help='Input channels')
    parser.add_argument('--num_durations', default=10, type=int, help='Number of durations')
    parser.add_argument('--n_clinical', default=30, type=int, help='Number of clinical features')
    parser.add_argument('--activation', default='sigmoid', type=str, help='Activation function')
    parser.add_argument('--loss_function', default='binary_crossentropy', type=str, help='loss function')
    parser.add_argument('--optimizer_function', default='adam', type=str, help='optmizer function')
    parser.add_argument('--input_shape', default=(1, 160, 160), type=int, help='Input shape')
    parser.add_argument('--freeze_layer', default=None, type=str, help='Freeze layer to train')

    # train
    parser.add_argument('--task', default='Task055', type=str, help='Task001')
    parser.add_argument('--gauss_prob', default=0.5, type=str, help='0.1')
    parser.add_argument('--rot_prob', default=0.2, type=str, help='0.1')
    parser.add_argument('--flip_prob', default=0.2, type=str, help='0.1')
    parser.add_argument('--target_c_index', default=0.69, type=str, help='0.7')
    parser.add_argument('--target_loss', default=0.09, type=str, help='0.1')
    parser.add_argument('--img_size', default='full', type=str, help='(full|bbox')
    parser.add_argument('--img_type', default='attn122', type=str, help='(mask|raw|attn123|attn122')
    parser.add_argument('--tumor_type', default='pn', type=str, help='(primary_node|primary|node')
    parser.add_argument('--surv_type', default='efs', type=str, help='rfs|os|lc|dc')

    # test 
    parser.add_argument('--data_set', default='ts', type=str, help='Used run type (va|ts|tx_maastro|tx_bwh)') 
    parser.add_argument('--eval_model', default='best_cindex_model', type=str, help='best_loss_model|best_cindex_model')                     

    # others 
    parser.add_argument('--load_train_data', action='store_true', help='If true, load data is performed.')
    parser.set_defaults(load_train_data=False)
    parser.add_argument('--load_model', action='store_true', help='If true, load model is performed.')
    parser.set_defaults(load_model=True)
    parser.add_argument('--train', action='store_true', help='If true, training is performed.')
    parser.set_defaults(train=False)
    parser.add_argument('--test', action='store_true', help='If true, test is performed.')
    parser.set_defaults(test=True)
    
    args = parser.parse_args()

    return args
