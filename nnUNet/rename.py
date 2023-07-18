import os


def rename(data_dir):
    for fn in sorted(os.listdir(data_dir)):
        print(fn)
        ID = fn.split('_')[1]
        ID = int(ID) + 1
        ID = f'{ID:03}'
        fn_new = 'OPC_' + str(ID) + '_0000.nii.gz'
        dir_old = os.path.join(data_dir, fn)
        dir_new = os.path.join(data_dir, fn_new)
        os.rename(dir_old, dir_new)
    print('rename complete!')


if __name__ == '__main__':

    data_dir = '/mnt/aertslab/USERS/Zezhong/HN_OUTCOME/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task501_PN/imagesTr'
    rename(data_dir)
