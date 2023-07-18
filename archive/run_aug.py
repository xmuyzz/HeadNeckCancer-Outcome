


from go_models.save_augmentation import save_augmentation
from go_models.DataLoader_Cox import DataLoader_Cox


if __name__ == '__main__':

    data_dir = '/mnt/aertslab/DATA/HeadNeck/HN_PETSEG/curated'
    proj_dir = '/mnt/HDD_6TB/HN_Outcome'
    aimlab_dir = '/mnt/aertslab/USERS/Zezhong/HN_OUTCOME'

    #save_augmentation(proj_dir, aimlab_dir)

    dl_train, dl_tune, dl_val = DataLoader_Cox(proj_dir)
