import numpy as np
import matplotlib.pyplot as plt
import os


if __name__ == '__main__':

    log_dir = '/mnt/aertslab/USERS/Zezhong/HN_OUTCOME/output/log'

    # plot c-index
    cindex_np = np.load(os.path.join(log_dir, 'os_c-index.npy'))
    print('cindex:', cindex_np)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    #ax.set_aspect('equal')
    plt.plot(cindex_np, linewidth=3)
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, 'c-index.png'), format='png', dpi=600)
    plt.close()
