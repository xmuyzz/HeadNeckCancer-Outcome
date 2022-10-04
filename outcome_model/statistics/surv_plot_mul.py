import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




def surv_plot_mul(proj_dir, n_curves, surv_fn, choose_patient=True,
                  plot_median=False):

    """plot survival curves
    """

    output_dir = os.path.join(proj_dir, 'output')
    pro_data_dir = os.path.join(proj_dir, 'pro_data')
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    if not os.path.exists(pro_data_dir): os.mkdir(pro_data_dir)

    surv = pd.read_csv(os.path.join(pro_data_dir, 'surv.csv'))
    duration_index = np.load(os.path.join(pro_data_dir, 'duration_index.npy'))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    #ax.set_aspect('equal')
    
    # plot multiple patients
    if choose_patient:
        pat_list = [190, 14, 20, 10, 150, 120, 140, 145, 150, 100]
    else: 
        pat_list = range(n_curves)
    for i in pat_list:
        plt.plot(
            duration_index, 
            surv.iloc[:, i], 
            linewidth=2,
            label=str(i)
            )
    # plot median survival curve
    if plot_median:
        surv_median = surv.median(axis=1)
        plt.plot(
            duration_index, 
            surv_median, 
            color='black',
            linestyle='dashed',
            linewidth=2, 
            label='median', 
            ) 
    fig.suptitle('overall survival', fontweight='bold', fontsize=13)
    plt.ylabel('Survial Probability', fontweight='bold', fontsize=12)
    plt.xlabel('Time (days)', fontweight='bold', fontsize=12)
    plt.xlim([0, 5000])
    plt.ylim([0, 1])
    ax.axhline(y=0, color='k', linewidth=2)
    ax.axhline(y=1, color='k', linewidth=2)
    ax.axvline(x=0, color='k', linewidth=2)
    ax.axvline(x=5000, color='k', linewidth=2)
    plt.xticks([0, 1000, 2000, 3000, 4000, 5000], fontsize=12, fontweight='bold')
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=12, fontweight='bold')
    #plt.legend(loc='lower left', prop={'size': 10, 'weight': 'bold'})
    plt.grid(True)
    #plt.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=None)
    surv_curv_fn = 'surv_curv_' + surv_fn.split('_surv')[0] + '.png'
    plt.savefig(os.path.join(output_dir, surv_curv_fn), format='png', dpi=600)
    #plt.show()
    plt.close()
    print('saved survival curves!')


if __name__ == '__main__':

    proj_dir = '/mnt/aertslab/USERS/Zezhong/HN_OUTCOME'
    surv_fn = 'resnet101_20_0.0001_surv.csv'
    n_curves = 5
    

    surv_plot_mul(
        proj_dir, 
        n_curves, 
        surv_fn
        )
