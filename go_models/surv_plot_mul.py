import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




def surv_plot_mul(out_dir, proj_dir, n_curves, fn):

    """plot survival curves
    """

    output_dir = os.path.join(out_dir, 'output')
    pro_data_dir = os.path.join(proj_dir, 'pro_data')
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    if not os.path.exists(pro_data_dir): os.mkdir(pro_data_dir)

    surv = pd.read_csv(os.path.join(pro_data_dir, 'surv.csv'))
    duration_index = np.load(os.path.join(pro_data_dir, 'duration_index.npy'))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    #ax.set_aspect('equal')
    
    # plot multiple patients
    #pat_list = [0, 10, 20, 40, 45, 50, 100, 110, 120, 130]
    for i in range(n_curves):
    #for i in pat_list:
        plt.plot(
            duration_index, 
            surv.iloc[:, i], 
            linewidth=2,
            label=str(i)
            )

    # plot mean survival curve 
    surv_mean = surv.mean(axis=1)
    plt.plot(
        duration_index, 
        surv_mean, 
        color='black',
        linestyle='dashed',
        linewidth=2, 
        label='mean', 
        )
    
    fig.suptitle('overall survival', fontweight='bold', fontsize=13)
    plt.ylabel('S(t | x)', fontweight='bold', fontsize=12)
    plt.xlabel('Time', fontweight='bold', fontsize=12)
    plt.xlim([0, 5000])
    plt.ylim([0, 1])
    ax.axhline(y=0, color='k', linewidth=2)
    ax.axhline(y=1, color='k', linewidth=2)
    ax.axvline(x=0, color='k', linewidth=2)
    ax.axvline(x=5000, color='k', linewidth=2)
    plt.xticks([0, 1000, 2000, 3000, 4000, 5000], fontsize=10, fontweight='bold')
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=10, fontweight='bold')
    plt.legend(loc='upper right', prop={'size': 10, 'weight': 'bold'})
    plt.grid(True)
    #plt.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=None)
    plt.savefig(os.path.join(output_dir, fn), format='png', dpi=600)
    #plt.show()
    plt.close()
    print('saved survival curves!')


