
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def model_calibration(surv, times, events):

    fig, ax = plt.subplots()
    observed = (times <= 1095) & (events == 1)
    predict = surv[1095]
    df_calib = pd.DataFrame({'pred': predict, 'true': observed})
    df_calib['bin'] = pd.qcut(df_calib['pred'], q=5, duplicates='drop')
    calib_data = df_calib.groupby('bin').agg('mean')
    df_calib['count'] = 1
    bin_counts = df_calib.groupby('bin').agg('count')['count']
    df_calib['abs_diff'] = np.abs(df_calib['pred'] - (1 - df_calib['true']))
    weighted_abs_diff = df_calib.groupby('bin').apply(lambda x: x['abs_diff'] * x['count'])
    ECE = weighted_abs_diff.sum() / df_calib['count'].sum()
    print(f'Expected Calibration Error (ECE): {ECE}')
    ax.plot(calib_data['pred'], calib_data['true'], marker='o', label=f'Time: {1095} days')
    ax.plot([0, 1], [1, 0], 'k--', label='Perfectly calibrated')
    ax.set_xlabel('Predicted probability')
    ax.set_ylabel('Observed frequency')
    plt.show()