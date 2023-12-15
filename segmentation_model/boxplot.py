import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os


proj_dir = '/mnt/kannlab_rfa/Zezhong/HeadNeck/segmentation'

df = pd.read_csv(proj_dir + '/dice_maastro.csv')

fig = plt.figure(figsize=(8, 5))
ax  = fig.add_subplot(1, 1, 1)
#ax.set_aspect('equal')
sns.color_palette('deep')
#sns.violinplot(data=df, width=0.5, color='0.8', orient='v', linewidth=2)
sns.boxplot(data=df, color='0.7', orient='v', linewidth=2, width=0.4, saturation=1, whis=1)
sns.stripplot(data=df, jitter=True, zorder=1, orient='v', size=5, linewidth=1, edgecolor=None)
#sns.swarmplot(data=df, orient='v', size=7)

#plt.ylabel('DSC', fontweight='bold', fontsize=20)
plt.ylim([-0.05, 1])
ax.spines[['right', 'top']].set_visible(False)

#for axis in ['top', 'bottom', 'left', 'right']:
for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(2)  # change width
    ax.spines[axis].set_color('black')    # change color
    ax.tick_params(width=2, length=4)
#plt.xticks(['PN', 'P', 'N'], fontsize=30, fontweight='bold')
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=15, fontweight='bold')
#plt.xticks([]) 
#plt.legend(loc='lower right', prop={'size': 16, 'weight': 'bold'})
#plt.grid(True)
plt.savefig(proj_dir + '/box_plot.png', format='png', dpi=150, bbox_inches='tight')