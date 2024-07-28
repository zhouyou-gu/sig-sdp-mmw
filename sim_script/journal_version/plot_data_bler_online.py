import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sim_src.util import GET_LOG_PATH_FOR_SIM_SCRIPT

FONT_SIZE = 9
fig_width_px = 300
fig_height_px = 250
dpi = 100  # Typical screen DPI, adjust if necessary
fig_width_in = fig_width_px / dpi
fig_height_in = fig_height_px / dpi

plt.rc('font', family='serif')
plt.rc('mathtext', fontset='cm')
plt.rc('font', size=FONT_SIZE)  # Default font size
plt.rc('axes', titlesize=FONT_SIZE)  # Font size of the axes title
plt.rc('axes', labelsize=FONT_SIZE)  # Font size of the x and y labels
plt.rc('xtick', labelsize=FONT_SIZE)  # Font size of the tick labels
plt.rc('ytick', labelsize=FONT_SIZE)  # Font size of the tick labels
plt.rc('legend', fontsize=FONT_SIZE)  # Font size for legends


data_name_list = [r'Heuristic',r'Proposed']
current_dir = os.path.dirname(os.path.abspath(__file__))

# Plot the data
fig, ax = plt.subplots(1,1,)
axs = [ax]
fig.set_size_inches(fig_width_in, fig_height_in)  # 3.5 inches width, height adjusted to maintain aspect ratio

N_REPEAT = 100
# Plot settings
markers = ['o', 's', '^','+']  # Different markers for each line
linesty = [':', '-',(0, (3, 1, 1, 1)), (0, (5, 1)),'-','-','-']  # Different markers for each line
p_names = [r'a',r'b']
lines = []
labels = []
colors = ["#8A2BE2","#F97306","#20B2AA","#FF1493"]
bar_fill_patterns = ['///', '\\\\\\', '|||', '---']

cell_size_list = [10]
bars = []

for a in [0]:
    file_name_list = ["mgain","mmw"]
    for tt in range(len(file_name_list)):
        t = file_name_list[tt]
        datas = np.zeros((100,11))
        ratios = np.zeros((100,11))
        for ss in range(11):
            fname = "online"+"-"+t+"-"+str(ss)+"-"+str(150)+"-"+str(cell_size_list[a])+"-"+"75"
            data_file = os.path.join(current_dir, "sim_mmw_online","sim_mmw_online-2024-July-27-19-17-21-ail",fname)
            data = np.genfromtxt(data_file, delimiter=',')
            data = np.mean(data[:, 2:],axis=1)
            datas[:,ss] = data
            ratios[:,ss] = datas[:,ss]/datas[:,0]
            # cdf = np.arange(1, len(data) + 1) / len(data)
        index = np.arange(6)*2
        bar_width = 0.5
        b = axs[a].bar(index + tt * bar_width - (len(data_name_list)-1)*bar_width/2., np.mean(datas,axis=0)[0:11:2], bar_width, label=file_name_list[tt],color=colors[tt])
        bars.append(b)
    axs[a].set_position([0.2, 0.2+a*0.4, 0.775, 0.6])
    # Add labels and title
    # axs.text(0.75, 0.1, p_names[a], transform=axs[a].transAxes, fontsize=FONT_SIZE)
    axs[a].grid(True)
    axs[a].set_ylabel(r'Average error rates')

axs[0].set_xlabel(r'User mobility (meter/second)')

# # uu = 5*20
# # ll = 15*20
# axs[0].set_xlim(5*20, 15*20)
# axs[1].set_xlim(5*20, 15*20)
#
# # axs[1].set_xlim(uu, ll)
# # # axs[2].set_xlim(uu, ll)
# axs[a].set_xlim(-0.2, 2)
axs[0].set_xticks(np.arange(6)*2)
axs[0].set_xticklabels(np.arange(6)*2/10.)
#
# axs[1].set_xlim(9, 18)
# axs[1].set_xticks([9,12,15,18])
axs[0].set_ylim(1e-4, 5e-2)
# axs[0].set_yticks([0.8,0.9,1])
# axs[1].set_ylim(0, 1)
# axs[1].set_yticks([0.,0.5,1])
#
#
# axs[0].tick_params(axis='x',direction='in')
# axs[0].tick_params(axis='y',direction='in')
# axs[1].tick_params(axis='x',direction='in')
# axs[1].tick_params(axis='y',direction='in')

axs[0].set_yscale('log')
# axs[1].set_yscale('log')


# Add a legend
fig.legend(bars[0:len(data_name_list)], data_name_list ,fontsize=FONT_SIZE, loc='lower left', bbox_to_anchor=(0.2, 0.85, 0.775, 0.1), mode="expand",ncol = 2 ,borderaxespad=0.1,handlelength=1.5, handleheight= 1)
# axs[0].legend(fontsize=8, loc='lower left', bbox_to_anchor=(0, 1.02, 5,0.1), ncol=3,borderaxespad=0.)
# plt.subplots_adjust(left=0.175, right=0.95,bottom=0.175,top=0.95)


# Save the figure as a PDF
output_path = os.path.join(current_dir, os.path.splitext(os.path.basename(__file__))[0]) + '.pdf'

fig.savefig(output_path, format='pdf', bbox_inches='tight', pad_inches=0.)

# Display the plot
plt.show()