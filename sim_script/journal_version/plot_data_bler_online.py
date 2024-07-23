import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sim_src.util import GET_LOG_PATH_FOR_SIM_SCRIPT

FONT_SIZE = 9
fig_width_px = 350
fig_height_px = 225
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


data_name_list = [r'MMW-$150$',r'MMW-$300$',"ADMM"]
file_name_list = ["mmw150","mmw300","scs"]
current_dir = os.path.dirname(os.path.abspath(__file__))

# Plot the data
fig, axs = plt.subplots(1,1,)
fig.set_size_inches(fig_width_in, fig_height_in)  # 3.5 inches width, height adjusted to maintain aspect ratio

N_REPEAT = 100
# Plot settings
markers = ['o', 's', '^','+']  # Different markers for each line
linesty = [':', '-',(0, (3, 1, 1, 1)), (0, (5, 1)),'-','-','-']  # Different markers for each line
p_names = [r'a',r'b']
lines = []
labels = []
for t in [5,10,15]:
    datas = []
    for ss in range(10):
        fname = "online-"+str(ss)+"-"+str(t)+"-"+"75"
        data_file = os.path.join(current_dir, "sim_mmw_online","sim_mmw_online-2024-July-23-23-52-19-ail",fname)
        data = np.genfromtxt(data_file, delimiter=',')
        data = data[:, 3:].ravel()
        data = np.mean(data)
        datas.append(data)
        # cdf = np.arange(1, len(data) + 1) / len(data)
    line, = axs.plot(np.arange(10)*0.2,datas, linestyle=linesty[int(t/5)],marker=markers[int(t/5)],linewidth=1.25,markerfacecolor='none')
    lines.append(line)
axs.set_position([0.16, 0.215, 0.35, 0.575])
# Add labels and title
axs.set_xlabel(r'$\epsilon_k \ \forall k$')
# axs.text(0.75, 0.1, p_names[a], transform=axs[a].transAxes, fontsize=FONT_SIZE)
axs.grid(True)



# axs[0].set_ylabel(r'CDF')
# # uu = 5*20
# # ll = 15*20
# axs[0].set_xlim(5*20, 15*20)
# axs[1].set_xlim(5*20, 15*20)
#
# # axs[1].set_xlim(uu, ll)
# # # axs[2].set_xlim(uu, ll)
# axs.set_xlim(-5, 0)
# axs[0].set_xticks([-5,-2.5,0])
# axs[0].set_xticklabels(['$\leq10^{-5}$','$10^{-2.5}$','$1$'])
#
# axs[1].set_xlim(9, 18)
# axs[1].set_xticks([9,12,15,18])
# axs.set_ylim(0.8, 1)
# axs[0].set_yticks([0.8,0.9,1])
# axs[1].set_ylim(0, 1)
# axs[1].set_yticks([0.,0.5,1])
#
#
# axs[0].tick_params(axis='x',direction='in')
# axs[0].tick_params(axis='y',direction='in')
# axs[1].tick_params(axis='x',direction='in')
# axs[1].tick_params(axis='y',direction='in')

# axs[0].set_xscale('log')
# axs[1].set_yscale('log')


# Add a legend
fig.legend(lines, data_name_list ,fontsize=FONT_SIZE, loc='lower left', bbox_to_anchor=(0.16, 0.85, 0.805, 0.1), mode="expand",ncol = 3 ,borderaxespad=0.1,handlelength=1.25)
# axs[0].legend(fontsize=8, loc='lower left', bbox_to_anchor=(0, 1.02, 5,0.1), ncol=3,borderaxespad=0.)
# plt.subplots_adjust(left=0.175, right=0.95,bottom=0.175,top=0.95)


# Save the figure as a PDF
output_path = os.path.join(current_dir, os.path.splitext(os.path.basename(__file__))[0]) + '.pdf'

fig.savefig(output_path, format='pdf', bbox_inches='tight', pad_inches=0.)

# Display the plot
plt.show()