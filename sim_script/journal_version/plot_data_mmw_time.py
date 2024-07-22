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


data_name_list = ["dual","loss","expm"]
current_dir = os.path.dirname(os.path.abspath(__file__))
file_name_list = ["mmw150-time-5-75","mmw150-time-10-75","mmw150-time-15-75"]



# Plot the data
fig, axs = plt.subplots(1,1,)
fig.set_size_inches(fig_width_in, fig_height_in)  # 3.5 inches width, height adjusted to maintain aspect ratio

N_REPEAT = 20
N_POINTS = len(file_name_list)
# Plot settings
markers = ['o', 's', '^','+']  # Different markers for each line
bar_fill_patterns = ['///', '\\\\\\', '---','+', 'x', 'o']

p_names = [r'a',r'b']
lines = []
labels = []

data_points = np.zeros((3,3))
for t in range(len(file_name_list)):
    data_file = os.path.join(current_dir, "sim_mmw_time","sim_mmw_time-2024-July-22-22-14-59-ail",file_name_list[t])
    data = np.genfromtxt(data_file, delimiter=',')
    data = np.mean(data[:,3:6],axis=0)/1e3
    data_points[t] = data

index = [5,10,15]
bar_width = 1
bars = []
for i in range(len(data_name_list)):
    b = axs.bar(i * bar_width - (len(data_name_list)-1)*bar_width/2. + np.asarray(index), data_points[:,i], bar_width, label=data_name_list[i],hatch=bar_fill_patterns[i])
    bars.append(b)

axs.set_position([0.165, 0.215, 0.805, 0.6])
# Add labels and title
axs.set_xlabel(r'WTSN size $l$ m')
axs.grid(True)
axs.set_axisbelow(True)



# axs[0].set_ylabel(r'Number of iterations')
axs.set_ylabel(r'Time (millisecond)')
# # uu = 5*20
# # ll = 15*20
# axs.set_xlim(5, 11)
# axs[1].set_xlim(5*20, 10*20)
#
# # axs[1].set_xlim(uu, ll)
# # axs[2].set_xlim(uu, ll)
# axs.set_ylim(0, 1)
# axs[1].set_ylim(0, 10)
axs.set_xticks(index)
axs.set_xticklabels([x*20 for x in index])
# axs[1].set_yticks([0,2,4,6,8,10])
# axs[0].set_yscale('log')
# axs[1].set_yscale('log')


# Add a legend
fig.legend(bars[0:len(data_name_list)], data_name_list ,fontsize=FONT_SIZE, loc='lower left', bbox_to_anchor=(0.165, 0.85, 0.805, 0.1), mode="expand",ncol = 4 ,borderaxespad=0.1,handlelength=1.5, handleheight= 1)
# axs[0].legend(fontsize=8, loc='lower left', bbox_to_anchor=(0, 1.02, 5,0.1), ncol=3,borderaxespad=0.)
# plt.subplots_adjust(left=0.175, right=0.95,bottom=0.175,top=0.95)


# Save the figure as a PDF
output_path = os.path.join(current_dir, os.path.splitext(os.path.basename(__file__))[0]) + '.pdf'

fig.savefig(output_path, format='pdf', bbox_inches='tight', pad_inches=0.)

# Display the plot
plt.show()