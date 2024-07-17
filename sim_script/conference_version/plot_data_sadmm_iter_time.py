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


data_name_list = ["With bounds","Without bounds"]
current_dir = os.path.dirname(os.path.abspath(__file__))

# Plot the data
fig, axs = plt.subplots(1,2,)
fig.set_size_inches(fig_width_in, fig_height_in)  # 3.5 inches width, height adjusted to maintain aspect ratio

N_REPEAT = 20
N_POINTS = 6
# Plot settings
markers = ['o', 's', '^','+']  # Different markers for each line
p_names = [r'a',r'b']
lines = []
labels = []
for a in [0]:
    file_name_list = ["g_bound_iter_list","n_bound_iter_list"]
    for ss in range(len(file_name_list)):
        s = file_name_list[ss]
        data_point = np.zeros(N_POINTS)
        for t in range(N_POINTS):
            fname = s+"-"+str(t+5)+"-"+"50"
            data_file = os.path.join(current_dir, "sim_sadmm_niter","sim_sadmm_niter-2024-July-17-02-58-47-ail",fname)
            data = np.genfromtxt(data_file, delimiter=',')
            data = np.mean(data[2:])
            data_point[t] = data
        line, = axs[a].plot(np.arange(5,5+N_POINTS)*20, data_point, marker=markers[ss],linewidth=1,markerfacecolor='none')
        lines.append(line)
        axs[a].set_position([0.16+a*0.48, 0.215, 0.325, 0.6])
        # Add labels and title
        axs[a].set_xlabel(r'WTSN size $l$ m')
        axs[a].text(0.1, 0.85, p_names[a], transform=axs[a].transAxes, fontsize=FONT_SIZE)
        axs[a].grid(True)

for a in [1]:
    file_name_list = ["g_bound_time_list","n_bound_time_list"]
    for ss in range(len(file_name_list)):
        s = file_name_list[ss]
        data_point = np.zeros(N_POINTS)
        for t in range(N_POINTS):
            fname = s+"-"+str(t+5)+"-"+"50"
            data_file = os.path.join(current_dir, "sim_sadmm_niter","sim_sadmm_niter-2024-July-17-02-58-47-ail",fname)
            data = np.genfromtxt(data_file, delimiter=',')
            data = np.mean(data[2:])/1e6
            data_point[t] = data
        line, = axs[a].plot(np.arange(5,5+N_POINTS)*20, data_point, marker=markers[ss],linewidth=1,markerfacecolor='none')
        lines.append(line)
        axs[a].set_position([0.16+a*0.48, 0.215, 0.325, 0.6])
        # Add labels and title
        axs[a].set_xlabel(r'WTSN size $l$ m')
        axs[a].text(0.1, 0.85, p_names[a], transform=axs[a].transAxes, fontsize=FONT_SIZE)
        axs[a].grid(True)

axs[0].set_ylabel(r'Number of iterations')
axs[1].set_ylabel(r'Time (second)')
# uu = 5*20
# ll = 15*20
axs[0].set_xlim(5*20, 10*20)
axs[1].set_xlim(5*20, 10*20)

# axs[1].set_xlim(uu, ll)
# axs[2].set_xlim(uu, ll)
axs[0].set_ylim(0, 10)
axs[1].set_ylim(0, 10)
axs[0].set_yticks([0,2,4,6,8,10])
axs[1].set_yticks([0,2,4,6,8,10])
# axs[0].set_yscale('log')
# axs[1].set_yscale('log')


# Add a legend
fig.legend(lines[0:2], data_name_list ,fontsize=FONT_SIZE, loc='lower left', bbox_to_anchor=(0.16, 0.85, 0.805, 0.1), mode="expand",ncol = 4 ,borderaxespad=0.1,handlelength=1)
# axs[0].legend(fontsize=8, loc='lower left', bbox_to_anchor=(0, 1.02, 5,0.1), ncol=3,borderaxespad=0.)
# plt.subplots_adjust(left=0.175, right=0.95,bottom=0.175,top=0.95)


# Save the figure as a PDF
output_path = os.path.join(current_dir, os.path.splitext(os.path.basename(__file__))[0]) + '.pdf'

fig.savefig(output_path, format='pdf', bbox_inches='tight', pad_inches=0.)

# Display the plot
plt.show()