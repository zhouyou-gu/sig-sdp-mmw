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


data_name_list = ["MMW-GB","MMW-NB","ADMM-GB"]
current_dir = os.path.dirname(os.path.abspath(__file__))

# Plot the data
fig, axs = plt.subplots(1,1,)
fig.set_size_inches(fig_width_in, fig_height_in)  # 3.5 inches width, height adjusted to maintain aspect ratio

N_REPEAT = 20
N_POINTS = 11
# Plot settings
markers = ['v', '^', 'o','+']  # Different markers for each line
p_names = [r'a',r'b']
lines = []
labels = []
data_point = np.zeros((11,3))
for t in range(N_POINTS):
    fname = "time-"+str(t+5)+"-"+"75"
    data_file = os.path.join(current_dir, "sim_mmw_scs_iter_time","sim_mmw_scs_iter_time-2024-July-23-02-23-45-ail",fname)
    data = np.genfromtxt(data_file, delimiter=',')
    data = np.mean(data[:,[3,7,5]],axis=0)
    data_point[t,:] = data/1e6
for i in range(3):
    line, = axs.plot((np.arange(5,5+N_POINTS)*20)**2*75e-4, data_point[:,i], marker=markers[i],linewidth=1,markerfacecolor='none')
    lines.append(line)
axs.set_position([0.18, 0.2, 0.775, 0.75])
# Add labels and title
axs.set_xlabel(r'Number of users, $K$')
axs.grid(True)



# axs[0].set_ylabel(r'Number of iterations')
axs.set_ylabel(r'Time (second)')
# # uu = 5*20
# # ll = 15*20
# axs[0].set_xlim(5*20, 10*20)
# axs[1].set_xlim(5*20, 10*20)
#
# # axs[1].set_xlim(uu, ll)
# # axs[2].set_xlim(uu, ll)
axs.set_ylim(0, 200)
axs.set_xlim(50, 700)
# axs[1].set_ylim(0, 10)
# axs[0].set_yticks([0,2,4,6,8,10])
axs.set_xticks(100*np.arange(8))
# axs[0].set_yscale('log')
# axs[1].set_yscale('log')


# Add a legend
fig.legend(lines, data_name_list ,fontsize=FONT_SIZE, loc='lower left', bbox_to_anchor=(0.215, 0.625, 0.2, 0.1),ncol = 1 ,borderaxespad=0.1,handlelength=1.5)
# axs[0].legend(fontsize=8, loc='lower left', bbox_to_anchor=(0, 1.02, 5,0.1), ncol=3,borderaxespad=0.)
# plt.subplots_adjust(left=0.175, right=0.95,bottom=0.175,top=0.95)


# Save the figure as a PDF
output_path = os.path.join(current_dir, os.path.splitext(os.path.basename(__file__))[0]) + '.pdf'

fig.savefig(output_path, format='pdf', bbox_inches='tight', pad_inches=0.)

# Display the plot
plt.show()