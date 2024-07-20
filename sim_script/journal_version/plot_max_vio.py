import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sim_src.util import GET_LOG_PATH_FOR_SIM_SCRIPT

FONT_SIZE = 9
fig_width_px = 750
fig_height_px = 150
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


data_name_list = ["SDP","LP","MINTP","MASSO"]
file_name_list = ["sadmm","ladmm","mgain","masso"]
current_dir = os.path.dirname(os.path.abspath(__file__))

# Plot the data
fig, axs = plt.subplots(1,3,)
fig.set_size_inches(fig_width_in, fig_height_in)  # 3.5 inches width, height adjusted to maintain aspect ratio



N_REPEAT = 20
N_POINTS = 6
# Plot settings
markers = ['o', 's', '^','+']  # Different markers for each line
p_names = [r'a',r'b',r'c']
lines = []
labels = []
T = [5,10,15]
y_labels = [None for i in range(9)]
y_labels[0] = 0.02
y_labels[4] = 0.06
y_labels[8] = 0.1

for a in range(3):
    t = T[a]
    image_data = np.zeros((9,2500))
    for i in range(2,11):
        fname = "mmw-dual-"+str(t)+"-"+"75"+"-"+str(i)
        data_file = os.path.join(current_dir, "sim_all_mmw","sim_all_mmw-2024-July-20-17-54-34-ail",fname)
        data = np.genfromtxt(data_file, delimiter=',')
        ub = data[0, 3:]
        image_data[i-2,:ub.size] = ub
    image_data[image_data==0] = np.nan
    im = axs[a].imshow(image_data,cmap='plasma',aspect='auto',vmin=0, vmax=5)
    axs[a].set_xscale('log')
    axs[a].set_xticks([1, 10, 100, 1000])
    axs[a].set_xlim(1, 2500)
    axs[a].set_position([0.09+a*0.3, 0.45, 0.285, 0.5])
    if a == 0:
        axs[a].set_ylabel(r'$\eta$')
        axs[a].set_yticks([i for i in range(9)])
        axs[a].set_yticklabels(y_labels)
    else:
        axs[a].set_yticks([i for i in range(9)])
        axs[a].set_yticklabels([])

    # Add labels and title
    axs[a].set_xlabel(r'Number of iterations')
    # axs[a].text(0.1, 0.85, p_names[a], transform=axs[a].transAxes, fontsize=FONT_SIZE)
    # axs[a].grid(True)
    if a == len(axs)-1:
        cbar = fig.colorbar(im, ax=axs, location='right', aspect=10, pad=0.015,fraction=0.05,shrink = 1)
        cbar.set_ticks([0,  5])




# Add a legend
# fig.legend(lines[0:4], data_name_list ,fontsize=FONT_SIZE, loc='lower left', bbox_to_anchor=(0.16, 0.85, 0.805, 0.1), mode="expand",ncol = 4 ,borderaxespad=0.1,handlelength=1)
# axs[0].legend(fontsize=8, loc='lower left', bbox_to_anchor=(0, 1.02, 5,0.1), ncol=3,borderaxespad=0.)
# plt.subplots_adjust(left=0.175, right=0.95,bottom=0.175,top=0.95)


# Save the figure as a PDF
output_path = os.path.join(current_dir, os.path.splitext(os.path.basename(__file__))[0]) + '.pdf'

fig.savefig(output_path, format='pdf', bbox_inches='tight', pad_inches=0.)

# Display the plot
plt.show()