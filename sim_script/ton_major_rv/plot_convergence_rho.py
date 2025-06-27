import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sim_src.util import GET_LOG_PATH_FOR_SIM_SCRIPT

FONT_SIZE = 9
fig_width_px = 300
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


data_name_list = [r"$\rho=0.0025$", r"$\rho=0.005$", r"$\rho=0.0075$", r"$\rho=0.01$", r"$\rho=0.0125$"]
current_dir = os.path.dirname(os.path.abspath(__file__))



# Plot the data
fig, axs = plt.subplots(1,1,)
fig.set_size_inches(fig_width_in, fig_height_in)  # 3.5 inches width, height adjusted to maintain aspect ratio

# Plot settings
markers = ['+', 'x', 'd']  # Different markers for each line
bar_fill_patterns = ['///', '\\\\\\', '---','+', 'x', 'o']


data_points = np.zeros((3,11))

data_file = os.path.join(current_dir, "sim_convergence_rho/sim_convergence_rho-2025-June-26-13-00-11-ail/mmw-dual-10-4")
data = np.genfromtxt(data_file, delimiter=',')
lines = []
colors = ["#FC5A50","#FF8C00","#069AF3"]
for i in range(5):
    gap_all = np.zeros(data.shape[1]-3)
    for repeat in range(5):
        ub = data[20*i*2+repeat*2, 3:]
        lb = data[20*i*2+repeat*2+1, 3:]
        gap = ub
        gap = gap/gap[0]
        gap_all += gap
    gap_all /= 5
    line, = axs.plot(gap_all, linestyle='-', linewidth=1.25)
    lines.append(line)


axs.set_position([0.19, 0.275, 0.77, 0.65])
# Add labels and title
axs.set_xlabel(r'Number of iterations')
axs.grid(True)
axs.set_axisbelow(True)



# axs[0].set_ylabel(r'Number of iterations')
axs.set_ylabel(r'$\max_c \mathbf{A}^{(c)} \bullet \bar{\mathbf{X}}$')
# # uu = 5*20
# # ll = 15*20
axs.set_xlim(50, 700)
axs.set_xticks(100*np.arange(8))
#
# # axs[1].set_xlim(uu, ll)
# # axs[2].set_xlim(uu, ll)
axs.set_ylim(0, 1.25)
# axs[1].set_ylim(0, 10)
# axs.set_xticks(index)
# axs.set_xticklabels([x*20 for x in index])
# axs[1].set_yticks([0,2,4,6,8,10])
# axs[0].set_yscale('log')
# axs[1].set_yscale('log')


# Add a legend
ncol = 2  # Number of columns in the legend
h, l = lines, data_name_list
nrows = -(-len(h) // ncol)                         # ceiling division
idx   = np.arange(len(h))
pad   = nrows * ncol - len(idx)
idx   = np.concatenate([idx, np.full(pad, -1)])    # pad
order = idx.reshape(nrows, ncol).T.ravel()
order = order[order >= 0]                          # drop sentinels
h = [h[i] for i in order]
l = [l[i] for i in order]
fig.legend(h, l ,fontsize=FONT_SIZE-1, loc='upper right', bbox_to_anchor=(0.925, 0.9),ncol = ncol ,borderaxespad=0.1,handlelength=1., handleheight= 1, handletextpad=0.2, 
 # frameon=True,          # draw a frame
# fancybox=False,        # <-- square corners (like MATLAB)
# edgecolor='black',     # black  frame edge
# facecolor='white',     # white background
framealpha=1,          # fully opaque
borderpad=0.3,         # tight inner padding (font-size units)
labelspacing=0.2,      # tight vertical space between rows
columnspacing=0.8,      # space between columns
)

# axs[0].legend(fontsize=8, loc='lower left', bbox_to_anchor=(0, 1.02, 5,0.1), ncol=3,borderaxespad=0.)
# plt.subplots_adjust(left=0.175, right=0.95,bottom=0.175,top=0.95)


# Save the figure as a PDF
output_path = os.path.join(current_dir, os.path.splitext(os.path.basename(__file__))[0]) + '.pdf'

fig.savefig(output_path, format='pdf', pad_inches=0.)

# Display the plot
plt.show()