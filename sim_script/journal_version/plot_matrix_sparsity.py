import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sim_src.util import GET_LOG_PATH_FOR_SIM_SCRIPT


from sim_src.env.env import env
from sim_src.util import *

FONT_SIZE = 9
fig_width_px = 425
fig_height_px = 175
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

current_dir = os.path.dirname(os.path.abspath(__file__))

# Plot the data
fig, axs = plt.subplots(1,3,)
fig.set_size_inches(fig_width_in, fig_height_in)  # 3.5 inches width, height adjusted to maintain aspect ratio



cell_size = [5,10,15]


# Plot settings
markers = ['o', 's', '^']  # Different markers for each line
d_names = ['n_sta','omega','Q_count']
p_names = [r'$l=100$m',r'$l=200$m',r'$l=300$m']
lines = []
labels = []

mats = []
for cs in [5,10,15]:
    e = env(cell_size=cs,sta_density_per_1m2=75e-4,seed=3)
    S, Q, _ = e.generate_S_Q_hmax()
    S_no_diag = S.copy()
    S_no_diag.setdiag(0)
    S_no_diag.eliminate_zeros()
    D = S + S.transpose() + Q + Q.transpose()
    D.eliminate_zeros()
    mats.append(D)

x = 135/fig_width_px
y = 135/fig_height_px
from scipy.sparse.csgraph import reverse_cuthill_mckee

for a in range(3):
    perm = reverse_cuthill_mckee(mats[a])
    mats[a] = mats[a][perm, :]
    mats[a] = mats[a][:, perm]
    row_indices, col_indices = mats[a].nonzero()

    axs[a].scatter(row_indices, col_indices,s=0.05/(a+1)**2)
    axs[a].set_aspect('equal', 'box')
    axs[a].set_position([0.035+a*(x+0.), 0.15, x, y])
    # Add labels and title
    # axs[a].set_xlabel(r'WTSN size $l$ m')
    axs[a].text(0.05, 0.025, p_names[a], transform=axs[a].transAxes, fontsize=FONT_SIZE)

    axs[a].grid(True)
    axs[a].tick_params(axis='both', which='major', labelsize=FONT_SIZE)
    axs[a].tick_params(axis='both', which='minor', labelsize=FONT_SIZE)
    axs[a].set_xlim(1, mats[a].shape[0])
    axs[a].set_ylim(1, mats[a].shape[0])
    axs[a].set_xticks([1, mats[a].shape[0]])
    axs[a].set_yticks([1, mats[a].shape[0]])
    axs[a].set_xticklabels([1,r'$K$'])
    axs[a].set_yticklabels([1,r'$K$'])
    axs[a].invert_yaxis()

# axs[0].set_ylabel(r'Numbers of users or constraints')
uu = 5*20
ll = 15*20
# axs[0].set_xlim(uu, ll)
# axs[1].set_xlim(uu, ll)
# axs[2].set_xlim(uu, ll)
# axs[0].set_ylim(0, 1000)
# axs[1].set_ylim(20, 120)
# axs[2].set_ylim(0, 6000)
# axs[2].set_yticks([0, 1000,2000,3000,4000,5000,6000])


# Add a legend
# fig.legend(lines[0:3], [r'$\rho=0.005$',r'$\rho=0.0075$',r'$\rho=0.01$'],fontsize=FONT_SIZE, loc='lower left', bbox_to_anchor=(0.18, 0.84, 0.775, 0.1), mode="expand",ncol = 3 ,borderaxespad=0.1,handlelength=1.5)
# axs[0].legend(fontsize=8, loc='lower left', bbox_to_anchor=(0, 1.02, 5,0.1), ncol=3,borderaxespad=0.)
# plt.subplots_adjust(left=0.175, right=0.95,bottom=0.175,top=0.95)


# Save the figure as a PDF
output_path = os.path.join(current_dir, os.path.splitext(os.path.basename(__file__))[0]) + '.pdf'

fig.savefig(output_path, format='pdf', bbox_inches='tight', pad_inches=0.)

# Display the plot
plt.show()