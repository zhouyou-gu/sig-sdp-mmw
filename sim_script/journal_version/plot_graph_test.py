import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sim_src.util import GET_LOG_PATH_FOR_SIM_SCRIPT

FONT_SIZE = 9
fig_width_px = 350
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

current_dir = os.path.dirname(os.path.abspath(__file__))
data_path_in_current_dir = "sim_graph_test-2024-July-19-18-23-09-ail"

data_file = os.path.join(current_dir, "sim_graph_test", data_path_in_current_dir, "graph_test")
data = pd.read_csv(data_file)
data.columns = ['glit', 'it', 'n_sta', 'cell_size', 'density', 'Q_count', 'gain_in', 'gain_out', 'association','omega']

# Group by cell_size and density, then average the results over repeats
grouped_data = data.groupby(['n_sta', 'density']).mean().reset_index()

# Plot the data
fig, axs = plt.subplots(1,3,)
fig.set_size_inches(fig_width_in, fig_height_in)  # 3.5 inches width, height adjusted to maintain aspect ratio





# Plot settings
markers = ['o', 's', '^']  # Different markers for each line
d_names = ['n_sta','omega','Q_count']
p_names = [r'$K$',r'$\Omega$',r'$C$']
lines = []
labels = []
for a in range(3):
    for i, density in enumerate(grouped_data['density'].unique()):
        subset = grouped_data[grouped_data['density'] == density]
        line, = axs[a].plot(subset['cell_size']*20, subset[d_names[a]], marker=markers[i],markersize=5,linewidth=1,markerfacecolor='none')
        lines.append(line)
    axs[a].set_position([0.18+a*0.3, 0.2, 0.175, 0.6])
    # Add labels and title
    axs[a].set_xlabel(r'WTSN size $l$ m')
    axs[a].text(0.1, 0.85, p_names[a], transform=axs[a].transAxes, fontsize=FONT_SIZE)

    axs[a].grid(True)
    axs[a].tick_params(axis='both', which='major', labelsize=FONT_SIZE)
    axs[a].tick_params(axis='both', which='minor', labelsize=FONT_SIZE)


axs[0].set_ylabel(r'Numbers of users or constraints')
uu = 5*20
ll = 15*20
axs[0].set_xlim(uu, ll)
axs[1].set_xlim(uu, ll)
axs[2].set_xlim(uu, ll)
axs[0].set_ylim(0, 1000)
axs[1].set_ylim(20, 120)
axs[2].set_ylim(0, 6000)
axs[2].set_yticks([0, 1000,2000,3000,4000,5000,6000])


# Add a legend
fig.legend(lines[0:3], [r'$\rho=0.005$',r'$\rho=0.0075$',r'$\rho=0.01$'],fontsize=FONT_SIZE, loc='lower left', bbox_to_anchor=(0.18, 0.84, 0.775, 0.1), mode="expand",ncol = 3 ,borderaxespad=0.1,handlelength=1.5)
# axs[0].legend(fontsize=8, loc='lower left', bbox_to_anchor=(0, 1.02, 5,0.1), ncol=3,borderaxespad=0.)
# plt.subplots_adjust(left=0.175, right=0.95,bottom=0.175,top=0.95)


# Save the figure as a PDF
output_path = os.path.join(current_dir, os.path.splitext(os.path.basename(__file__))[0]) + '.pdf'

fig.savefig(output_path, format='pdf', bbox_inches='tight', pad_inches=0.)

# Display the plot
plt.show()