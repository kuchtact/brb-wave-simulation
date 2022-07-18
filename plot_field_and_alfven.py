import numpy as np
from dedalus import public as de
import matplotlib.pyplot as plt
from background_field import get_background_field
from plot_parameters import set_params, get_config
set_params()


config = get_config()

r_start = config['domain']['r_start']
r_end = config['domain']['r_end']

r_basis = np.linspace(r_start, r_end, num=100)
Bz0 = get_background_field(r_basis, config)
v_alfven = (6 * Bz0**2 / (1 - Bz0**2))**0.5
beta = (1 - Bz0**2) / Bz0**2

ax = plt.axes()
ax.set_xlabel(r'$\tilde{r}$')
ax.set_xlim(0, 1)
ax.set_ylabel(r'$\tilde{B} = B / B_H$', color='blue')
ax.set_title('Background field and Alfven speed')
ax.set_ylim(0, 1)
ax.tick_params(axis='y', labelcolor='blue')

opposite_ax = ax.twinx()
opposite_ax.set_ylabel(r'$\tilde{v}$', color='red')
opposite_ax.set_ylim(0, int(max(v_alfven) * 1.1))
opposite_ax.tick_params(axis='y', labelcolor='red')

B_plot = ax.plot(r_basis, Bz0, label=r'$\tilde{B}_{z0}$', color='blue')
v_plot = opposite_ax.plot(r_basis, v_alfven, label=r'$\tilde{v}_A$', color='red')
beta_plot = ax.plot(r_basis, beta, label=r'$\beta$', color='green')

# Combine the legends
legends = B_plot + v_plot + beta_plot
labels = [l.get_label() for l in legends]
plt.legend(legends, labels)

plt.savefig('./plots/field_and_alfven.pdf', format='pdf', dpi=300)
plt.show()
