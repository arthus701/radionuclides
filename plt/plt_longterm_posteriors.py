import sys
import os

import numpy as np
import arviz as az

from scipy.stats import gamma

from matplotlib import pyplot as plt

from styles import setup

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR + '/../src/'))

setup()

fac = 0.63712**3

paperwidth = 46.9549534722518 / 6

fig, axs = plt.subplots(
    1, 2,
    figsize=(paperwidth, 0.35 * paperwidth),
)
fig.subplots_adjust(hspace=0.)

prefix = 'radio_longterm'

idata_fname = f'../out/{prefix}_result.nc'
iData = az.from_netcdf(idata_fname)

axs[0].hist(
    iData.posterior['sm_tau_longterm'].values.flatten(),
    density=True,
    color='C0',
    bins=51,
    alpha=0.5,
)
axs[0].hist(
    iData.posterior['sm_tau_longterm'].values.flatten(),
    density=True,
    color='C0',
    bins=51,
    histtype='step',
)

t_min, t_max = axs[0].get_xlim()
t_arr = np.linspace(t_min, t_max, 201)

axs[0].set_title(r'$\tau_\text{long-term}$')
axs[0].plot(
    100 + t_arr,
    gamma.pdf(t_arr, a=3.5, scale=500),
    ls='--',
    color='grey',
    zorder=-1,
)
axs[0].set_xlim(t_min, t_max)

axs[0].set_xlabel('[yrs.]')

axs[1].set_title(r'$\sigma_\text{long-term}$')
axs[1].hist(
    iData.posterior['sm_longterm_scale'].values.flatten(),
    density=True,
    color='C0',
    bins=51,
    alpha=0.5,
)
axs[1].hist(
    iData.posterior['sm_longterm_scale'].values.flatten(),
    density=True,
    color='C0',
    bins=51,
    histtype='step',
)

s_min, s_max = axs[1].get_xlim()
s_arr = np.linspace(s_min, s_max, 201)

axs[1].plot(
    s_arr,
    gamma.pdf(s_arr, a=3, scale=200/3),
    ls='--',
    color='grey',
    zorder=-1,
)
axs[1].set_xlim(s_min, s_max)

axs[1].set_xlabel('[MV]')

for ax in axs:
    ax.set_yticks([])

fig.tight_layout()

# fig.savefig(
#     '../fig/longterm_posteriors.pdf',
#     bbox_inches='tight',
#     pad_inches=0.,
# )

plt.show()
