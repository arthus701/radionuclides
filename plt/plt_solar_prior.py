# import sys
import numpy as np

import arviz as az

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from scipy.stats import beta
from styles import setup

# sys.path.insert(1, '../src/')
# from common import knots_solar, n_ref_solar
# prior sampling was run with different knot spacing
# change after re-run
prefix = 'radio_longterm'

with np.load(f'../out/{prefix}_ensemble.npz') as fh:
    knots_solar = fh['knots_solar']
n_ref_solar = 5

setup()

prefix = 'solar_prior'

paperwidth = 46.9549534722518 / 6
fig = plt.figure(figsize=(paperwidth, 0.6 * paperwidth))
kappas = [0., 0.15, 0.5]

gs = GridSpec(len(kappas), 4, figure=fig)

sample_axs = np.array(
    [fig.add_subplot(gs[it, 0:2]) for it in range(len(kappas))]
)
for ax in sample_axs[:-1]:
    ax.sharex(sample_axs[-1])
    ax.sharey(sample_axs[-1])
    ax.tick_params(labelbottom=False)

hist_axs = np.array(
    [fig.add_subplot(gs[it, 2]) for it in range(len(kappas))]
)
for ax in hist_axs[:-1]:
    ax.sharex(hist_axs[-1])
    ax.tick_params(labelbottom=False)

cdf_axs = np.array(
    [fig.add_subplot(gs[it, 3]) for it in range(len(kappas))]
)
for ax in cdf_axs[:-1]:
    ax.sharex(cdf_axs[-1])
    ax.sharey(cdf_axs[-1])
    ax.tick_params(labelbottom=False)

tL = 100
tR = 1000

a = np.array(
    [
        0.25, 2.5,
    ]
)
b = np.array(
    [
        0.25, 10.,
    ]
)

cdf = np.linspace(0, 1, 201)

for it, kappa in enumerate(kappas):
    idata_fname = f'../out/{prefix}_kappa_{kappa:.2f}_result.nc'
    iData = az.from_netcdf(idata_fname)

    sample_axs[it].plot(
        knots_solar[:-n_ref_solar],
        iData.posterior['sm_at_knots'].values[0, :3, :-n_ref_solar].T,
        color='C0',
        alpha=0.3,
    )

    sample_axs[it].plot(
        knots_solar[:-n_ref_solar],
        iData.posterior['sm_at_knots'].values[0, 4, :-n_ref_solar].T,
        color='C0',
    )
    sample_axs[it].plot(
        knots_solar[-n_ref_solar-1:],
        iData.posterior['sm_at_knots'].values[0, 0, -n_ref_solar-1:],
        color='C0',
    )

    hist_axs[it].hist(
        1.025 * iData.posterior['sm_at_knots'].values[:, :, :-n_ref_solar].flatten() + 24.18,
        bins=np.linspace(100, 1100, 51),
    )

    inverse_approx = (
        tL + tR * (
            (1 - kappa) * beta.cdf(
                cdf,
                a=a[0],
                b=b[0],
            )
            + kappa * beta.cdf(
                cdf,
                a=a[1],
                b=b[1],
            )
        )
    )
    cdf_axs[it].plot(
        cdf,
        inverse_approx,
    )

sample_axs[0].set_ylim(100, 1100)
sample_axs[-1].set_xlabel('time [yrs.]')
for it, ax in enumerate(sample_axs):
    ax.set_ylabel(
        fr'$\kappa = {kappas[it]:.2f}$' + '\n'
        + r'$\phi$ [MV]'
    )

hist_axs[-1].set_xlabel(r'$\phi$ [MV]')
hist_axs[-1].set_xlim(100, 1150)
hist_axs[-1].set_xticks([200, 600, 1000])
for ax in hist_axs:
    ax.set_yticks([])
    ax.tick_params(labelleft=False)

cdf_axs[-1].set_xlabel('cdf')
cdf_axs[-1].set_yticks([200, 600, 1000])
for ax in cdf_axs:
    ax.set_ylabel(r'$\phi$ [MV]')

fig.set_constrained_layout(True)

# fig.savefig(
#     '../fig/solar_prior.pdf',
#     bbox_inches='tight',
#     pad_inches=0.,
# )

plt.show()
