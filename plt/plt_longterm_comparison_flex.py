import sys
import os

import numpy as np
import arviz as az

from matplotlib import pyplot as plt

from styles import setup

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR + '/../src/'))
from common import n_ref_solar

setup()

fac = 0.63712**3

paperwidth = 46.9549534722518 / 6

fig, axs = plt.subplots(
    3, 1,
    sharex=True,
    figsize=(paperwidth, 0.55 * paperwidth),
)
twin_solar = axs[2].twinx()
fig.subplots_adjust(hspace=0.)

prefix = 'radio_longterm'

with np.load(f'../out/{prefix}_ensemble.npz') as fh:
    knots = fh['knots']
    coeffs = fh['coeffs']
    knots_solar = fh['knots_solar']
    solar = 1.025 * fh['solar'] + 24.18
    solar = solar[:, ::2]

    dips = fac * np.sqrt(np.sum(coeffs[:3]**2, axis=0))

idata_fname = f'../out/{prefix}_result.nc'
iData = az.from_netcdf(idata_fname)

longterm_samples_US05 = (
    iData
    .posterior['sm_longterm']
    .values
    .reshape(2000, -1).T[:, ::2]
)

longterm_samples = (
    1.025 * longterm_samples_US05
    + 24.18
)

axs[0].plot(
    knots,
    np.mean(dips, axis=1),
    color='C0',
    zorder=5,
    # label=rf'$\tau$={time} yrs.'
    label='longterm',
)
axs[0].plot(
    knots,
    dips[:, :10],
    color='C0',
    zorder=-1,
    alpha=0.2,
    lw=1
)
axs[1].plot(
    knots,
    np.mean(coeffs[3], axis=1),
    color='C0',
    zorder=5,
)
axs[1].plot(
    knots,
    coeffs[3, :, :10],
    color='C0',
    zorder=-1,
    alpha=0.2,
    lw=1
)

axs[2].plot(
    knots_solar[:-n_ref_solar],
    np.mean(longterm_samples, axis=1),
    color='C0',
    zorder=5,
)
axs[2].plot(
    knots_solar[:-n_ref_solar],
    longterm_samples[:, :10],
    color='C0',
    zorder=-1,
    alpha=0.2,
    lw=1
)
# solar[:-n_ref_solar] -= longterm_samples
twin_solar.plot(
    knots_solar,
    np.mean(solar, axis=1),
    color='C0',
    alpha=1,
)

with np.load('../out/radio_ensemble.npz') as fh:
    knots = fh['knots']
    coeffs = fh['coeffs'][:, :, ::2]
    knots_solar = fh['knots_solar']
    solar = 1.025 * fh['solar'] + 24.18
    solar = solar[:, ::2]

axs[0].plot(
    knots,
    np.mean(dips, axis=1),
    color='C1',
    zorder=1,
    label='no longterm'
)
axs[1].plot(
    knots,
    np.mean(coeffs[3], axis=1),
    color='C1',
    zorder=1,
)

twin_solar.plot(
    knots_solar,
    np.mean(solar, axis=1),
    color='C1',
    zorder=-1,
)

axs[0].set_ylabel("DM [$10^{22}$Am$^2$]")
axs[1].set_ylabel(r"$g_2^0$ [$\mu$T]")
axs[0].legend(frameon=False, ncols=2)
axs[2].set_ylabel(r"$\Phi_\mathrm{HE17}$ [MV]")

bottom, top = twin_solar.get_ylim()
axs[2].set_ylim(bottom - 500, top - 500)
axs[2].set_yticks([-200, 0, 200])
twin_solar.set_ylabel(r"$\Phi_\mathrm{HE17}$ [MV]")

axs[2].set_xlim(-7000, 2000)
axs[2].set_xlabel('time [yrs.]')

axs[2].set_zorder(twin_solar.get_zorder()+1)
axs[1].patch.set_visible(False)
axs[2].patch.set_visible(False)

fig.align_ylabels(axs)

# fig.savefig(
#     '../fig/longterm_comparison_flex.pdf',
#     bbox_inches='tight',
#     pad_inches=0.,
# )

plt.show()
