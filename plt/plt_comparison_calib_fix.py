import sys
import os

import numpy as np
from pandas import read_excel

from matplotlib import pyplot as plt

from styles import setup

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR + '/../src/'))

setup()

fac = 0.63712**3

paperwidth = 46.9549534722518 / 6

fig, axs = plt.subplots(
    3, 1,
    sharex=True,
    figsize=(paperwidth, 0.55 * paperwidth),
)

labels = [
    'mag only',
    'mag + radio',
    'common scaling',
]
for it, prefix in enumerate([
    'arch', 'radio', 'radio_calib_fix'
]):
    with np.load(f'../out/{prefix}_ensemble.npz') as fh:
        knots = fh['knots']
        coeffs = fh['coeffs']
        if 'radio' in prefix:
            knots_solar = fh['knots_solar']
            solar = 1.025 * fh['solar'] + 24.18
            solar = solar[:, ::2]

        dips = fac * np.sqrt(np.sum(coeffs[:3]**2, axis=0))

    axs[0].plot(
        knots,
        np.mean(dips, axis=1),
        color=f'C{it}',
        zorder=5,
        label=labels[it]
    )
    axs[1].plot(
        knots,
        np.mean(coeffs[3], axis=1),
        color=f'C{it}',
        zorder=5,
    )
    axs[1].axhline(
        np.mean(coeffs[3]),
        color=f'C{it}',
        zorder=-1,
        ls='--',
        lw=1,
    )

    if 'radio' in prefix:
        axs[2].plot(
            knots_solar,
            np.mean(solar, axis=1),
            color=f'C{it}',
            zorder=5,
        )


hpa_data = read_excel(
    '../dat/hpa_data.xlsx',
    sheet_name=2,
)

knots_hpa = hpa_data['Year']
solar_hpa = hpa_data['phi mean (MV)']
dm_hpa = hpa_data['DM mean (1E22 Am2)']
g20_hpa = hpa_data['g20 mean (nT)']

axs[0].plot(
    knots_hpa,
    dm_hpa,
    color='C3',
    zorder=3,
    label='Nilsson et al. (2024)',
)
axs[1].plot(
    knots_hpa,
    g20_hpa / 1e3,
    color='C3',
    zorder=3,
)
axs[1].axhline(
    np.mean(g20_hpa / 1e3),
    color='C3',
    zorder=-2,
    ls='--',
)
axs[2].plot(
    knots_hpa,
    solar_hpa,
    color='C3',
    zorder=3,
)

axs[0].set_ylabel("DM [$10^{22}$Am$^2$]")
axs[1].set_ylabel(r"$g_2^0$ [$\mu$T]")
axs[0].legend(frameon=False, ncols=2)
axs[2].set_ylabel(r"$\Phi_\mathrm{HE17}$ [MV]")

axs[1].set_yticks([-4, 0, 4])
axs[1].set_ylim(-5.5, 5.5)

axs[2].set_xlim(-7000, 2000)
axs[2].set_xlabel('time [yrs.]')

axs[1].patch.set_visible(False)
axs[2].patch.set_visible(False)

fig.align_ylabels(axs)
fig.tight_layout()
fig.subplots_adjust(hspace=0.)

# fig.savefig(
#     '../fig/calib_fix_comparison.pdf',
#     bbox_inches='tight',
#     pad_inches=0.,
# )

plt.show()
