import sys
import os

import numpy as np
import arviz as az

from pandas import read_excel

from matplotlib import pyplot as plt

from scipy.signal import butter, sosfiltfilt

from styles import setup

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR + '/../src/'))
from common import (
    knots_solar,
    knots_solar_fine,
)
from utils import npinterp


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter(
        order,
        [lowcut, highcut],
        fs=fs,
        btype='band',
        analog=False,
        output='sos',
    )
    y = sosfiltfilt(sos, data)
    return y


def interp_BP(x, y, lower=1 / 18, upper=1 / 6):
    interp_x, dt = np.linspace(
        x.min(),
        x.max(),
        10001,
        retstep=True
    )
    interp_y = np.interp(
        interp_x,
        x,
        y,
    )

    interp_y_BP = butter_bandpass_filter(
        interp_y,
        lower,
        upper,
        fs=1 / dt,
    )
    return np.interp(x, interp_x, interp_y_BP)


setup()

paperwidth = 46.9549534722518 / 6

brehm_data_sm = read_excel(
    '../dat/41561_2020_674_MOESM2_ESM.xlsx',
    sheet_name='Figure1c',
)
brehm_data_sm.dropna(how='all', inplace=True)
brehm_sm_t = brehm_data_sm['ETH Simulated time (yr AD)'].values[::-1]
brehm_sm = \
    brehm_data_sm['ETH solar modulation parameter (MeV)'].values[::-1]

brehm_sm_sample_frequency = 1. / np.mean(
    np.abs(
        brehm_sm_t[1:] - brehm_sm_t[:-1]
    )
)
brehm_sm_BP = butter_bandpass_filter(
    brehm_sm,
    1 / 18,
    1 / 6,
    fs=brehm_sm_sample_frequency,
)

iData = az.from_netcdf('../out/one_dim_result.nc')

solar = iData.posterior['sm_at_knots'].values[:, ::2, :]
solar = solar.reshape(-1, len(knots_solar)).T
solar = 1.025 * solar + 24.18

solar_fine = iData.posterior['sm_fast_at_knots'].values[:, ::2, :]
solar_fine = solar_fine.reshape(-1, len(knots_solar_fine)).T
solar_fine = 1.025 * solar_fine + 24.18

solar_total = npinterp(
    knots_solar_fine,
    knots_solar,
    solar
) + solar_fine

sample_frequency = 1. / np.mean(
    knots_solar_fine[1:] - knots_solar_fine[:-1]
)
full_samples_BP = butter_bandpass_filter(
    solar_total.T,
    1 / 18,
    1 / 8.1,
    fs=sample_frequency,
).T

fig, axs = plt.subplots(
    2, 1,
    figsize=(paperwidth, 0.5 * paperwidth),
)

axs[0].plot(
    knots_solar_fine,
    np.mean(full_samples_BP, axis=1),
    color='C0',
    label='This study',
    zorder=5,
)
axs[0].plot(
    brehm_sm_t,
    brehm_sm_BP,
    color='C1',
    label='Brehm et al. (2021)',
    zorder=3,
)

axs[1].plot(
    knots_solar_fine,
    np.mean(full_samples_BP, axis=1),
    color='C0',
    label='This study',
    zorder=5,
)
axs[1].fill_between(
    knots_solar_fine,
    np.mean(full_samples_BP, axis=1)
    - np.std(full_samples_BP, axis=1),
    np.mean(full_samples_BP, axis=1)
    + np.std(full_samples_BP, axis=1),
    color='C0',
    zorder=-1,
    alpha=0.2,
)

axs[0].set_ylabel(r"$\Phi_\mathrm{HE17}$ [MV]")
axs[1].set_ylabel(r"$\Phi_\mathrm{HE17}$ [MV]")

axs[0].legend(
    ncols=2,
    frameon=False,
    loc='upper right',
)
axs[0].set_xlim(900, 2000)
axs[1].set_xlabel('time [yrs.]')
axs[1].set_xlim(-1100, 2000)
axs[1].set_xlabel('time [yrs.]')

fig.align_ylabels(axs)
fig.tight_layout()

# fig.savefig(
#     '../fig/comparison_brehm_filtered.pdf',
#     bbox_inches='tight',
#     pad_inches=0.,
# )

plt.show()
