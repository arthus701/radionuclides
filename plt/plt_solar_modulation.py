import sys
import os

import numpy as np

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from pandas import read_excel

from scipy.stats import norm
from scipy.signal import butter, sosfiltfilt

from styles import setup

# from pymagglobal.utils import i2lm_l, i2lm_m
sys.path.insert(1, '../src/')
from common import n_ref_solar

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR + '/../src/'))

setup()


def butter_lowpass_filter(data, cutoff, fs, order=5):
    sos = butter(order, cutoff, fs=fs, btype='low', analog=False, output='sos')
    y = sosfiltfilt(sos, data)
    return y


prefix = 'radio'
idata_fname = f'../out/{prefix}_result.nc'

with np.load(f'../out/{prefix}_ensemble.npz') as fh:
    knots_solar = fh['knots_solar']
    solar = 1.025 * fh['solar'] + 24.18
    solar = solar[:, ::2]


hpa_data = read_excel(
    '../dat/hpa_data.xlsx',
    sheet_name=2,
)

knots_hpa = hpa_data['Year']
solar_hpa = hpa_data['phi mean (MV)']

sample_frequency = 1. / np.mean(
    knots_solar[1:] - knots_solar[:-1]
)
solar_filt = butter_lowpass_filter(
    solar.T,
    1 / 650.,
    sample_frequency,
).T

paperwidth = 46.9549534722518 / 6
fig = plt.figure(figsize=(paperwidth, 0.3 * paperwidth))

gs = GridSpec(1, 3)

fig.subplots_adjust(wspace=0.08)

ax = fig.add_subplot(gs[0:2])
ax.plot(
    knots_solar,
    solar[:, :50],
    color='C0',
    alpha=0.03,
)
ax.plot(
    knots_solar,
    np.mean(solar, axis=1),
    color='C0',
    alpha=1,
    label='This study',
)
ax.plot(
    knots_hpa,
    solar_hpa,
    color='C1',
    zorder=-1,
    label='Nilsson et al. (2024)',
)
# ax.plot(
#     knots_solar,
#     np.mean(solar_filt, axis=1),
#     color='C0',
#     zorder=5,
# )

ax.set_xlabel("time [yrs.]")
ax.set_ylabel(r"$\Phi_\mathrm{HE17}$ [MV]")
ax.legend(frameon=False, ncols=2)

# left, bottom, width, height = ax.get_position().bounds
# hist_ax = fig.add_axes(
#     (left + width + 0.02, bottom, 0.3, height),
# )

hist_ax = fig.add_subplot(gs[2])

hist_ax.hist(
    solar[:-n_ref_solar].flatten(),
    bins=51,
    density=True,
)
hist_ax.set_xticks([200, 600, 1000])
hist_ax.set_yticks([])
hist_ax.set_xlabel(r"$\Phi_\mathrm{HE17}$ [MV]")
hist_ax.set_xlim(140, 1000)
arr = np.linspace(0, 1000, 501)

# Nilsson 2024 parameters
w = 0.09

mu_1 = 315
si_1 = 70

mu_2 = 600
si_2 = 130

hist_ax.plot(
    arr,
    w * norm.pdf(
        arr,
        loc=mu_1,
        scale=si_1,
    ) +
    (1 - w) * norm.pdf(
        arr,
        loc=mu_2,
        scale=si_2,
    ),
    color='C1',
    # alpha=0.7,
)

w = 0.18

mu_1 = 375
si_1 = 80

mu_2 = 665
si_2 = 90

# hist_ax.plot(
#     arr,
#     w * norm.pdf(
#         arr,
#         loc=mu_1,
#         scale=si_1,
#     ),
#     color='black',
#     ls='--',
# )
# hist_ax.plot(
#     arr,
#     (1 - w) * norm.pdf(
#         arr,
#         loc=mu_2,
#         scale=si_2,
#     ),
#     color='black',
#     ls='--',
# )

# fig.savefig(
#     '../fig/solar_modulation.pdf',
#     bbox_inches='tight',
#     pad_inches=0.05,
# )

# plt.figure()
# plt.hist(
#     solar[:-n_ref_solar].flatten(),
#     bins=51,
#     density=True,
#     cumulative=True,
# )

plt.show()
