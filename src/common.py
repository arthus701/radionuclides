""" Collection of parameters and settings common to the stan and pyMC approach.
"""
import os
import numpy as np
import pandas as pd

from scipy.signal import butter, sosfiltfilt

from utils import matern_kernel, moving_average

from parameters import (
    t_min,
    t_max,
    t_solar_fine,
    step_solar_coarse,
    step_solar_fine,
    mu_solar,
    sigma_solar,
    tau_solar,
    use_11year_cycle,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def butter_lowpass_filter(data, cutoff, fs, order=5):
    sos = butter(order, cutoff, fs=fs, btype='low', analog=False, output='sos')
    y = sosfiltfilt(sos, data)
    return y


def calc_HPA(g20, phi):
    # g20 in uT
    hpa = g20 * 1e3 * (
        -1.0615e-15*phi**3 + 6.1089e-12*phi**2 - 1.9895e-08*phi + 6.0743e-5
    )

    return hpa


def prod_C14(dm, phi):
    Q = 1 / (
        0.09063
        + 0.03115 * dm
        + 0.0002615 * phi
        - 0.0001024 * dm**2
        + 1.91e-05 * dm * phi
        + 1.071e-08 * phi**2
    )

    return Q


def prod_Be10(dm, phi):
    Q = 1 / (
        5.576
        + 1.896 * dm
        + 0.01375 * phi
        - 0.01299 * dm**2
        + 0.001494 * dm * phi
        - 2.851e-07 * phi**2
    )

    return Q


# Derived parameters
if use_11year_cycle:
    knots_solar_fine = np.flip(
        np.arange(
            t_max,
            t_solar_fine-step_solar_fine,
            -step_solar_fine,
        )
    )
    knots_solar_coarse = np.flip(
        np.arange(
            t_solar_fine-step_solar_coarse,
            t_min-step_solar_coarse,
            -step_solar_coarse,
        )
    )
    knots_solar = np.hstack(
        [
            knots_solar_coarse,
            knots_solar_fine,
        ],
    )
else:
    knots_solar = np.flip(
        np.arange(
            t_max + step_solar_coarse,
            t_min,
            -step_solar_coarse,
        )
    )
    step_solar_fine = step_solar_coarse

# -----------------------------------------------------------------------------
# Solar modulation model

# Extract variations on regular and fast scale
solar_constr = pd.read_table(
    SCRIPT_DIR + '/../dat/US10_phi_mon_tab_230907.txt',
    sep=r'\s+',
    skiprows=23,
    header=None,
    dtype=float,
    engine='python',
    names=[
        'Year',
        'Phi (MV)',
    ],
)

idx_solar = np.argmin(
    np.abs(solar_constr['Year'].min() - knots_solar)
).flatten()
ref_min = knots_solar[idx_solar].item() - step_solar_fine / 2
bins = np.arange(ref_min, t_max + 2 * step_solar_fine, step_solar_fine)
solar_constr['Interval'] = pd.cut(solar_constr['Year'], bins)

ref_solar_years = []
ref_solar = []

for group in solar_constr.groupby('Interval', observed=True):
    ref_solar_years.append(group[0].mid)
    ref_solar.append(group[1]['Phi (MV)'].mean())

ref_solar_years = np.array(ref_solar_years)
ref_solar = np.array(ref_solar)

ref_solar_df = pd.DataFrame(
    data={
        't': ref_solar_years,
        'Phi': ref_solar,
    }
)
ref_solar_df['Phi avg.'] = moving_average(ref_solar_df, tau_solar, key='Phi')
ref_solar_df['Phi fast'] = ref_solar_df['Phi'] - ref_solar_df['Phi avg.']

n_ref_solar = len(ref_solar_df)

cov_obs = matern_kernel(
    ref_solar_df['t'].values,
    tau=tau_solar,
    sigma=sigma_solar,
)
cor_obs = matern_kernel(
    knots_solar,
    ref_solar_df['t'].values,
    tau=tau_solar,
    sigma=sigma_solar,
)
cov_solar = matern_kernel(
    knots_solar,
    tau=tau_solar,
    sigma=sigma_solar,
)

_icov_obs = np.linalg.inv(cov_obs + 1e-4*np.eye(len(ref_solar_df['t'].values)))
prior_mean_solar = mu_solar + cor_obs @ _icov_obs @ \
    (ref_solar_df['Phi avg.'] - mu_solar)
cov_solar = cov_solar - cor_obs @ _icov_obs @ cor_obs.T

# prior_mean_solar = np.ones(len(knots_solar)) * mean_solar
prior_mean_solar = prior_mean_solar[:-n_ref_solar]
chol_solar = np.linalg.cholesky(cov_solar+1e-6*np.eye(len(knots_solar)))[
    :-n_ref_solar,
    :-n_ref_solar,
]


# -----------------------------------------------------------------------------
# Data setup
# radionuclide production rate data
radData = pd.read_table(
    SCRIPT_DIR + '/../dat/CRN_9k_230922.txt'
)
radData.rename(
    columns={
        'Year': 't',
        'C14_INTCAL20': 'C14',
        'Be10_Greenland': 'Be10_NH',
        'Be10_Antarctica': 'Be10_SH',
    },
    inplace=True,
)

# Use 5 % errors for all C14 records
radData['dC14'] = 0.05 * np.abs(radData['C14'])

radData['dBe10_NH'] = 0.1   # * np.abs(radData['Be10_NH'])
radData['dBe10_SH'] = 0.1   # * np.abs(radData['Be10_SH'])

radData.sort_values(by='t', inplace=True)

# Tau = 2, using Brehm when possible
annual_C14_data = pd.read_excel(
    SCRIPT_DIR
    + '/../dat/'
    + 'ProductionRates100Versions_Matern3_2sigma2tau2.xlsx',
    skiprows=7,
)
annual_C14_data['t'] = 1950 + annual_C14_data['age -yr BP']
# annual_ensemble = annual_C14_data.values[:, 5:-1]

# annual_C14_data['C14'] = annual_ensemble.mean(axis=1)
# annual_C14_data['dC14'] = annual_ensemble.std(axis=1)

annual_C14_data['C14'] = annual_C14_data.values[:, 1]
annual_C14_data['dC14'] = annual_C14_data.values[:, 2]

annual_C14_data['C14'] = moving_average(annual_C14_data, 2)

annual_C14_data = annual_C14_data[annual_C14_data['t'] > -1000]

annual_C14_data.reset_index(inplace=True, drop=True)
annual_C14_data['dC14'] = 0.1

annual_C14_data = annual_C14_data[['t', 'C14', 'dC14']]

# exclude solar storm
idx = annual_C14_data.query('773.5 <= t and t <= 775.5').index
annual_C14_data.loc[idx, 'C14'] = np.nan

t_min_C14 = annual_C14_data['t'].min()

bins = radData[t_min_C14 < radData['t']]['t'].values
binwidth = 22
bins = np.concatenate(
    (
        [bins[0] - binwidth / 2],
        bins + binwidth / 2,
    )
)

annual_C14_data.dropna(how='any', inplace=True)

annual_C14_data['time_bin'] = pd.cut(annual_C14_data['t'], bins)
annual_C14_data['C14_detrended'] = annual_C14_data.groupby(
    'time_bin', observed=True,
)['C14'].transform(lambda x: x - x.mean())

brehm_data_CE = (969, 1933)
brehm_data_BCE = (-1000, -2)

annual_C14_data = annual_C14_data.query(
    f'({brehm_data_CE[0]} <= t and t <= {brehm_data_CE[1]})'
    f'or ({brehm_data_BCE[0]} <= t and t <= {brehm_data_BCE[1]})'
)
annual_C14_data = annual_C14_data.query(
    f'{t_min} <= t and t <= {t_max}'
)
annual_C14_data.reset_index(inplace=True, drop=True)

radData = radData.query(
    f'{t_min} <= t and t <= {t_max}'
)
radData.reset_index(inplace=True, drop=True)

idx_GL = np.asarray(radData.query('C14 == C14').index, dtype=int)
idx_NH = np.asarray(radData.query('Be10_NH == Be10_NH').index, dtype=int)
idx_SH = np.asarray(radData.query('Be10_SH == Be10_SH').index, dtype=int)

if __name__ == '__main__':
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(
        1, 1,
        figsize=(10, 5),
    )

    ax.errorbar(
        radData['t'],
        radData['C14'],
        yerr=radData['dC14'],
        ls='',
        color='grey',
        alpha=0.3,
    )
    ax.scatter(
        radData['t'],
        radData['C14'],
        ls='',
        color='grey',
        marker='.',
    )
    ax.errorbar(
        annual_C14_data['t'],
        annual_C14_data['C14'],
        yerr=annual_C14_data['dC14'],
        ls='',
        color='C0',
        alpha=0.3,
    )
    ax.scatter(
        annual_C14_data['t'],
        annual_C14_data['C14'],
        ls='',
        color='C0',
        marker='.',
    )
    ax.set_xlabel('time [yrs.]')
    ax.set_ylabel('$^{14}$C production (normalized)')

    fig.tight_layout()

    plt.show()
