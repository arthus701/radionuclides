""" Collection of parameters and settings common to the stan and pyMC approach.
"""
import os
import numpy as np
import pandas as pd

from scipy.signal import butter, sosfiltfilt

from paleokalmag.utils import dsh_basis
from paleokalmag.data_handling import Data
# from paleokalmag.data_handling import read_data

from pymagglobal.utils import lmax2N, i2lm_l    # , scaling

from utils import matern_kernel, moving_average

from parameters import (
    lmax,
    t_min,
    t_max,
    step,
    t_solar_fine,
    step_solar_coarse,
    step_solar_fine,
    gamma_0,
    alpha_list,
    tau_list,
    dip,
    alpha_dip,
    omega,
    xi,
    chi,
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
n_coeffs = lmax2N(lmax)
knots = np.arange(
    t_min,
    t_max + step,
    step,
)
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

# -----------------------------------------------------------------------------
# Magnetic field model
# Prior mean and covariance (as cholesky factors)
prior_mean = np.zeros((n_coeffs, len(knots)))
prior_mean[0] = gamma_0

alphas = np.array([alpha_list[i2lm_l(it)-1]**2 for it in range(n_coeffs)])
taus = np.array([tau_list[i2lm_l(it)-1] for it in range(n_coeffs)])

alphas[0:dip] = alpha_dip**2
taus[0:dip] = 1 / omega

dt = np.abs(knots[:, None] - knots[None, :])
diag = alphas
itau = np.diag(1./taus)
frac = dt[:, None, :, None] * itau[None, :, None, :]
frac = np.abs(frac)
# first axis are times, second axis are coeffs
cov = (1 + frac) * np.exp(-frac) * np.diag(diag)[None, :, None, :]
cov = np.diagonal(cov, axis1=1, axis2=3).copy()

# axial dipole with different matrix:
for it in range(dip):
    cov[:, :, it] = alpha_dip**2 * 0.5 / xi * (
        (xi + chi) * np.exp((xi - chi)*dt)
        + (xi - chi) * np.exp(-(xi + chi)*dt)
    )

# Set the number of knots to be replaced by the reference model
n_ref = 3

_Kalmag = np.genfromtxt(
    SCRIPT_DIR + '/../dat/Kalmag_CORE_MEAN_Radius_6371.2.txt'
)
# Ref coeffs should be of shape (n_coeffs, n_ref)
ref_coeffs = _Kalmag[1:n_coeffs+1] / 1000

chol = np.zeros((n_coeffs, len(knots)-n_ref, len(knots)-n_ref))

for it in range(n_coeffs):
    _cov = np.copy(cov[:len(knots)-n_ref, :len(knots)-n_ref, it])
    _cor = cov[:, len(knots)-n_ref:, it]
    _icov = np.linalg.inv(cov[len(knots)-n_ref:, len(knots)-n_ref:, it])
    prior_mean[it] += _cor @ _icov @ \
        (ref_coeffs[it] - prior_mean[it, len(knots)-n_ref:])

    _cov -= _cor[:len(knots)-n_ref] @ _icov @ _cor[:len(knots)-n_ref].T
    chol[it, :, :] = np.linalg.cholesky(_cov)

# overwrite for usage in pyMC model
prior_mean = prior_mean[:, :len(knots)-n_ref]
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
bins = np.arange(ref_min, t_max+step_solar_fine, step_solar_fine)
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
# thermoremament magnetic data
rawData = pd.read_csv(
    SCRIPT_DIR + '/../dat/afm9k2_data.csv',
)
rawData['FID'] = 'afm9k.2_data'

data = Data(rawData)

idx_D = np.asarray(data.idx_D, dtype=int)
idx_I = np.asarray(data.idx_I, dtype=int)
idx_F = np.asarray(data.idx_F, dtype=int)

z_at = data.inputs

base = dsh_basis(lmax, z_at)
base = base.reshape(lmax2N(lmax), z_at.shape[1], 3)

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
if use_11year_cycle:
    # Tau = 2, using Brehm when possible
    annual_C14_data = pd.read_excel(
        SCRIPT_DIR
        + '/../dat/'
        + 'ProductionRates100Versions_Matern3_2sigma2tau2.xlsx',
        skiprows=7,
    )
    annual_C14_data['t'] = 1950 + annual_C14_data['age -yr BP']
    annual_ensemble = annual_C14_data.values[:, 5:-1]
    annual_C14_data['C14'] = annual_ensemble.mean(axis=1)
    annual_C14_data['dC14'] = annual_ensemble.std(axis=1)
    annual_C14_data['C14'] = moving_average(annual_C14_data, 2)

    annual_C14_data = annual_C14_data[annual_C14_data['t'] > -1000]

    annual_C14_data.reset_index(inplace=True, drop=True)
    annual_C14_data['dC14'] = 0.1

    idxs = radData.query(f't > {min(annual_C14_data["t"])}').index
    radData.loc[idxs, 'C14'] = np.nan
    radData['dC14'] = np.nan
    merge_rows = []
    for _, row in annual_C14_data[['t', 'C14', 'dC14']].iterrows():
        # try:
        idx = radData.query(f't == {row["t"]}').index
        if 0 == len(idx):
            merge_rows.append(row)
        elif len(idx) == 1:
            radData.loc[idx, 'C14'] = row['C14']
            # radData.loc[idx, 'dC14'] = row['dC14']
            radData.loc[idx, 'dC14'] = 0.05      # * np.abs(row['C14'])
        else:
            raise ValueError(f'Multiple entries found for t = {row["t"]}')

    radData = pd.concat(
        [
            pd.DataFrame(merge_rows),
            radData,
        ],
    )

radData.sort_values(by='t', inplace=True)
radData.reset_index(inplace=True, drop=True)
idx = radData['dC14'].isna()
radData.loc[idx, 'dC14'] = 0.05     # * np.abs(radData.loc[idx, 'C14'])
# Use 5 % errors for all C14 records
# radData['dC14'] = 0.05 * np.abs(radData['C14'])
radData['dBe10_NH'] = 0.1   # * np.abs(radData['Be10_NH'])
radData['dBe10_SH'] = 0.1   # * np.abs(radData['Be10_SH'])

# exclude solar storm
idx = radData.query('773.5 <= t and t <= 775.5').index
radData.loc[idx, 'C14'] = np.nan

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
    ax.set_xlabel('time [yrs.]')
    ax.set_ylabel('$^{14}$C production (normalized)')

    fig.tight_layout()

    plt.show()
