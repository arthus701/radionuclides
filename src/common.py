""" Collection of parameters and settings common to the stan and pyMC approach.
"""

import numpy as np
import pandas as pd

from scipy.signal import butter, sosfiltfilt

from paleokalmag.utils import dsh_basis
from paleokalmag.data_handling import Data
# from paleokalmag.data_handling import read_data

from pymagglobal.utils import REARTH, lmax2N, i2lm_l    # , scaling

from utils import matern_kernel, moving_average


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


# -----------------------------------------------------------------------------
# Model parameters
# GP Settings
lmax = 5
# R = 2800

# 2023-11-22
# New parameters, estimated with cleaned dataset and gradients
# dip = 3
# R = 2800
# gamma_0 = -32.79
# alpha_dip = 39.60
# omega = 1 / 153.08602220031185
# xi = 1 / 48.71563374353433
# chi = np.sqrt(xi**2 + omega**2)
# alpha_wodip = 94.54
# tau_wodip = 513.9
#
# scl = np.flip(np.unique(scaling(R, REARTH, lmax)))
# _alphas = np.ones(lmax) * alpha_wodip
# _alphas *= scl
# alpha_dip *= scl[0]
# _taus = tau_wodip / (np.arange(lmax)+1)

# 2023-09-13
# Try two parameter axial dipole, pfm9k.2 parameters
dip = 1
R = REARTH
gamma_0 = -32.5
alpha_dip = 10
omega = 1/741
chi = 1/138
xi = np.sqrt(chi**2 - omega**2)
_alphas = [
    3.5,
    1.765,
    1.011,
    0.455,
    0.177,
]
_taus = [
    200,
    133,
    174,
    138,
    95,
]


# with np.load('../dat/hyperparameters.npz', allow_pickle=True) as fh:
#     res = fh['res'].item()
#
# gamma_0 = res.x[0]
# alpha_dip = res.x[1]
# tau_dip = res.x[2]
# alpha_wodip = res.x[3]
# tau_wodip = res.x[4]
# rho = res.x[5]

# Range and resolution
t_min = -7000
t_solar_fine = -1100
t_max = 2000
step = 50
step_solar_coarse = 22
step_solar_fine = 2

# MCMC parameters
mcmc_params = {
    'n_samps': 500,
    'n_warmup': 1000,
    'n_chains': 4,
    'target_accept': 0.95,
}

# Other parameters
# tDir = (3.4 + 2) * 57.3 / 140   # Directional truncation error in Deg.
# tDir = 1.4                    # smaller value after investigation
# tInt = 2.                       # Intensity truncation error in uT
# set to zero when using Andreas' dataset
tDir = 0
tInt = 0
alpha_nu = 2.
beta_nu = 0.1

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

alphas = np.array([_alphas[i2lm_l(it)-1]**2 for it in range(n_coeffs)])
taus = np.array([_taus[i2lm_l(it)-1] for it in range(n_coeffs)])

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

_Kalmag = np.genfromtxt('../dat/Kalmag_CORE_MEAN_Radius_6371.2.txt')
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
# Check multimodal prior
# Set higher (~600) to check for less variation
mean_solar = 650
sigma_solar = 191
tau_solar = 25.6
tau_solar_fast = 4
tau_fast_period = 11.

# Extract variations on regular and fast scale
solar_constr = pd.read_table(
    '../dat/US10_phi_mon_tab_230907.txt',
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
prior_mean_solar = mean_solar + cor_obs @ _icov_obs @ \
    (ref_solar_df['Phi avg.'] - mean_solar)
cov_solar = cov_solar - cor_obs @ _icov_obs @ cor_obs.T

# prior_mean_solar = np.ones(len(knots_solar)) * mean_solar
prior_mean_solar = prior_mean_solar[:-n_ref_solar]
chol_solar = np.linalg.cholesky(cov_solar+1e-6*np.eye(len(knots_solar)))[
    :-n_ref_solar,
    :-n_ref_solar,
]


# -----------------------------------------------------------------------------
# Data setup
# with np.load(
#     '../dat/rejection_list.npz',
#     allow_pickle=True,
# ) as fh:
#     to_reject = fh['to_reject']
#
# rawData = read_data(
#     '../dat/archkalmag_data.csv',
#     rejection_lists=to_reject[:, 0:3],
# )
# rawData = rawData.query("dt <= 100")
# rawData = rawData.query(f'{t_min} <= t <= {t_max}')
# rawData = rawData.sample(n=1000, random_state=161)
# Test minimal errors.
# rawData['dD'].where(rawData['dD'] > 10, other=10, inplace=True)


# UIDs of non-converging archeo times
rem_uids = [
    12064,  # Volcanic from central America, around 1000 BCE
    12065,  # Volcanic from central America, around 1000 BCE
    12066,  # Volcanic from central America, around 1000 BCE
    11064,  # Archeo from central America, around 1000 BCE
    10942,  # Archeo from central America, around 1000 BCE
    8568,   # Archeo from central America, around 1000 BCE
    11049,  # Archeo from central America, around 1000 BCE
    10962,  # Archeo from central America, around 1000 BCE
    10982,  # Archeo from central America, around 1000 BCE
    # 564,    # Hawaii
    # 8317,   # Seoul
    # 11402,  # Chuncheon, South Korea
    # 11035,  # Archeo from central America, 800 CE
    # 11071,  # Archeo from central America, 800 CE
    # 11034,  # Archeo from central America, 900 CE
    # 10926,  # Archeo from central America, 1100 CE
    # 12099,  # Archeo from Pau d'Alho Cave, Brazil 1189 CE
]

# for rem in rem_uids:
#     rawData.drop(
#         index=rawData.query(f"UID == {rem}").index,
#         inplace=True,
#     )

# There's a mistake in GEOMAGIA. Several records from Ecuador are reported with
# the wrong age, BP was confused with BCE.
# https://doi.org/10.1016/j.jsames.2020.102733
age_fix_uids = [
    13229,
    13230,
    13231,
    13232,
    13234,
    13235,
    13236,
    13237,
    13238,
    13239,
    13240,
    13241,
    13242,
    13243,
]
# for age_fix in age_fix_uids:
#     idx = rawData.query(f"UID == {age_fix}").index
#     rawData.loc[idx, 't'] = 1950 + rawData.loc[idx, 't']
#
# rawData.reset_index(inplace=True, drop=True)

rawData = pd.read_csv(
    '../dat/afm9k2_data.csv',
)
rawData['FID'] = 'from Andreas'

data = Data(rawData)

idx_D = np.asarray(data.idx_D, dtype=int)
idx_I = np.asarray(data.idx_I, dtype=int)
idx_F = np.asarray(data.idx_F, dtype=int)

z_at = data.inputs

base = dsh_basis(lmax, z_at)
base = base.reshape(lmax2N(lmax), z_at.shape[1], 3)

radData = pd.read_table(
    '../dat/CRN_9k_230922.txt'
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
# Tau = 4, old as of 2025-05-13
# annual_C14_data = pd.read_excel(
#     '../dat/14C_production_rate_tau4.xlsx',
# )
# annual_C14_data['t'] = 1950 - annual_C14_data['age yr BP']
# annual_C14_data.rename(
#     columns={
#         'P14 averge ': 'C14',
#         'P14 1 sigma': 'dC14',
#     },
#     inplace=True,
# )

# Try using annual data from Brehm et al. (2021)
# annual_C14_data = pd.read_excel(
#     '../dat/41561_2020_674_MOESM2_ESM.xlsx',
#     sheet_name='Figure1b',
# )
# annual_C14_data.dropna(how='all', inplace=True)
# annual_C14_data.rename(
#     columns={
#         'ETH Simulated time (yr AD)': 't',
#         'ETH normalized production': 'C14',
#     },
#     inplace=True,
# )
# Tau = 1, using Brehm when possible
annual_C14_data = pd.read_excel(
    '../dat/ProductionRates100Versions.xlsx',
    skiprows=7,
)
annual_C14_data['t'] = 1950 + annual_C14_data['age -yr BP']
annual_ensemble = annual_C14_data.values[:, 1:-1]
annual_C14_data['C14'] = annual_ensemble.mean(axis=1)
annual_C14_data['dC14'] = annual_ensemble.std(axis=1)
annual_C14_data.reset_index(inplace=True, drop=True)


# annual_C14_data['C14'] = moving_average(annual_C14_data, 2)
annual_C14_data = annual_C14_data[annual_C14_data['t'] > -1000]

annual_C14_data.reset_index(inplace=True, drop=True)
# XXX
# annual_C14_data['dC14'] = 0.05
annual_C14_data['dC14'] = \
    0.08 * annual_C14_data['dC14'] / annual_C14_data['dC14'].mean()

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
# radData['dC14'] = np.nan

radData.sort_values(by='t', inplace=True)
radData.reset_index(inplace=True, drop=True)
idx = radData['dC14'].isna()
radData.loc[idx, 'dC14'] = 0.05     # * np.abs(radData.loc[idx, 'C14'])
# XXX Use 5 % errors for all C14 records
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
