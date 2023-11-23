""" Collection of parameters and settings common to the stan and pyMC approach.
"""

import numpy as np
from pandas import read_table, cut

from scipy.signal import butter, sosfiltfilt

from paleokalmag.utils import dsh_basis
from paleokalmag.data_handling import read_data, Data

from pymagglobal.utils import REARTH, lmax2N, i2lm_l, scaling

from utils import sqe_kernel


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


def prod_Be10(dm, phi):
    Q = 1 / (
        5.576
        + 1.896*dm
        + 0.01375*phi
        - 0.01299*dm**2
        + 0.001494*dm*phi
        - 2.851e-07*phi**2
    )

    return Q


# -----------------------------------------------------------------------------
# Model parameters
# GP Settings
lmax = 5
# R = 2800

# 2023-11-22
# New parameters, estimated with cleaned dataset and gradients
dip = 3
R = 2800
gamma_0 = -32.79
alpha_dip = 39.60
omega = 1 / 153.08602220031185
xi = 1 / 48.71563374353433
chi = np.sqrt(xi**2 + omega**2)
alpha_wodip = 94.54
tau_wodip = 513.9

scl = np.flip(np.unique(scaling(R, REARTH, lmax)))
_alphas = np.ones(lmax) * alpha_wodip
_alphas *= scl
alpha_dip *= scl[0]
_taus = tau_wodip / (np.arange(lmax)+1)

# 2023-09-13
# Try two parameter axial dipole, pfm9k.2 parameters
# dip = 1
# R = REARTH
# gamma_0 = -32.5
# alpha_dip = 10
# omega = 1/741
# chi = 1/138
# xi = np.sqrt(chi**2 - omega**2)
# _alphas = [
#     3.5,
#     1.765,
#     1.011,
#     0.455,
#     0.177,
# ]
# _taus = [
#     200,
#     133,
#     174,
#     137,
#     95,
# ]


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
t_max = 2000
step = 50
step_solar = 22

# MCMC parameters
mcmc_params = {
    'n_samps': 500,
    'n_warmup': 1000,
    'n_chains':  4,
    'target_accept': 0.95,
}

# Other parameters
tDir = (3.4 + 2) * 57.3 / 140   # Directional truncation error in Deg.
# tDir = 1.4                    # smaller value after investigation
tInt = 2.                       # Intensity truncation error in uT
alpha_nu = 2.
beta_nu = 0.1

# Derived parameters
n_coeffs = lmax2N(lmax)
knots = np.arange(
    t_min,
    t_max + step,
    step,
)
knots_solar = np.flip(
    np.arange(
        t_max + step_solar,
        t_min,
        -step_solar,
    )
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
mean_solar = 477.5
sigma_solar = 191
tau_solar = 25.6

solar_constr = read_table(
    '../dat/US10_phi_mon_tab_230907.txt',
    delim_whitespace=True,
    skiprows=23,
    header=None,
    dtype=float,
    engine='python',
    names=[
        'Year',
        'Phi (MV)',
    ],
)
bins = knots_solar[-6:]+11
solar_constr['Interval'] = cut(solar_constr['Year'], bins)

ref_solar_years = []
ref_solar = []

for group in solar_constr.groupby('Interval', observed=True):
    ref_solar_years.append(group[0].mid)
    ref_solar.append(group[1]['Phi (MV)'].mean())

ref_solar_years = np.array(ref_solar_years)
ref_solar = np.array(ref_solar)

t_const = min(ref_solar_years)
ind_const = np.argmax(t_const <= knots_solar)
n_ref_solar = len(knots_solar) - ind_const

cov_obs = sqe_kernel(
    ref_solar_years,
    tau=tau_solar,
    sigma=sigma_solar,
)
cor_obs = sqe_kernel(
    knots_solar,
    ref_solar_years,
    tau=tau_solar,
    sigma=sigma_solar,
)
cov_solar = sqe_kernel(
    knots_solar,
    sigma=sigma_solar,
    tau=tau_solar,
)

_icov_obs = np.linalg.inv(cov_obs + 1e-4*np.eye(len(ref_solar_years)))
prior_mean_solar = mean_solar + cor_obs @ _icov_obs @ \
    (ref_solar - mean_solar)
cov_solar = cov_solar - cor_obs @ _icov_obs @ cor_obs.T

prior_mean_solar = prior_mean_solar[:-n_ref_solar]
chol_solar = np.linalg.cholesky(cov_solar+1e-6*np.eye(len(knots_solar)))[
    :-n_ref_solar,
    :-n_ref_solar,
]

# -----------------------------------------------------------------------------
# Data setup
with np.load(
    '../dat/rejection_list.npz',
    allow_pickle=True,
) as fh:
    to_reject = fh['to_reject']

rawData = read_data(
    '../dat/archkalmag_data.csv',
    rejection_lists=to_reject[:, 0:3],
)
rawData = rawData.query("dt <= 100")
rawData = rawData.query(f'{t_min} <= t <= {t_max}')
# rawData = rawData.sample(n=1000, random_state=161)

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

for rem in rem_uids:
    rawData.drop(
        index=rawData.query(f"UID == {rem}").index,
        inplace=True,
    )

rawData.reset_index(inplace=True, drop=True)

data = Data(rawData)

idx_D = np.asarray(data.idx_D, dtype=int)
idx_I = np.asarray(data.idx_I, dtype=int)
idx_F = np.asarray(data.idx_F, dtype=int)

z_at = data.inputs

base = dsh_basis(lmax, z_at)
base = base.reshape(lmax2N(lmax), z_at.shape[1], 3)

radData = read_table(
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

idx_GL = np.asarray(radData.query('C14 == C14').index, dtype=int)
idx_NH = np.asarray(radData.query('Be10_NH == Be10_NH').index, dtype=int)
idx_SH = np.asarray(radData.query('Be10_SH == Be10_SH').index, dtype=int)
