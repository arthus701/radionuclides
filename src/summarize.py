import sys

import numpy as np
import arviz as az
from common import (
    n_coeffs,
    knots,
    prior_mean,
    chol,
    n_ref,
    ref_coeffs,
    knots_solar,
    prior_mean_solar,
    chol_solar,
    n_ref_solar,
    ref_solar,
)

try:
    prefix = sys.argv[1]
except IndexError:
    prefix = "radio"

iData = az.InferenceData.from_netcdf(
    f'../dat/{prefix}_result.nc',
)

summary = az.summary(iData)
summary.to_csv(f'../dat/{prefix}_summary.csv')

g_cent = np.array(
    iData.posterior['g_cent'],
).reshape(-1, n_coeffs, len(knots)-n_ref)

coeffs = np.einsum(
    'ijk, ...ik -> ij...',
    chol,
    g_cent,
) + prior_mean[:, :, None]

coeffs = np.hstack(
    (
        coeffs,
        np.repeat(ref_coeffs[:, :, None], coeffs.shape[2], axis=2),
    )
)

if prefix == 'arch':
    np.savez(
        f'../dat/{prefix}_ensemble.npz',
        knots=knots,
        coeffs=coeffs,
    )


sm_cent = np.array(
    iData.posterior['sm_cent'],
).reshape(-1, len(knots_solar)-n_ref_solar)

sm = np.einsum(
    'ij, ...j -> i...',
    chol_solar,
    sm_cent,
) + prior_mean_solar[:, None]

sm = np.vstack(
    (
        sm,
        np.repeat(ref_solar[:, None], sm.shape[1], axis=1),
    )
)

np.savez(
    f'../dat/{prefix}_ensemble.npz',
    knots=knots,
    coeffs=coeffs,
    knots_solar=knots_solar,
    solar=sm,
)
