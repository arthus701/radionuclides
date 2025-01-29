import sys

import numpy as np
import arviz as az
from common import (
    n_coeffs,
    knots,
    knots_solar,
)

try:
    prefix = sys.argv[1]
except IndexError:
    prefix = "radio"

iData = az.InferenceData.from_netcdf(
    f'../out/{prefix}_result.nc',
)

summary = az.summary(iData)
summary.to_csv(f'../out/{prefix}_summary.csv')

coeffs = np.array(
    iData.posterior['gs_at_knots'],
).reshape(-1, n_coeffs, len(knots))
coeffs = coeffs.transpose(1, 2, 0)

if 'arch' in prefix:
    np.savez(
        f'../out/{prefix}_ensemble.npz',
        knots=knots,
        coeffs=coeffs,
    )
    exit()


sm = np.array(
    iData.posterior['sm_at_knots'],
).reshape(-1, len(knots_solar))

np.savez(
    f'../out/{prefix}_ensemble.npz',
    knots=knots,
    coeffs=coeffs,
    knots_solar=knots_solar,
    solar=sm.T,
)
