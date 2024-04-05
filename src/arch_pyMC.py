import numpy as np

import pymc as pm

from pytensor import tensor as pt

from utils import interp

from common import (
    knots,
    n_coeffs,
    n_ref,
    ref_coeffs as _ref_coeffs,
    prior_mean,
    chol,
    z_at,
    base,
    data,
    idx_D,
    idx_I,
    idx_F,
    tDir,
    tInt,
    alpha_nu,
    beta_nu,
)

ref_coeffs = pt.as_tensor(_ref_coeffs)
base_tensor = pt.as_tensor(base.transpose(1, 0, 2))

with pm.Model() as mcModel:
    nus = pm.Gamma(
        'nu',
        alpha=alpha_nu,
        beta=beta_nu,
        size=3,
    )

    t_cent = pm.Normal(
        't_cent',
        mu=0,
        sigma=1,
        size=data.n_inp,
    )
    ts = z_at[3] - t_cent * np.sqrt(data.errs_T_raw)

    g_cent = pm.Normal(
        'g_cent',
        mu=0,
        sigma=1,
        size=(n_coeffs, len(knots)-n_ref),
    )

    gs_at = pt.batched_dot(chol, g_cent) + prior_mean

    gs_at_knots = pt.horizontal_stack(
        gs_at,
        ref_coeffs,
    )

    gs = interp(
        ts,
        knots,
        gs_at_knots.T,
    )

    nez_tensor = pt.batched_dot(gs, base_tensor)

    _f = pt.sqrt(
        pt.sum(
            pt.square(nez_tensor),
            axis=1,
        ),
    )

    _i = pt.rad2deg(
        pt.arcsin(
            nez_tensor[:, 2] / _f,
        ),
    )

    rI = pm.Deterministic(
        'rI',
        (_i[idx_I] - data.outputs[len(idx_D):len(idx_D)+len(idx_I)]) /
        np.sqrt(
            data.errs[len(idx_D):len(idx_D)+len(idx_I)]
            + tDir**2
        ),
    )

    d = pt.rad2deg(
        pt.arctan2(
            nez_tensor[idx_D, 1],
            nez_tensor[idx_D, 0],
        ),
    )

    _rD = d - data.outputs[:len(idx_D)]
    rD = pm.Deterministic(
        'rD',
        (_rD - 360 * (_rD > 180) + 360 * (-180 > _rD)) / pt.sqrt(
            data.errs[:len(idx_D)]
            + (tDir / pt.cos(pt.deg2rad(_i[idx_D])))**2
        ),
    )

    rF = pm.Deterministic(
        'rF',
        (_f[idx_F] - data.outputs[len(idx_D)+len(idx_I):]) / np.sqrt(
            data.errs[len(idx_D)+len(idx_I):]
            + tInt**2
        ),
    )

    rD_obs = pm.StudentT(
        'd_obs',
        nu=1 + nus[0],
        mu=rD,
        sigma=1.,
        observed=np.zeros(len(idx_D)),
    )

    rI_obs = pm.StudentT(
        'i_obs',
        nu=1 + nus[1],
        mu=rI,
        sigma=1.,
        observed=np.zeros(len(idx_I)),
    )

    rF_obs = pm.StudentT(
        'f_obs',
        nu=1 + nus[2],
        mu=rF,
        sigma=1.,
        observed=np.zeros(len(idx_F)),
    )


if __name__ == '__main__':
    from common import mcmc_params
    from pymc.sampling import jax as pmj
    # from numpyro.infer.initialization import init_to_median

    with mcModel:
        idata = pmj.sample_numpyro_nuts(
            mcmc_params['n_samps'],
            tune=mcmc_params['n_warmup'],
            progressbar=True,
            # cores=1,
            chains=mcmc_params['n_chains'],
            target_accept=mcmc_params['target_accept'],
            postprocessing_backend='cpu',
            nuts_kwargs={
                # 'dense_mass': [('betas'), ('NEZ_residual'), ('t_cent')],
                # 'init_strategy': init_to_median,
            },
        )

    idata.to_netcdf('../dat/arch_fdp2_max_result.nc')
