import numpy as np

import pymc as pm

from pytensor import tensor as pt

from utils import interp1d

from common import (
    mean_solar,
    sigma_solar,
    knots_solar,
    n_ref_solar,
    ref_solar,
    chol_solar,
    # chol_solar_longterm,
    prior_mean_solar,
    radData,
)

with pm.Model() as mcModel:
    # -------------------------------------------------------------------------
    # Solar modulation
    sm_cent = pm.Normal(
        'sm_cent',
        mu=0,
        sigma=1,
        size=(len(knots_solar)-n_ref_solar,),
    )

    # correlated samples
    sm_at = chol_solar @ sm_cent + prior_mean_solar

    # uniform correlated samples via cdf (normal w. prior mean and prior var)
    sm_at -= mean_solar
    sm_at /= sigma_solar
    sm_at_uniform = pt.exp(
        pm.logcdf(
            pm.Normal.dist(
                mu=0.,
                sigma=1.,
            ),
            sm_at,
        )
    )

    # transform to correlated bimodal via inverse cdf
    kappa = 0.15
    # tL_cent = pm.HalfNormal(
    #     'tL_cent',
    #     sigma=1.,
    # )
    # tL = 200 * tL_cent
    # tR = pm.TruncatedNormal(
    #     'tR',
    #     mu=1000.,
    #     sigma=200.,
    #     lower=0.,
    # )

    tL = 100
    tR = 1000

    alpha = np.array(
        [
            0.25, 2.5,
        ]
    )
    beta = np.array(
        [
            0.25, 10.,
        ]
    )

    cdf = pt.as_tensor(np.linspace(0, 1, 2001))

    inverse_approx = (
        tL + tR * (
            (1 - kappa) * pt.exp(
                pm.logcdf(
                    pm.Beta.dist(
                        alpha=alpha[0],
                        beta=beta[0],
                    ),
                    cdf,
                )
            )
            + kappa * pt.exp(
                pm.logcdf(
                    pm.Beta.dist(
                        alpha=alpha[1],
                        beta=beta[1],
                    ),
                    cdf,
                )
            )
        )
    )

    sm_at_bimod = interp1d(
        sm_at_uniform,
        cdf,
        inverse_approx,
    )

    # add longterm component
    # sm_cent_longterm = pm.Normal(
    #     'sm_cent_longterm',
    #     mu=0,
    #     sigma=1,
    #     size=(len(knots_solar)-n_ref_solar,),
    # )
    # sm_longterm_scale = pm.Gamma(
    #     'sm_longterm_scale',
    #     alpha=3,
    #     beta=3/200,
    #     size=1,
    # )
    # sm_at_longterm = sm_longterm_scale * chol_solar_longterm @ sm_cent_longterm
    sm_at_both = sm_at_bimod

    # penalize smaller than zero
    # zero_bound = pm.math.sum(
    #     pm.math.log(
    #         pm.math.sigmoid(
    #             1e-2 * (sm_at_both - 150)
    #         )
    #     )
    # )
    # pm.Potential('zero_bound', zero_bound)

    sm_at_knots = pm.Deterministic(
        'sm_at_knots',
        pm.math.concatenate(
            (
                sm_at_both,
                ref_solar,
            ),
        ),
    )

    sm_rad = interp1d(
        radData['t'],
        knots_solar,
        sm_at_knots,
    )


if __name__ == '__main__':
    from common import mcmc_params
    from pymc.sampling import jax as pmj
    from numpyro.infer.initialization import init_to_mean

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
                'init_strategy': init_to_mean,
            },
        )

    idata.to_netcdf(
        f'../dat/solar_prior_kappa_{kappa:.2f}_result.nc'
    )
