import numpy as np

import pymc as pm

from pytensor import tensor as pt

from utils import interp1d, npinterp

from parameters import (
    mu_solar,
    sigma_solar,
    # tau_solar,
    # tau_solar_fast,
    tau_fast_period,
)

from common import (
    knots_solar,
    knots_solar_fine,
    ref_solar_years,
    ref_solar,
    ref_solar_years_fine,
    ref_solar_fine,
    chol_solar,
    radData,
    annual_C14_data,
    idx_GL,
    idx_NH,
    idx_SH,
    prod_Be10,
    calc_HPA,
    prod_C14,
)

# from brehm_data import brehm_data as annual_C14_data
from annual_be10_data import annual_Be10_data

from fast_component import SolarFastComponent
# from periodic_component import SolarPeriodicComponent

fac = 0.63712**3

with np.load('../out/radio_ensemble.npz') as fh:
    knots = fh['knots']
    coeffs = fh['coeffs']

gs_at_knots = coeffs.mean(axis=-1)

with pm.Model() as mcModel:
    # -------------------------------------------------------------------------
    # Solar modulation
    sm_cent = pm.Normal(
        'sm_cent',
        mu=0,
        sigma=1,
        size=len(knots_solar),
    )
    # XXX One could use unscaled samples directly
    # correlated samples
    sm_at = chol_solar @ sm_cent + mu_solar
    # uniform correlated samples via cdf (normal w. prior mean and prior var)
    sm_at -= mu_solar
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
    kappa = pm.Beta(
        'kappa',
        alpha=1.,
        beta=7.,
    )
    tL_cent = pm.HalfNormal(
        'tL_cent',
        sigma=1.,
    )
    tL = 200 * tL_cent
    tR = pm.TruncatedNormal(
        'tR',
        mu=1000.,
        sigma=200.,
        lower=0.,
    )

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

    sm_bimod = interp1d(
        sm_at_uniform,
        cdf,
        inverse_approx,
    )
    # penalize smaller than zero
    # zero_bound = pm.math.sum(
    #     pm.math.log(
    #         pm.math.sigmoid(
    #             1e-1 * (sm_bimod - 150)
    #         )
    #     )
    # )
    # pm.Potential('zero_bound', zero_bound)

    sm_at_knots = pm.Deterministic(
        'sm_at_knots',
        sm_bimod,
    )

    sm_at_ref = interp1d(
        ref_solar_years,
        knots_solar,
        sm_at_knots,
    )

    pm.Normal(
        'sm_anchor_avg',
        mu=sm_at_ref - ref_solar,
        sigma=50,
        observed=np.zeros_like(ref_solar),
    )

    sm_rad = interp1d(
        radData['t'].values,
        knots_solar,
        sm_at_knots,
    )

    gs_rad = npinterp(
        radData['t'].values,
        knots,
        gs_at_knots[:4].T,
    )

    dm = fac * pt.sqrt(
        pt.sum(
            pt.square(gs_rad[:, :3]),
            axis=1,
        ),
    )
    # g_2_0 = gs_rad[:, 3]
    q_GL = prod_Be10(dm, sm_rad)
    hpa = calc_HPA(gs_rad[:, 3], sm_rad)

    q_GL_cal = prod_Be10(dm.mean(), mu_solar)

    q_C14 = prod_C14(dm, sm_rad)
    q_C14_cal = prod_C14(dm.mean(), mu_solar)

    s_fac = pm.Normal(
        's_fac',
        sigma=1.,
        size=3,
    )

    cal_nh = q_GL_cal * (1 + s_fac[0]*0.05)
    cal_sh = q_GL_cal * (1 + s_fac[1]*0.05)
    cal_c14 = q_C14_cal * (1 + s_fac[2]*0.05)

    rNH = pm.Deterministic(
        'rNH',
        (
            (q_GL * (1 + hpa / 2.))[idx_NH] / cal_nh
            - radData.loc[idx_NH, 'Be10_NH'].values
        ) / radData.loc[idx_NH, 'dBe10_NH'].values,
    )
    be10_nh_obs = pm.Normal(
        'Be10_NH',
        # nu=1 + nus[3],
        mu=rNH,
        sigma=1.,
        observed=np.zeros(len(idx_NH)),
    )

    rSH = pm.Deterministic(
        'rSH',
        (
            (q_GL * (1 - hpa / 2.))[idx_SH] / cal_sh
            - radData.loc[idx_SH, 'Be10_SH'].values
        ) / radData.loc[idx_SH, 'dBe10_SH'].values,
    )
    be10_sh_obs = pm.Normal(
        'Be10_SH',
        # nu=1 + nus[4],
        mu=rSH,
        sigma=1.,
        observed=np.zeros(len(idx_SH)),
    )

    rC14 = pm.Deterministic(
        'rC14',
        (
            q_C14[idx_GL] / cal_c14
            - radData.loc[idx_GL, 'C14'].values
        ) / radData.loc[idx_GL, 'dC14'].values,
    )
    c14_obs = pm.Normal(
        'C14',
        # nu=1 + nus[5],
        mu=rC14,
        sigma=1.,
        observed=np.zeros(len(idx_GL)),
    )

    # Model 11-year cycle as redidual, using annual data
    # solar_11 = SolarPeriodicComponent(
    #     knots_solar_fine,
    #     period_solar=tau_fast_period,
    #     # tau_solar=tau_solar,
    #     tau_solar=20.,
    #     # ref_solar_knots=ref_solar_df['t'].values,
    #     # ref_solar=ref_solar_df['Phi fast'].values,
    # )
    solar_11 = SolarFastComponent(
        knots_solar_fine,
        # tau_solar=tau_solar,
        tau_solar=2.,
        # ref_solar_knots=ref_solar_df['t'].values,
        # ref_solar=ref_solar_df['Phi fast'].values,
    )
    sm_fast_at_knots = solar_11.get_sm_at_fast()

    sm_fast_at_ref = interp1d(
        ref_solar_years_fine,
        knots_solar_fine,
        sm_fast_at_knots,
    )

    pm.Normal(
        'sm_anchor_fine',
        mu=sm_fast_at_ref - ref_solar_fine,
        sigma=20,
        observed=np.zeros_like(ref_solar_fine),
    )
    # -------------------------------------------------------------------------
    # Annual C14
    gs_rad_fine_C14 = npinterp(
        annual_C14_data['t'].values,
        knots,
        gs_at_knots[:4].T,
    )

    dm_fine_C14 = fac * pt.sqrt(
        pt.sum(
            pt.square(gs_rad_fine_C14[:, :3]),
            axis=1,
        ),
    )

    sm_rad_fine_C14 = interp1d(
        annual_C14_data['t'].values,
        knots_solar,
        sm_at_knots,
    )
    sm_11_fine_C14 = interp1d(
        annual_C14_data['t'].values,
        knots_solar_fine,
        sm_fast_at_knots,
    )

    q_C14_fine = prod_C14(dm_fine_C14, sm_rad_fine_C14)
    q_C14_11_fine = prod_C14(dm_fine_C14, sm_rad_fine_C14 + sm_11_fine_C14)

    residual_q_C14 = q_C14_11_fine - q_C14_fine

    rC14_11 = pm.Deterministic(
        'rC14_11',
        (
            residual_q_C14 / cal_c14
            - annual_C14_data['C14_detrended'].values
        ) / annual_C14_data['dC14'].values,
    )
    c14_11_obs = pm.Normal(
        'C14_11',
        # nu=1 + nus[5],
        mu=rC14_11,
        sigma=1.,
        observed=np.zeros(len(annual_C14_data)),
    )

    # -------------------------------------------------------------------------
    # Annual Be10
    gs_rad_fine_Be10 = npinterp(
        annual_Be10_data['t'].values,
        knots,
        gs_at_knots[:4].T,
    )

    dm_fine_Be10 = fac * pt.sqrt(
        pt.sum(
            pt.square(gs_rad_fine_Be10[:, :3]),
            axis=1,
        ),
    )

    sm_rad_fine_Be10 = interp1d(
        annual_Be10_data['t'].values,
        knots_solar,
        sm_at_knots,
    )
    sm_11_fine_Be10 = interp1d(
        annual_Be10_data['t'].values,
        knots_solar_fine,
        sm_fast_at_knots,
    )

    q_Be10_fine = prod_Be10(dm_fine_Be10, sm_rad_fine_Be10)
    q_Be10_11_fine = prod_Be10(
        dm_fine_Be10,
        sm_rad_fine_Be10 + sm_11_fine_Be10
    )

    residual_q_Be10 = q_Be10_11_fine - q_Be10_fine

    rBe10_11_NH = pm.Deterministic(
        'rBe10_11_NH',
        (
            residual_q_Be10 / cal_nh
            - annual_Be10_data['Be10_detrended'].values
        ) / annual_Be10_data['dBe10'].values,
    )
    # be10_11_obs = pm.Normal(
    #     'Be10_11_NH',
    #     # nu=1 + nus[5],
    #     mu=rBe10_11_NH,
    #     sigma=1.,
    #     observed=np.zeros(len(annual_Be10_data)),
    # )

if __name__ == '__main__':
    from pymc.sampling import jax as pmj
    from numpyro.infer.initialization import init_to_mean

    from parameters import mcmc_params, prefix

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

    idata.to_netcdf(f'../out/{prefix}result.nc')
