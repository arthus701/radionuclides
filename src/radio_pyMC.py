import numpy as np

import pymc as pm

from pytensor import tensor as pt

from utils import interp, interp1d

from common import (
    knots,
    n_coeffs,
    n_ref,
    ref_coeffs as _ref_coeffs,
    prior_mean,
    gamma_0,
    chol,
    z_at,
    base,
    data,
    idx_D,
    idx_I,
    idx_F,
    tDir,
    tInt,
    # alpha_nu,
    # beta_nu,
    mean_solar,
    sigma_solar,
    tau_solar_fast,
    knots_solar,
    knots_solar_fine,
    n_ref_solar,
    ref_solar_df,
    chol_solar,
    prior_mean_solar,
    radData,
    idx_GL,
    idx_NH,
    idx_SH,
    prod_Be10,
    calc_HPA,
    prod_C14,
)

from fast_component import SolarFastComponent

ref_coeffs = pt.as_tensor(_ref_coeffs)
base_tensor = pt.as_tensor(base.transpose(1, 0, 2))
fac = 0.63712**3


with pm.Model() as mcModel:
    # -------------------------------------------------------------------------
    # Archeomag
    # nus = pm.Gamma(
    #     'nu',
    #     alpha=alpha_nu,
    #     beta=beta_nu,
    #     size=3,
    # )

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

    gs_at_knots = pm.Deterministic(
        'gs_at_knots',
        pt.horizontal_stack(
            gs_at,
            ref_coeffs,
        ),
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
        # nu=1 + nus[0],
        nu=4,
        mu=rD,
        sigma=1.,
        observed=np.zeros(len(idx_D)),
    )

    rI_obs = pm.StudentT(
        'i_obs',
        # nu=1 + nus[1],
        nu=4,
        mu=rI,
        sigma=1.,
        observed=np.zeros(len(idx_I)),
    )

    rF_obs = pm.StudentT(
        'f_obs',
        # nu=1 + nus[2],
        nu=4,
        mu=rF,
        sigma=1.,
        observed=np.zeros(len(idx_F)),
    )

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
    kappa = pm.Beta(
        'kappa',
        alpha=1.,
        beta=7.,
    )
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
    tL = 0.61 * 200
    tR = 831.155

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
    sm_bimod_at_knots = pm.Deterministic(
        'sm_bimod_at_knots',
        pm.math.concatenate(
            (
                sm_bimod,
                ref_solar_df['Phi avg.'].values,
            ),
        ),
    )

    # add fast component
    solar_fast = SolarFastComponent(
        knots_solar_fine,
        tau_solar_fast,
        ref_solar_knots=ref_solar_df['t'].values,
        ref_solar=ref_solar_df['Phi fast'].values,
    )
    # sm_at_both = pm.math.concatenate(
    #     (
    #         sm_bimod[:idx],
    #         sm_bimod[idx:] + solar_fast.get_sm_at_fast(),
    #     )
    # )

    # penalize smaller than zero
    # zero_bound = pm.math.sum(
    #     pm.math.log(
    #         pm.math.sigmoid(
    #             1e-1 * (sm_bimod - 150)
    #         )
    #     )
    # )
    # pm.Potential('zero_bound', zero_bound)
    idx = len(knots_solar) - len(knots_solar_fine)
    sm_at_knots = pm.Deterministic(
        'sm_at_knots',
        pm.math.concatenate(
            (
                sm_bimod_at_knots[:idx],
                sm_bimod_at_knots[idx:] + solar_fast.get_sm_at_fast(),
            )
        )
    )
    sm_rad = interp1d(
        radData['t'],
        knots_solar,
        sm_at_knots,
    )
    # exclude fast component from 10Be data, i.e. use bimod only
    sm_rad_10Be = interp1d(
        radData['t'],
        knots_solar,
        sm_bimod_at_knots,
    )
    gs_rad = interp(
        radData['t'],
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
    q_GL = prod_Be10(dm, sm_rad_10Be)
    hpa = calc_HPA(gs_rad[:, 3], sm_rad_10Be)

    q_GL_cal = prod_Be10(fac*abs(gamma_0), mean_solar)

    q_C14 = prod_C14(dm, sm_rad)
    q_C14_cal = prod_C14(fac*abs(gamma_0), mean_solar)

    s_fac = pm.Normal(
        's_fac',
        sigma=1.,
        size=3,
    )
    # XXX Parametrize the 0.1
    cal_nh = q_GL_cal * (1 + s_fac[0]*0.05)
    cal_sh = q_GL_cal * (1 + s_fac[1]*0.05)
    cal_c14 = q_C14_cal * (1 + s_fac[2]*0.05)

    # XXX Estimate error levels?
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


if __name__ == '__main__':
    from common import mcmc_params
    from pymc.sampling import jax as pmj
    from numpyro.infer.initialization import init_to_median

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
                'init_strategy': init_to_median,
            },
        )

    # idata.to_netcdf('../out/radio_result.nc')
    suffix = f'fast_{tau_solar_fast:d}_'

    idata.to_netcdf(f'../out/radio_{suffix}result.nc')
