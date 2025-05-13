import numpy as np
import pymc as pm

from utils import cosine_kernel, sqe_kernel


class SolarPeriodicComponent():
    def __init__(
        self,
        knots_solar,
        period_solar,
        tau_solar,
        ref_solar_knots=None,
        ref_solar=None,
        jitter=1e-4,
    ):
        self.knots = knots_solar
        self.period = period_solar
        self.tau = tau_solar
        self.jitter = jitter
        self.ref_solar = ref_solar

        if self.tau < 0:
            raise ValueError(
                f'Timescale has to be larger than 0 and not {self.tau}.'
            )
        if self.tau == 0:
            self.chol_solar = None
        else:
            if ref_solar_knots is not None:
                n_ref_solar = len(ref_solar_knots)

                cov_solar = cosine_kernel(
                    self.knots,
                    sigma=200.,
                    p=self.period,
                ) * sqe_kernel(
                    self.knots,
                    tau=self.tau,
                    sigma=1.,
                )
                cov_obs = cosine_kernel(
                    ref_solar_knots,
                    sigma=200.,
                    p=self.period,
                ) * sqe_kernel(
                    ref_solar_knots,
                    tau=self.tau,
                    sigma=1.,
                )
                cor_obs = cosine_kernel(
                    self.knots,
                    ref_solar_knots,
                    sigma=200.,
                    p=self.period,
                ) * sqe_kernel(
                    self.knots,
                    ref_solar_knots,
                    tau=self.tau,
                    sigma=1.,
                )

                _icov_obs = np.linalg.inv(
                    cov_obs + 1e-4*np.eye(len(ref_solar_knots))
                )
                prior_mean = cor_obs @ _icov_obs @ ref_solar
                cov_solar = cov_solar - cor_obs @ _icov_obs @ cor_obs.T

                chol_solar = np.linalg.cholesky(
                    cov_solar + self.jitter * np.eye(len(self.knots))
                )

                self.prior_mean = prior_mean[:-n_ref_solar] / 200
                self.chol_solar = \
                    chol_solar[:-n_ref_solar, :-n_ref_solar] / 200
            else:
                self.prior_mean = 0

                cov_solar = cosine_kernel(
                    self.knots,
                    sigma=200.,
                    p=self.period,
                ) * sqe_kernel(
                    self.knots,
                    tau=self.tau,
                    sigma=1.,
                )

                self.chol_solar = np.linalg.cholesky(
                    cov_solar + self.jitter * np.eye(len(self.knots))
                )

    def get_sm_at_fast(self):
        if self.tau == 0:
            return 0

        sm_cent_fast = pm.Normal(
            'sm_cent_fast',
            mu=0,
            sigma=1,
            size=(len(self.prior_mean),),
        )
        sm_fast_scale = pm.Gamma(
            'sm_fast_scale',
            alpha=3,
            beta=3/200,
            size=1,
        )
        # damping = pm.math.sigmoid(
        #     0.1 * (self.knots + 100)
        # )
        damping = 1
        sm_fast = damping * sm_fast_scale * (
            self.prior_mean + self.chol_solar @ sm_cent_fast
        )
        sm_fast_at_knots = pm.Deterministic(
            'sm_fast_at_knots',
            pm.math.concatenate(
                (
                    sm_fast,
                    self.ref_solar,
                ),
            )
        )
        return sm_fast_at_knots
