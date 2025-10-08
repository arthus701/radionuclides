import numpy as np
import pymc as pm

from utils import sqe_kernel


class SolarPeriodicComponent():
    def __init__(
        self,
        knots_solar,
        ref_solar_knots=None,
        ref_solar=None,
        jitter=1e-4,
    ):
        self.knots = knots_solar
        self.jitter = jitter
        self.prior_mean = np.zeros_like(self.knots)

        self.mu_scale = 175
        self.sigma_scale = 50
        self.tau_scale = 20

        cov_scale = sqe_kernel(
            self.knots,
            tau=self.tau_scale,
            sigma=self.sigma_scale,
        )
        self.chol_scale = np.linalg.cholesky(
            cov_scale + self.jitter * np.eye(len(self.knots))
        )

        self.mu_period = 10.4
        self.sigma_period = 1
        self.tau_period = 100

        cov_period = sqe_kernel(
            self.knots,
            tau=self.tau_period,
            sigma=self.sigma_period,
        )
        self.chol_period = np.linalg.cholesky(
            cov_period + self.jitter * np.eye(len(self.knots))
        )

    def get_sm_at_fast(self):
        # phase = pm.Normal(
        #     'sm_fast_phase',
        #     mu=0,
        #     sigma=1,
        #     size=1,
        # )
        # eyeball from visually fitting sine with fixed period of 10.4 years
        phase = -50 / 360

        sm_cent_fast_scale = pm.Normal(
            'sm_cent_fast_scale',
            mu=0,
            sigma=1,
            size=(len(self.prior_mean),),
        )
        scale = self.mu_scale + self.chol_scale @ sm_cent_fast_scale

        sm_cent_fast_period = pm.Normal(
            'sm_cent_fast_period',
            mu=0,
            sigma=1,
            size=(len(self.prior_mean),),
        )
        period = self.mu_period + self.chol_period @ sm_cent_fast_period
        inv_period = 1 / period
        cycles_per_year = pm.math.cumsum(
            inv_period * (self.knots[2] - self.knots[1])
        )

        sm_fast_at_knots = pm.Deterministic(
            'sm_fast_at_knots',
            self.prior_mean
            + scale * pm.math.sin(2 * np.pi * (phase + cycles_per_year))
        )

        return sm_fast_at_knots
