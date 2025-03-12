import numpy as np
import pymc as pm
from pytensor import tensor as pt

from utils import matern_kernel


class SolarFastComponent():
    def __init__(self, knots_solar, tau_solar, n_ref_solar=0, jitter=1e-4):
        self.knots = knots_solar[:-n_ref_solar]
        self.tau = tau_solar
        self.jitter = jitter

        if self.tau is not None:
            cov_solar_fast = matern_kernel(
                self.knots,
                sigma=1,
                tau=tau_solar,
            )
            self.chol_solar_fast = np.linalg.cholesky(
                cov_solar_fast + self.jitter * np.eye(len(self.knots))
            )

    def get_sm_at_fast(self):
        if self.tau == 0:
            return 0
        if self.tau is None:
            sm_tau_fast = 1 + pm.Gamma(
                'sm_tau_fast',
                alpha=2.5,
                beta=1,
                size=1,
            )
            frac = pt.as_tensor(
                np.abs(
                    self.knots[:, None]
                    - self.knots[None, :]
                )
            ) / sm_tau_fast
            cov_solar_fast = (1. + frac) * pm.math.exp(-frac)

            chol_solar_fast = pt.slinalg.cholesky(
                cov_solar_fast
                + self.jitter * pt.as_tensor(np.eye(len(self.knots)))
            )
        else:
            chol_solar_fast = self.chol_solar_fast

        sm_cent_fast = pm.Normal(
            'sm_cent_fast',
            mu=0,
            sigma=1,
            size=(len(self.knots),),
        )
        sm_fast_scale = pm.Gamma(
            'sm_fast_scale',
            alpha=3,
            beta=3/200,
            size=1,
        )
        damping = pm.math.sigmoid(
            0.1 * (self.knots + 100)
        )
        sm_fast = pm.Deterministic(
            'sm_fast',
            damping * sm_fast_scale * chol_solar_fast @ sm_cent_fast,
        )
        return sm_fast
