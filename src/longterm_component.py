import numpy as np
import pymc as pm
from pytensor import tensor as pt

from utils import matern_kernel


class SolarLongtermComponent():
    def __init__(self, knots_solar, tau_solar_longterm, n_ref_solar=0):
        self.knots = knots_solar[:-n_ref_solar]
        self.tau = tau_solar_longterm

        if self.tau is not None:
            cov_solar_longterm = matern_kernel(
                self.knots,
                sigma=1,
                tau=tau_solar_longterm,
            )
            self.chol_solar_longterm = np.linalg.cholesky(
                cov_solar_longterm+1e-6*np.eye(len(self.knots))
            )

    def get_sm_at_longterm(self):
        if self.tau == 0:
            return 0
        if self.tau is None:
            sm_tau_longterm = 100 + pm.Gamma(
                'sm_tau_longterm',
                alpha=7.5,
                beta=1 / 100,
                size=1,
            )
            frac = pt.as_tensor(
                np.abs(
                    self.knots[:, None]
                    - self.knots[None, :]
                )
            ) / sm_tau_longterm
            cov_solar_longterm = (1. + frac) * pm.math.exp(-frac)

            chol_solar_longterm = pt.slinalg.cholesky(
                cov_solar_longterm + 1e-6*pt.as_tensor(np.eye(len(self.knots)))
            )
        else:
            chol_solar_longterm = self.chol_solar_longterm

        sm_cent_longterm = pm.Normal(
            'sm_cent_longterm',
            mu=0,
            sigma=1,
            size=(len(self.knots),),
        )
        sm_longterm_scale = pm.Gamma(
            'sm_longterm_scale',
            alpha=3,
            beta=3/200,
            size=1,
        )
        return sm_longterm_scale * chol_solar_longterm @ sm_cent_longterm
