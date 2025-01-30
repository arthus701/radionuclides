import numpy as np
import pymc as pm

from utils import matern_kernel


class SolarLongtermComponent():
    def __init__(self, knots_solar, tau_solar_longterm, n_ref_solar=0):
        self.knots = knots_solar
        self.tau = tau_solar_longterm
        self.n_ref = n_ref_solar
        if self.tau is not None:
            cov_solar_longterm = matern_kernel(
                self.knots,
                sigma=1,
                tau=tau_solar_longterm,
            )
            self.chol_solar_longterm = np.linalg.cholesky(
                cov_solar_longterm+1e-6*np.eye(len(self.knots))
            )[
                :-self.n_ref,
                :-self.n_ref,
            ]

    def get_sm_at_longterm(self):
        if self.tau is None:
            return 0

        sm_cent_longterm = pm.Normal(
            'sm_cent_longterm',
            mu=0,
            sigma=1,
            size=(len(self.knots)-self.n_ref,),
        )
        sm_longterm_scale = pm.Gamma(
            'sm_longterm_scale',
            alpha=3,
            beta=3/200,
            size=1,
        )
        return sm_longterm_scale * self.chol_solar_longterm @ sm_cent_longterm
