
import numpy as np
from pymagglobal.utils import REARTH

# -----------------------------------------------------------------------------
prefix = 'radio_'
fix_calibration = False
use_longterm = False
use_11year_cycle = False

# -----------------------------------------------------------------------------
# Range and resolution
t_min = -7000
t_solar_fine = -1100
t_max = 2000
step = 50
step_solar_coarse = 22
step_solar_fine = 2

# -----------------------------------------------------------------------------
# Magnetic field parameters
lmax = 5
dip = 1
R = REARTH
gamma_0 = -32.5
alpha_dip = 10
omega = 1/741
chi = 1/138
xi = np.sqrt(chi**2 - omega**2)
alpha_list = [
    3.5,
    1.765,
    1.011,
    0.455,
    0.177,
]
tau_list = [
    200,
    133,
    174,
    138,
    95,
]


# -----------------------------------------------------------------------------
# Solar variation parameters
mu_solar = 650
sigma_solar = 203
tau_solar = 27.2
tau_solar_fast = 4
tau_fast_period = 10.4

# -----------------------------------------------------------------------------
# Other parameters
# set to zero when using Andreas' dataset
tDir = 0                        # Directional truncation error in Deg.
tInt = 0                        # Intensity truncation error in uT
# for the error distribution, currently not in use
alpha_nu = 2.
beta_nu = 0.1

# -----------------------------------------------------------------------------
# MCMC parameters
mcmc_params = {
    'n_samps': 500,
    'n_warmup': 1000,
    'n_chains': 4,
    'target_accept': 0.95,
}
