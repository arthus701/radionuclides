import numpy as np
from numpy import ndarray

import pandas as pd

import jax.numpy as jnp
from pytensor import tensor as pt, shared

from pytensor.tensor.math import Argmax
from pytensor.link.jax.dispatch import jax_funcify


@jax_funcify.register(Argmax)
def jax_funcify_Argmax(op, **kwargs):
    # Obtain necessary "static" attributes from the Op being converted
    axis = op.axis

    # Create a JAX jit-able function that implements the Op
    def argmax(x, axis=axis):
        if axis is not None:
            if 1 < len(axis):
                raise ValueError("JAX doesn't support tuple axis.")
            axis = axis[0]

        return jnp.argmax(x, axis=axis).astype("int64")

    return argmax


def interp(_x, _xp, _fp):
    """
    Simple equivalent of np.interp to compute a linear interpolation. This
    function will not extrapolate, but use the edge values for points outside
    of _xp.
    """
    if isinstance(_xp, ndarray) or isinstance(_xp, list):
        xp = shared(_xp)
    else:
        xp = _xp

    if isinstance(_fp, ndarray) or isinstance(_fp, list):
        fp = shared(_fp)
    else:
        fp = _fp

    # No extrapolation!
    x = pt.clip(_x, xp[0], xp[-1])
    # First we find the nearest neighbour
    ind = pt.argmin((x[:, None] - xp[None, :])**2, axis=1)
    xi = xp[ind]
    # Figure out if we are on the right or the left of nearest
    s = pt.sign(x - xi).astype(int)
    # Perform linear interpolation
    # (1-pt.abs(s))*1e-10) to prevent divide by zero error if we are exactly
    # at the knot points
    a = (fp[ind + s] - fp[ind]) \
        / (xp[ind + s] - xp[ind] + (1-pt.abs(s))*1e-10)[:, None]

    b = fp[ind] - a * xp[ind, None]

    return a * x[:, None] + b


def interp1d(_x, _xp, _fp):
    """
    Simple equivalent of np.interp to compute a linear interpolation. This
    function will not extrapolate, but use the edge values for points outside
    of _xp.
    """
    if isinstance(_xp, ndarray) or isinstance(_xp, list):
        xp = shared(_xp)
    else:
        xp = _xp

    if isinstance(_fp, ndarray) or isinstance(_fp, list):
        fp = shared(_fp)
    else:
        fp = _fp

    # No extrapolation!
    x = pt.clip(_x, xp[0], xp[-1])
    # First we find the nearest neighbour
    ind = pt.argmin((x[:, None] - xp[None, :])**2, axis=1)
    xi = xp[ind]
    # Figure out if we are on the right or the left of nearest
    s = pt.sign(x - xi).astype(int)
    # Perform linear interpolation
    # (1-pt.abs(s))*1e-10) to prevent divide by zero error if we are exactly
    # at the knot points
    a = (fp[ind + s] - fp[ind]) / (
        xp[ind + s] - xp[ind] + (1-pt.abs(s))*1e-10
    )

    b = fp[ind] - a * xp[ind]

    return a * x + b


def matern_kernel(x, y=None, tau=2, sigma=1.):
    if y is None:
        y = x
    x = np.asarray(x)
    y = np.asarray(y)
    frac = np.abs((x[:, None] - y[None, :])) / tau
    res = sigma**2 * (1. + frac) * np.exp(-frac)
    return res.reshape(x.shape[0], y.shape[0])


def quasiperiodic_kernel(x, y=None, tau=2, sigma=1., p=11., gamma=1.):
    # https://arxiv.org/pdf/2207.12164
    if y is None:
        y = x
    x = np.asarray(x)
    y = np.asarray(y)
    frac_p = np.pi * np.abs((x[:, None] - y[None, :])) / p
    frac_t = np.abs((x[:, None] - y[None, :])) / tau
    res = sigma**2 * np.exp(-gamma * np.sin(frac_p)**2 - frac_t)
    return res.reshape(x.shape[0], y.shape[0])


def sqe_kernel(x, y=None, tau=2, sigma=1.):
    if y is None:
        y = x
    x = np.asarray(x)
    y = np.asarray(y)
    frac = np.abs((x[:, None] - y[None, :])) / tau
    res = sigma**2 * np.exp(-0.5*frac**2)
    return res.reshape(x.shape[0], y.shape[0])


def cosine_kernel(x, y=None, sigma=1., p=11.):
    if y is None:
        y = x
    x = np.asarray(x)
    y = np.asarray(y)
    frac_p = np.pi * np.abs((x[:, None] - y[None, :])) / p
    res = sigma**2 * np.cos(frac_p)**2
    return res.reshape(x.shape[0], y.shape[0])


def periodic_kernel(x, y=None, tau=2, sigma=1., p=11.):
    # https://arxiv.org/pdf/2207.12164
    if y is None:
        y = x
    x = np.asarray(x)
    y = np.asarray(y)
    frac_p = np.pi * np.abs((x[:, None] - y[None, :])) / p
    res = sigma**2 * np.exp(-2 / tau**2 * np.sin(frac_p)**2)
    return res.reshape(x.shape[0], y.shape[0])


def bin_average(df, window):
    bins = []
    avg_values = []
    current_bin = df['t'].max()
    while current_bin > df['t'].min():
        mask = (
            (df['t'] >= current_bin - window / 2)
            & (df['t'] <= current_bin + window / 2)
        )
        bins.append(current_bin)
        avg_values.append(df.loc[mask, 'C14'].mean())
        current_bin -= window

    return pd.DataFrame(
        data={
            't': bins,
            'C14': avg_values,
        }
    )


def moving_average(df, window, key='C14'):
    avg_values = []
    for i in range(len(df)):
        x_center = df.loc[i, 't']
        mask = (
            (df['t'] >= x_center - window / 2)
            & (df['t'] <= x_center + window / 2)
        )
        avg_values.append(df.loc[mask, key].mean())

    return np.array(avg_values)
