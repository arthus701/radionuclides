import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

from common import annual_C14_data, radData, solar_constr

PERIOD = 10.4   # years

t_array, dt = np.linspace(1800, 2000, 501, retstep=True)

inv_period = 1 / PERIOD
cycles_per_year = np.cumsum(
    inv_period * dt * np.ones_like(t_array)
)


def signal(phase):
    return np.sin(2 * np.pi * (phase / 360 + cycles_per_year))


fig, ax = plt.subplots(
    1, 1,
    figsize=(13, 6),
)

bnds = ax.get_position().bounds
slider_width = bnds[2]
sliderloc = ax.get_position().bounds
sax = fig.add_axes([sliderloc[0], sliderloc[1]-0.1, slider_width, 0.02])

phase_slider = Slider(
    sax,
    'Phase',
    -180,
    180,
    valinit=0,
    valstep=0.1,
    valfmt='%.1f',
)

line, = ax.plot(
    t_array,
    signal(0),
)
ax.set_xlim(t_array.min(), t_array.max())
ax.set_ylim(-1.1, 1.1)

t_min = annual_C14_data['t'].min()

bins = radData[t_min < radData['t']]['t'].values
binwidth = 22
bins = np.concatenate(
    (
        [bins[0] - binwidth / 2],
        bins + binwidth / 2,
    )
)

annual_C14_data = annual_C14_data[1800 <= annual_C14_data['t']]

annual_data_normalized = annual_C14_data['C14_detrended'].values
annual_data_normalized -= annual_data_normalized.mean()
annual_data_normalized /= np.abs(annual_data_normalized).max()

ax.plot(
    annual_C14_data['t'].values,
    -annual_data_normalized,
    color='grey',
    marker='.',
    zorder=-1,
)

solar_constr.dropna(how='any', inplace=True)
solar_constr['time_bin'] = pd.cut(solar_constr['Year'], bins)
solar_constr['Phi_detrended'] = solar_constr.groupby(
    'time_bin', observed=True,
)['Phi (MV)'].transform(lambda x: x - x.mean())

solar_normalized = solar_constr['Phi_detrended'].values
solar_normalized -= solar_normalized.mean()
solar_normalized /= np.abs(solar_normalized).max()

ax.plot(
    solar_constr['Year'],
    solar_normalized,
    color='black',
    marker='.',
    zorder=-1,
)


def update(val):
    line.set_ydata(
        signal(phase_slider.val)
    )
    fig.canvas.draw_idle()


phase_slider.on_changed(update)


def on_key(event):
    if event.key == 'right':
        new_val = min(
            phase_slider.val + phase_slider.valstep,
            phase_slider.valmax,
        )
        phase_slider.set_val(new_val)
    elif event.key == 'left':
        new_val = max(
            phase_slider.val - phase_slider.valstep,
            phase_slider.valmin,
        )
        phase_slider.set_val(new_val)


fig.canvas.mpl_connect('key_press_event', on_key)
plt.show()
