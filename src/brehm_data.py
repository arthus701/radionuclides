import numpy as np

import pandas as pd

from common import (
    t_min,
    t_max,
    radData,
)

brehm_21_data = pd.read_excel(
    '../dat/41561_2020_674_MOESM2_ESM.xlsx',
    sheet_name='Figure1b',
)
brehm_21_data.dropna(how='all', inplace=True)

integer_indcs = []
for idx, row in brehm_21_data.iterrows():
    integer = np.floor(row['ETH Simulated time (yr AD)'])
    if row['ETH Simulated time (yr AD)'] == integer:
        integer_indcs.append(idx)
brehm_21_data = brehm_21_data.iloc[integer_indcs]

brehm_21_t = brehm_21_data['ETH Simulated time (yr AD)'].values
brehm_21_C14 = brehm_21_data['ETH normalized production'].values

t_min_C14 = brehm_21_t.min()
bins = radData[t_min_C14 < radData['t']]['t'].values
binwidth = 22
bins = np.concatenate(
    (
        [bins[0] - binwidth / 2],
        bins + binwidth / 2,
    )
)
brehm_21_data['time_bin'] = pd.cut(
    brehm_21_data['ETH Simulated time (yr AD)'],
    bins,
)
brehm_21_detrended = brehm_21_data.groupby(
    'time_bin', observed=True,
)['ETH normalized production'].transform(lambda x: x - x.mean())

brehm_data = pd.DataFrame(
    data={
        't': brehm_21_t,
        'C14': brehm_21_C14,
        'dC14': 0.05 * np.ones_like(brehm_21_t),
        'C14_detrended': brehm_21_detrended,
    }
)
brehm_data.sort_values(by='t', inplace=True)
brehm_data = brehm_data.query(
    f'{t_min} <= t and t <= {t_max}'
)
brehm_data.dropna(how='any', inplace=True)
brehm_data.reset_index(inplace=True, drop=True)

if __name__ == '__main__':
    from matplotlib import pyplot as plt

    fig, axs = plt.subplots(
        2, 1,
        figsize=(10, 4),
        sharex=True,
    )

    axs[0].errorbar(
        brehm_data['t'],
        brehm_data['C14'],
        yerr=brehm_data['dC14'],
        marker='.',
        color='grey',
        ls='',
    )
    axs[1].errorbar(
        brehm_data['t'],
        brehm_data['C14_detrended'],
        yerr=brehm_data['dC14'],
        marker='.',
        color='grey',
        ls='',
    )

    axs[0].set_title('$^{14}$C production rates from Brehm et al.')

    axs[1].set_xlabel('time [yrs.]')

    axs[0].set_ylabel('Raw')
    axs[1].set_ylabel('Detrended')

    fig.tight_layout()

    plt.show()
