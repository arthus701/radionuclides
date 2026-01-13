import numpy as np

import pandas as pd

from common import radData, t_min, t_max

raw_data = pd.read_csv(
    '../dat/10Be_conc_stepf_2000yr_norm_SouthPole.txt',
    sep='\t',
)
annual = [
    'Milcent',
    'Dye-3',
    'NGRIP',
]

annual_Be10_data = raw_data[[r'% age AD', 'NGRIP']].copy()
annual_Be10_data.dropna(how='any', inplace=True)

annual_Be10_data.rename(
    columns={
        r'% age AD': 't',
        'NGRIP': 'Be10'
    },
    inplace=True,
)
t_min_Be10 = annual_Be10_data['t'].min()

bins = radData[t_min_Be10 < radData['t']]['t'].values
binwidth = 22
bins = np.concatenate(
    (
        [bins[0] - binwidth / 2],
        bins + binwidth / 2,
    )
)

integer_indcs = []
for idx, row in annual_Be10_data.iterrows():
    integer = np.floor(row['t'])
    if row['t'] == integer:
        integer_indcs.append(idx)
annual_Be10_data = annual_Be10_data.iloc[integer_indcs]

annual_Be10_data['time_bin'] = pd.cut(annual_Be10_data['t'], bins)
annual_Be10_data['Be10_detrended'] = annual_Be10_data.groupby(
    'time_bin', observed=True,
)['Be10'].transform(lambda x: x - x.mean())


annual_Be10_data['dBe10'] = 0.1

annual_Be10_data = annual_Be10_data.query(
    f'{t_min} <= t and t <= {t_max}'
)
annual_Be10_data.dropna(how='any', inplace=True)
annual_Be10_data.sort_values(by='t', inplace=True)
annual_Be10_data.reset_index(inplace=True, drop=True)

if __name__ == '__main__':
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(
        1, 1,
        figsize=(8, 4),
    )

    ax.scatter(
        annual_Be10_data['t'],
        annual_Be10_data['Be10_detrended'],
        marker='.',
    )

    plt.show()
