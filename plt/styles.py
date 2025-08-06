import matplotlib

figwidth = 455.24408 / 72
figheight = 256.0748 / 72

radio = {
    'name': 'radio',
    'prefix': 'radio_pfm_afm9k',
    'color': 'C0',
}

radio_bimodal = {
    'name': 'radio_flexible',
    'prefix': 'radio_bimodal',
    'color': 'C0',
}

radio_fixed = {
    'name': 'radio_fixed',
    'prefix': 'radio_bimodal_fixed',
    'color': 'C6',
}

radio_longterm_500 = {
    'name': 'radio_lt_500',
    'prefix': 'radio_bimodal_longterm_500',
    'color': 'C1',
}

radio_longterm_1000 = {
    'name': 'radio_lt_1000',
    'prefix': 'radio_bimodal_longterm_1000',
    'color': 'C3',
}

radio_precalib = {
    'name': 'radio_precalib',
    'prefix': 'radio_bimodal_calib_fix',
    'color': 'C3',
}

radio_afm9k = {
    'name': 'radio rerun',
    'prefix': 'radio_afm9k',
    'color': 'C9',
}

radio_fdp2d = {
    'name': 'upd prior',
    'prefix': 'radio_fdp2d',
    'color': 'C1',
}

akm = {
    'name': 'ArchKalmag14k.r',
    'prefix': 'ArchKalmag14k.r',
    'color': 'C4',
}

afm9k = {
    'name': 'afm9k rerun',
    'prefix': 'arch_afm9k',
    'color': 'C7',
}

pfm9k2 = {
    'name': 'pfm9k.2',
    'prefix': 'pfm9k2',
    'color': 'C5',
}

arch = {
    'name': 'archeo',
    'prefix': 'arch_pfm_afm9k',
    'color': 'C2',
}

arch_fdp2d = {
    'name': 'archeo (upd)',
    'prefix': 'arch_fdp2d',
    'color': 'C3',
}

models = [
    # akm,
    # pfm9k2,
    arch,
    # afm9k,
    # arch_fdp2d,
    # radio,
    radio_bimodal,
    # radio_fixed,
    radio_longterm_1000,
    # radio_fdp2d,
    # radio_afm9k,
]

beamercolor = (70/255, 60/255, 70/255, 1.)

tlabel = 'time [yrs.]'


def setup():
    # matplotlib.use("pgf")
    matplotlib.rcParams.update({
        # 'pgf.texsystem': 'xelatex',
        # 'font.family': 'sans-serif',
        # 'font.sans-serif': 'Fira Sans',
        'font.size': 10,
        # 'text.usetex': True,
        # 'pgf.rcfonts': False,
        # 'text.color': beamercolor,
        # 'axes.labelcolor': beamercolor,
        # 'xtick.color': beamercolor,
        # 'ytick.color': beamercolor,
        # 'lines.linewidth': 0.8,
        'lines.markersize': 3,
    })
