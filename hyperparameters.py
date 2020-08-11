MU = {
        'sea': 1e-2, 'hyperplane_slow': 1e-3, 'hyperplane_fast': 1e-3, 'mixed': 1e-3, 'sine1': 1e-3,
        'circles': 1e-3, 'rcv': 1e-5, 'covtype': 1e-4, 'airline': 1e-3, 'elec': 1e-4, 'powersupply': 1e-3
    } #
STEP_SIZE = {
        'sea': 1e-3, 'hyperplane_slow': 1e-1, 'hyperplane_fast': 1e-2, 'mixed': 0.1, 'sine1': 2e-1,
        'circles':0.1, 'rcv': 5e-1, 'covtype': 5e-3, 'airline': 2e-2, 'elec': 1e-1,    'powersupply': 2e-2
    }
b = {'default': 100, 'elec': 34}
r = {'default': 4}

NOMINAL = {
        'mixed': [0, 1],
        'elec': [1, 2, 3, 4, 5, 6, 7]
    }