def grid_create(list_params):

    import numpy as np
    import itertools as iter
    import copy
    from pprint import pprint
    import astropy.constants as const

    # cartiesian product of lists
    product = [x for x in iter.product(*list_params.values())]

    # write the model parameters into a separate model list
    foo = open('/Users/yaolun/GoogleDrive/research/lime_models/drop_grid.txt', 'w')
    colhead = ('model_name', 'hy_model', 'cs', 'tsc', 'a_model', 'a_params0', 'a_params1', 'a_params2', 'a_params3', 'a_params4')
    foo.write('{:<14s}  {:<14s}  {:<14s}  {:<14s}  {:<14s}  {:<14s}  {:<14s}  {:<14s}  {:<14s}  {:<14s}\n'.format(*colhead))

    ref = {'model_name': 1, 'hy_model': 'model14', 'cs': 0.37, 'tsc': 'bhr71shallow',
           'a_model': 'drop3', 'Xo': 5e-9, 'Xd': 1e-11,'Tevap': 100.0, 'ndepl': 1e6, 'a_params4': -1}

    for i, mod in enumerate(product):
        params_dum = copy.copy(ref)
        for j, col in enumerate(list_params.keys()):
            params_dum[col] = mod[j]

        output = (params_dum['model_name']+i, params_dum['hy_model'],params_dum['cs'],params_dum['tsc'],
                  params_dum['a_model'],params_dum['Xo'],params_dum['Xd'],params_dum['Tevap'],
                  params_dum['ndepl'],params_dum['a_params4'])
        foo.write('{:<14d}  {:<14s}  {:<14.3f}  {:<14s}  {:<14s}  {:<14e}  {:<14e}  {:<14f}  {:<14e}  {:<14d}\n'.format(*output))
    foo.close()

    # return list_params.keys

import numpy as np

# grid of age and view_angle
list_params = {'Xo': 10**np.arange(-10, -7.5, 1.0),
               'Xd': 10**np.arange(-13, -10, 1.0),
               'Tevap': np.arange(100,101),
               'ndepl': 10**np.arange(5, 7.5, 1.0)}

grid_create(list_params)
