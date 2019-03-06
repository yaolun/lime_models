def grid_create(list_params):

    import numpy as np
    import itertools as iter
    import copy
    from pprint import pprint
    import astropy.constants as const
    import os

    # cartiesian product of lists
    product = [x for x in iter.product(*list_params.values())]

    # read the exisiting model list
    if os.path.exists('/Users/yaolun/GoogleDrive/research/lime_models/drop_grid.txt'):
        model_list = open('/Users/yaolun/GoogleDrive/research/lime_models/drop_grid.txt','r').readlines()
        for i, line in enumerate(model_list[2:]):
            if not line.startswith('#'):
                model_list[i+2] = '# '+line
        last_model_num = i+1
        foo = open('/Users/yaolun/GoogleDrive/research/lime_models/drop_grid.txt', 'w')
        for line in model_list:
            foo.write(line)
    else:
        last_model_num = 0
        # write the model parameters into a separate model list
        foo = open('/Users/yaolun/GoogleDrive/research/lime_models/drop_grid.txt', 'w')
        colhead = ('model_name', 'moldata', 'lower_level', 'hy_model', 'cs', 'velfile', 'a_model', 'a_params0', 'a_params1', 'a_params2', 'a_params3', 'a_params4')
        foo.write('{:<14s}  {:<14s}  {:<14s}  {:<14s}  {:<14s}  {:<14s}  {:<14s}  {:<14s}  {:<14s}  {:<14s}  {:<14s}  {:<14s}\n'.format(*colhead))

    ref = {'model_name': last_model_num+1, 'moldata': 'hco+@xpol.dat', 'lower_level': '3',
           'hy_model': 'model57', 'cs': 0.37, 'tsc': 'none',
           'a_model': 'drop3', 'Xo': 5e-9, 'Xd': 1e-11,'Tevap': 30.0, 'ndepl': 1e6, 'a_params4': 100}

    for i, mod in enumerate(product):
        params_dum = copy.copy(ref)
        for j, col in enumerate(list_params.keys()):
            params_dum[col] = mod[j]

        output = (params_dum['model_name']+i, params_dum['moldata'], params_dum['lower_level'],
                  params_dum['hy_model'],params_dum['cs'],params_dum['tsc'],
                  params_dum['a_model'],params_dum['Xo'],params_dum['Xd'],params_dum['Tevap'],
                  params_dum['ndepl'],params_dum['a_params4'])
        foo.write('{:<14d}  {:<14s}  {:<14s}  {:<14s}  {:<14.3f}  {:<14s}  {:<14s}  {:<14e}  {:<14e}  {:<14f}  {:<14e}  {:<14d}\n'.format(*output))
    foo.close()

    # return list_params.keys

import numpy as np

# grid of age and view_angle
# 1st run
# list_params = {'Xo': 10**np.arange(-10, -7.5, 1.0),
#                'Xd': 10**np.arange(-13, -10, 1.0),
#                'Tevap': np.arange(100,101),
#                'ndepl': 10**np.arange(5, 7.5, 1.0)}

# 2nd run
list_params = {'Xo': 10**np.arange(-9.5, -7.0, 1.0),
               'Xd': 10**np.arange(-12.5, -9.5, 1.0),
               'Tevap': np.arange(30,31),
               'ndepl': 10**np.arange(5.5, 8.0, 1.0)}

grid_create(list_params)
