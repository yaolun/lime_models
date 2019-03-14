import numpy as np
import h5py
import astropy.constants as const
from scipy.interpolate import interp1d
from astropy.io import ascii
from LIMEanalyses import *
import shutil
import sys, os
from pprint import pprint


pc = const.pc.cgs.value
au = const.au.cgs.value
c = const.c.cgs.value
mh = const.m_p.cgs.value+const.m_e.cgs.value

def write_hdf5(data, filename='infall.h5'):
    (lime_out, auxdata) = data
    n_cells = np.int32(len(lime_out['Tk']))
    T       = lime_out['Tk'] # Gas  Temperature               (K) [n_cells]
    # Generalize later?: T_dust = ? # Dust Temperature (K) [n_cells]
    r       = np.vstack((lime_out['x'], lime_out['y'], lime_out['z'])).T # Voronoi cell center positions  (cm) [n_cells, 3]
    v       = np.vstack((lime_out['vx'], lime_out['vy'], lime_out['vz'])).T # Voronoi cell center velocities (cm/s) [n_cells, 3]
    k_gas   = lime_out['av_gas'] # Gas absorption coefficient     (1/cm) [n_cells]
    j_gas   = lime_out['jv_gas'] # Gas emissivity (erg/s/cm^3/Hz) [n_cells]
    rho_dust = lime_out['density']*mmw*mh/g2d # Dust density (g/cm^3 of dust) [n_cells]
    # Generalize later?: kappa_dust = ? # Dust opacity (cm^2/g of dust) [n_table]

    # Parameters (will go in another file and can be changed later)
    kappa_dust = auxdata['kappa_v'] # Dust opacity at line center (cm^2/g of dust)
    r_max = auxdata['r_max']*au

    print(filename)

    with h5py.File(filename, 'w') as f:
        f.attrs['transition'] = auxdata['transition']
        f.attrs['EA'] = auxdata['EA'] # Einstein-A of the line (s^-1)
        f.attrs['nu0'] = auxdata['nu0'] # Line frequency (Hz)
        f.attrs['trans_up'] = auxdata['trans_up'] # Upper transition level
        f.attrs['degeneracy'] = auxdata['degeneracy'] # Degeneracy of the upper and lower levels (e.g. 11,9)
        f.attrs['n_cells'] = n_cells
        f.attrs['r_max'] = r_max
        f.attrs['kappa_dust'] = kappa_dust  # Dust opacity at line center (cm^2/g of dust)
        # if you need to ensure np.float64
        # f.create_dataset('T', data=np.array(T, dtype=np.float64)) # Temperature (K)
        f.create_dataset('T', data=T) # Temperature (K)
        f['T'].attrs['units'] = b'K'
        f.create_dataset('r', data=r) # Positions (cm)
        f['r'].attrs['units'] = b'cm'
        f.create_dataset('v', data=v) # Velocities (cm/s)
        f['v'].attrs['units'] = b'cm/s'
        f.create_dataset('k_gas', data=k_gas) # Gas absorption coefficient (1/cm/s)
        f['k_gas'].attrs['units'] = b'1/cm/s'
        f.create_dataset('j_gas', data=j_gas) # Gas emissivity (erg/s/cm^3/Hz/sr)
        f['j_gas'].attrs['units'] = b'erg/s/cm^3/Hz/sr'
        f.create_dataset('rho_dust', data=rho_dust) # Dust density (g/cm^3 of dust)
        f['rho_dust'].attrs['units'] = b'g/cm^3'

import argparse
import os
import numpy as np
parser = argparse.ArgumentParser(description='Options for converting LIME output for COLT')
# parser.add_argument('--pathfile', required=True, help='[required] path file for getting paths from run_pylime_path.txt')
parser.add_argument('--model_num', help='model number for converting from LIME to COLT (accept multiple entries separated by comma)')
parser.add_argument('--model_range', help='a range of model number to run')
parser.add_argument('--pathfile', default='run_path_write_hdf5.txt', help='the pathfile for "mod_dir", "limeaid_dir", "rtout", "velfile", and "dustpath"')
parser.add_argument('--subpath', help='any sub-directory following the default path')
parser.add_argument('--mod_dir', help='the model directory', type=str)
parser.add_argument('--limeaid_dir', help='the path of lime-aid', type=str)
parser.add_argument('--rtout', help='user-defined path to the rtout to overwrite the path in lime_config.txt')
parser.add_argument('--velfile', help='user-defined path to the TSC velocity file to overwrite the path in lime_config.txt')
parser.add_argument('--transition', help='the transition to model (default: hco+4-3)', default='hco+4-3')
parser.add_argument('--dustpath', help='the path to the dust model (default path for laptop)')
args = vars(parser.parse_args())

# read in the paths
if args['pathfile'] != None:
    path = np.genfromtxt(args['pathfile'], dtype=str).T
    dict_path = {}
    for name, val in zip(path[0],path[1]):
        dict_path[name] = val
else:
    dict_path = {}

# update the paths with the command line option
for k in args.keys():
    if (k in ['mod_dir', 'limeaid_dir', 'rtout', 'velfile', 'dustpath']) and args[k] != None:
        dict_path[k] = args[k]
# check if all paths are specified
path_flag = 1
for p in ['mod_dir', 'limeaid_dir', 'rtout', 'velfile', 'dustpath']:
    if p not in dict_path.keys():
        print(p+' not specified')
        path_flag = path_flag * 0
if path_flag == 0:
    print('Insufficient paths.  Exit...')
    sys.exit()

print('\n')
print('====================')
print('Please double check!')
print('====================')
print('The following conversion(s) to LIME-AID will use the Hyperion output from {:<s}'.format(dict_path['rtout']))
print('\n')
# dustpath = '/Volumes/SD-Mac/Google Drive/research/lime_models/dust_oh5_interpolated.txt'

# Line parameters
# load from pre-defined file "transitions.txt"
trans = ascii.read('transitions.txt')
if args['transition'] in trans['transition']:
    auxdata = {'transition': args['transition'],
               'EA': trans['EA'][trans['transition'] == args['transition']].data[0],
               'nu0': trans['nu0'][trans['transition'] == args['transition']].data[0],
               'trans_up': trans['trans_up'][trans['transition'] == args['transition']].data[0],
               'degeneracy': list(map(float, str(trans['degeneracy'][trans['transition'] == args['transition']].data[0]).split(',')))}
else:
    print('Transition data not found.  Abort....')
    sys.exit()

print('Selected transition for calculation: {:<s}'.format(args['transition']))
pprint(auxdata)

# if model_range option is used instead
if args['model_range'] != None:
    mod_start = int(args['model_range'].split(',')[0])
    mod_end = int(args['model_range'].split(',')[1])+1
    args['model_num'] = ','.join(np.arange(mod_start, mod_end).astype('str'))

for m in args['model_num'].split(','):
    print('Converting model '+m)
    # LIME model parameters
    if args['subpath'] == None:
        mod_dir = dict_path['mod_dir']+'model'+m+'/'
        args['subpath'] = ''
    else:
        mod_dir = dict_path['mod_dir']+args['subpath']+'/model'+m+'/'

    # read the lime_config.txt
    config = np.genfromtxt(mod_dir+'lime_config.txt', dtype=str).T
    dict_config = {}
    for name, val in zip(config[0],config[1]):
        dict_config[name] = val

    outfilename = 'infall_model'+m
    if args['rtout'] != None:
        rtout = args['rtout']
    elif dict_path['rtout'] == 'none':
        # this is the default option
        rtout = dict_config['rtout']
    else:
        rtout = dict_path['rtout']

    if args['velfile'] != None:
        velfile = args['velfile']
    elif dict_path['velfile'] == 'none':
        # this is the default option
        velfile = mod_dir+'tsc_regrid.h5'
        # velfile = dict_config['velfile']
    else:
        if os.path.exist(dict_config['velfile']):
            velfile = dict_config['velfile']
        elif os.path.exist(dict_path['velfile']):
            velfile = dict_path['velfile']
        else:
            print('velfile not found.  It may cause error if a re-calculation is required.')

    if dict_path['dustpath'] == 'lime':
        dustpath = dict_config['dustfile']
    else:
        dustpath = dict_path['dustpath']

    # Dust parameters
    g2d = float(dict_config['g2d'])
    mmw = float(dict_config['mmw'])
    # dust_lime = ascii.read('/Volumes/SD-Mac/Google Drive/research/lime_models/dust_oh5_interpolated.txt', names=['wave', 'kappa_dust'])
    dust_lime = ascii.read(dustpath, names=['wave', 'kappa_dust'])
    f_dust = interp1d(dust_lime['wave'], dust_lime['kappa_dust'])
    kappa_v_dust = f_dust(c/auxdata['nu0']*1e4)
    auxdata['kappa_v'] = float(kappa_v_dust)
    auxdata['g2d'] = g2d
    auxdata['mmw'] = mmw

    # TODO:  add r_max

    grid = mod_dir+'grid5'
    pop = mod_dir+'populations.pop'
    config = mod_dir+'lime_config.txt'
    recalVelo = False

    dict_params = {'limeaid_dir': dict_path['limeaid_dir'],
                   'mod_dir': mod_dir,
                   'rtout': rtout,
                   'velfile': velfile,
                   'dustpath': dustpath,
                   'recalVelo': recalVelo}

    print('=-=-=-=-=-=-=-=-=-=-')
    print('Settings and Options')
    print('=-=-=-=-=-=-=-=-=-=-')
    for i, k in enumerate(dict_params.keys()):
        print('{:<14s}:  {:<s}'.format(k, str(dict_params[k])))
    print('=-=-=-=-=-=-=-=')
    print('Line parameters')
    print('=-=-=-=-=-=-=-=')
    for i, k in enumerate(auxdata.keys()):
        if type(auxdata[k]) == list():
            aux_str = ', '.join(map(str, auxdata[k]))
        else:
            aux_str = str(auxdata[k])
        print('{:<14s}:  {:<s}'.format(k, aux_str))
    # pprint(dict_params)
    # pprint(auxdata)

    lime_out, auxdata = LIMEanalyses(config=config).LIME2COLT(grid, 5, pop, auxdata,
                                     velfile=velfile, rtout=rtout, recalVelo=recalVelo)

    if not os.path.exists(dict_path['limeaid_dir']+'inits/'+args['subpath']):
        os.makedirs(dict_path['limeaid_dir']+'inits/'+args['subpath'])

    write_hdf5((lime_out, auxdata), filename=dict_path['limeaid_dir']+'inits/'+args['subpath']+'/'+outfilename+auxdata['transition']+'.h5')

    print('write to '+dict_path['limeaid_dir']+'inits/'+args['subpath']+outfilename+auxdata['transition']+'.h5')
