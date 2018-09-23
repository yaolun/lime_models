import numpy as np
import h5py
import astropy.constants as const
from scipy.interpolate import interp1d
from astropy.io import ascii
from LIMEanalyses import *
import shutil

pc = const.pc.cgs.value
au = const.au.cgs.value
c = const.c.cgs.value
mh = const.m_p.cgs.value+const.m_e.cgs.value

def write_hdf5((lime_out, auxdata), filename='infall.h5'):
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
parser.add_argument('--subpath', help='any sub-directory following the default path')
parser.add_argument('--mod_dir', help='the model directory',
                    default='/Volumes/SD-Mac/lime_runs/', type=str)
parser.add_argument('--colt_dir', help='the path of colt-lime', default='/Users/yaolun/programs/colt-lime/', type=str)
parser.add_argument('--rtout', help='user-defined path to the rtout to overwrite the path in lime_config.txt')
parser.add_argument('--velfile', help='user-defined path to the TSC velocity file to overwrite the path in lime_config.txt')
parser.add_argument('--transition', help='the transition to model (default: hco+4-3)', default='hco+4-3')
parser.add_argument('--dustpath', default='/Volumes/SD-Mac/Google Drive/research/lime_models/dust_oh5_interpolated.txt',
                    help='the path to the dust model (default path for laptop)')
args = vars(parser.parse_args())

# dustpath = '/Volumes/SD-Mac/Google Drive/research/lime_models/dust_oh5_interpolated.txt'

# Line parameters
# HCO+ 4-3
if args['transition'] == 'hco+4-3':
    auxdata = {'transition': args['transition'],
               'EA': 3.6269e-03,
               'nu0': 356.7342880e9,
               'trans_up': 4,
               'degeneracy': [9,7]}  # degeneracy from upper to lower
    suffix = ''
if args['transition'] == 'hco+3-2':
    # HCO+ 3-2
    auxdata = {'transition': args['transition'],
               'EA': 1.4757e-03,
               'nu0': 267.5576190e9,
               'trans_up': 3,
               'degeneracy': [7,5]}  # degeneracy from upper to lower
    suffix = '_'+args['transition']

if args['transition'] == 'hcn4-3':
    auxdata = {'transition': args['transition'],
               'EA': 2.0540e-03,
               'nu0': 354.5054779e9,
               'trans_up': 4,
               'degeneracy': [9,7]}
    suffix = '_'+args['transition']

if args['transition'] == 'cs7-6':
    auxdata = {'transition': args['transition'],
               'EA': 8.395e-4,
               'nu0': 342.8828503e9,
               'trans_up': 7,
               'degeneracy': [15,13]}
    suffix = '_'+args['transition']

print('Selected transition for calculation: {:<s}'.format(args['transition']))

    # Methanol 23-22
    # auxdata = {}


# Methanol 23-22
# auxdata = {}

# if model_range option is used instead
if args['model_range'] != None:
    mod_start = int(args['model_range'].split(',')[0])
    mod_end = int(args['model_range'].split(',')[1])+1
    args['model_num'] = ','.join(np.arange(mod_start, mod_end).astype('str'))

for m in args['model_num'].split(','):
    print('Converting model '+m)
    # LIME model parameters
    if args['subpath'] == None:
        mod_dir = args['mod_dir']+'model'+m+'/'
        args['subpath'] = ''
    else:
        mod_dir = args['mod_dir']+args['subpath']+'/model'+m+'/'

    # read the lime_config.txt
    config = np.genfromtxt(mod_dir+'lime_config.txt', dtype=str).T
    dict_config = {}
    for name, val in zip(config[0],config[1]):
        dict_config[name] = val

    outfilename = 'infall_model'+m
    # rtout = '/Volumes/SD-Mac/model14.rtout'
    # velfile = '/Users/yaolun/programs/misc/TSC/rho_v_env'
    if args['rtout'] == None:
        rtout = dict_config['rtout']
    else:
        rtout = args['rtout']
    if args['velfile'] == None:
        velfile = dict_config['velfile']
    else:
        velfile = args['velfile']
    mmw = float(dict_config['mmw'])
    if args['dustpath'] == 'lime':
        dustpath = dict_config['dustfile']
    else:
        dustpath = args['dustpath']

    # Dust parameters
    g2d = float(dict_config['g2d'])
    # dust_lime = ascii.read('/Volumes/SD-Mac/Google Drive/research/lime_models/dust_oh5_interpolated.txt', names=['wave', 'kappa_dust'])
    dust_lime = ascii.read(dustpath, names=['wave', 'kappa_dust'])
    f_dust = interp1d(dust_lime['wave'], dust_lime['kappa_dust'])
    kappa_v_dust = f_dust(c/auxdata['nu0']*1e4)
    auxdata['kappa_v'] = float(kappa_v_dust)

    # TODO:  add r_max

    grid = mod_dir+'grid5'
    pop = mod_dir+'populations.pop'
    config = mod_dir+'lime_config.txt'
    recalVelo = False

    lime_out, auxdata = LIMEanalyses(config=config).LIME2COLT(grid, 5, pop, auxdata,
                                     velfile=velfile, rtout=rtout, recalVelo=recalVelo)

    if not os.path.exists(args['colt_dir']+'inits/'+args['subpath']+'/'):
        os.makedirs(args['colt_dir']+'inits/'+args['subpath']+'/')

    write_hdf5((lime_out, auxdata), filename=args['colt_dir']+'inits/'+args['subpath']+'/'+outfilename+suffix+'.h5')

    # shutil.copyfile(outfilename+'.h5', '/Users/yaolun/programs/colt-lime/inits/'+args['subpath']+'/'+outfilename+'.h5')
    print('write to '+args['colt_dir']+'inits/'+args['subpath']+'/'+outfilename+suffix+'.h5')
