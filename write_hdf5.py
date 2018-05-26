import numpy as np
import h5py
import astropy.constants as const
from scipy.interpolate import interp1d
from astropy.io import ascii
pc = const.pc.cgs.value
c = const.c.cgs.value
mh = const.m_p.cgs.value+const.m_e.cgs.value
mmw = 2.37

from LIMEanalyses import *
mod_dir = '/Volumes/SD-Mac/lime_runs/model28/'
grid = mod_dir+'grid5'
pop = mod_dir+'populations.pop'
config = mod_dir+'lime_config.txt'

# HCO+ 4-3
auxdata = {'EA': 3.6269e-03, 'nu0': 356.7342880e9, 'trans_up': 4, 'degeneracy': [9,7]}  # degeneracy from upper to lower

# load dust opacity
g2d = 100
dust_lime = ascii.read('/Volumes/SD-Mac/Google Drive/research/lime_models/dust_oh5.txt', names=['wave', 'kappa_gas'])
f_dust = interp1d(dust_lime['wave'], dust_lime['kappa_gas']*g2d)
kappa_v_dust = f_dust(c/auxdata['nu0']*1e4)
auxdata['kappa_v'] = kappa_v_dust

lime_out = LIMEanalyses(config=config).LIME2COLT(grid, 5, pop, auxdata)

def write_hdf5(lime_out, filename='infall.h5'):
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
    d_L     = 200*pc # Distance to source (cm)
    arcsec2 = 0.1**2 # Size of each pixel (arcsec^2)
    n_pix   = 200*200 # Number of pixels

    with h5py.File(filename, 'w') as f:
        f.attrs['n_cells'] = n_cells
        # if you need to ensure np.float64
        # f.create_dataset('T', data=np.array(T, dtype=np.float64)) # Temperature (K)
        f.create_dataset('T', data=T) # Temperature (K)
        f['T'].attrs['units'] = b'K'
        f.create_dataset('r', data=r) # Positions (cm)
        f['r'].attrs['units'] = b'cm'
        f.create_dataset('v', data=v) # Velocities (cm/s)
        f['v'].attrs['units'] = b'cm/s'
        f.create_dataset('k_gas', data=k_gas) # Gas absorption coefficient (1/cm)
        f['k_gas'].attrs['units'] = b'1/cm'
        f.create_dataset('j_gas', data=j_gas) # Gas emissivity (erg/s/cm^3/Hz)
        f['j_gas'].attrs['units'] = b'erg/s/cm^3/Hz'
        f.create_dataset('rho_dust', data=rho_dust) # Dust density (g/cm^3 of dust)
        f['rho_dust'].attrs['units'] = b'g/cm^3'

def read_hdf5(filename='infall.h5'):
    with h5py.File(filename, 'r') as f:
        print('Opened file:', filename)
        print('Attributes:', [item for item in f.attrs.items()])
        print('Datasets: ', [key for key in f.keys()])
        n_cells = f.attrs['n_cells']
        T = f['T'][:]
        r = f['T'][:,:]

write_hdf5(lime_out)
read_hdf5()
