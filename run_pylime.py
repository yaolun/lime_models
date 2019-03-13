# script for iterating pylime models
from astropy.io import ascii
import numpy as np
import os
import shutil
from subprocess import Popen, call
import glob
import sys

import argparse
parser = argparse.ArgumentParser(description='Options for running LIME in batch mode')
parser.add_argument('--pathfile', required=True,
                    help='[required] the path file')
parser.add_argument('--image_only', action='store_true',
                    help='only imaging the existing results of LIME')
parser.add_argument('--no_image', action='store_true',
                    help='only run the RTE calculation.')
parser.add_argument('--vr_factor', default='1.0', help='artifically reduce the radial velocity by a factor')
parser.add_argument('--vr_offset', default='0.0', help='additional offset added to vr.  (vr < 0 for infall, thus vr_offset < 0 would increase the infall velocity.)')
parser.add_argument('--age', help='one-time change to the age for the current run [unit: yr]')
parser.add_argument('--omega', help='one-time change to the oemga for the current run [unit: s-1].')
parser.add_argument('--gridding', action='store_true', help='Only run LIME for the gridding purpose.  It will use the "gridding" version in getDensity(), which takes the density profile from the TSC-Fortran output with no cavity.')
parser.add_argument('--dry_run', action='store_true', help='Test run the program until the point where LIME will be executed.')
parser.add_argument('--hybrid_tsc', action='store_true', help='Option to have an angular momentum consered envelope inside the centrifugal radius')

# parser.add_argument('--model_list', help='specify model list other than the default one (model_list.txt)')
args = vars(parser.parse_args())

# read in the path file
path_list = np.genfromtxt(args['pathfile'], dtype=str).T
dict_path = {}
for name, val in zip(path_list[0],path_list[1]):
    dict_path[name] = val

# user-dependent
# if args['model_list'] == None:
    # model_list = ascii.read('/scratch/LIMEmods/pylime/YLY/lime_models/model_list.txt', comment='#')
    # outdir_base = '/scratch/LIMEmods/pylime/YLY/run/'
model_list = ascii.read(dict_path['model_list'], comment='#')
outdir_base = dict_path['outdir']
# else:
#     model_list = ascii.read(dict_path['model_list']+args['model_list']+'.txt', comment='#')
#     outdir_base = '/scratch/LIMEmods/pylime/YLY/run/'+args['model_list']+'/'
# pylime = '/scratch/LIMEmods/pylime/lime/pylime.0504'
pylime = dict_path['pylime']
config_template = open(dict_path['lime_config_template'], 'r').readlines()
# get the keys
p = {}
for line in config_template:
    if line.startswith('#'):
        continue
    p[line.strip().split()[0]] = line.strip().split()[1]

if p['gridIn'] != 'None':
    print('Use existing LIME grid from "{:<s}"'.format(p['gridIn']))

for i, m in enumerate(model_list['model_name']):

    # use the config file as the communication between model.py and user-defined model list
    # foo = open('/scratch/LIMEmods/pylime/YLY/lime_models/lime_config.txt', 'w')
    foo = open(dict_path['limemod_dir']+'lime_config.txt', 'w')

    p['dustfile'] = dict_path['dust_file']
    p['rtout'] = dict_path['hyperion_dir']+model_list['hy_model'][i]+'.rtout'
    p['TSC_dir'] = dict_path['TSC_dir']
    # p['velfile'] = dict_path['tsc_dir']+str(model_list['tsc'][i])
    p['cs'] = str(model_list['cs'][i])
    p['vr_factor'] = args['vr_factor']
    p['vr_offset'] = args['vr_offset']
    p['moldata'] = dict_path['moldata_dir']+model_list['moldata'][i]
    if args['age'] != None:
        p['age'] = args['age']
    if args['hybrid_tsc']:
        p['hybrid_tsc'] = '1'
    else:
        p['hybrid_tsc'] = '0'

    if args['omega'] != None:
        p['omega'] = args['omega']

    # model parameters - only abundance now
    # the names of parameters will be the same as the ones in the header of model_list.txt
    outdir = outdir_base+'model'+str(m)+'/'

    p_names = ['outdir']
    p_values = [outdir]

    p_names.extend(model_list.keys()[1:])
    p_values.extend([str(model_list[_p][i]) for _p in model_list.keys()[1:]])
    if p_values[p_names.index('velfile')] == 'none':
        p_values[p_names.index('velfile')] = outdir+'tsc_regrid.h5'
    p_values[p_names.index('moldata')] = dict_path['moldata_dir']+model_list['moldata'][i]

    # write out the default parameters
    for i, name in enumerate(p.keys()):
        if name not in p_names:
            foo.write('{:<14s}  {:<s}\n'.format(name, p[name]))
    # write out the parameters specified in the model_list
    for i, (name, val) in enumerate(zip(p_names, p_values)):
        if name in ['hy_model']:
            continue
        foo.write('{:<14s}  {:<s}\n'.format(name, val))

    foo.close()

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # make a copy of config and model.py to the model directory
    # user-dependent
    shutil.copyfile(dict_path['limemod_dir']+'lime_config.txt',
                    outdir+'lime_config.txt')
    # copy the lime_config.txt to the same directory of model.py
    if dict_path['limemod_dir']+'lime_config.txt' != os.getcwd()+'/lime_config.txt':
        shutil.copyfile(dict_path['limemod_dir']+'lime_config.txt', os.getcwd()+'/lime_config.txt')
    shutil.copyfile('model.py', outdir+'model.py')
    shutil.copyfile('Hyperion2LIME.py', outdir+'Hyperion2LIME.py')

    # make sure the "image_only" file is reset everytime LIME runs
    if os.path.exists(outdir+'image_only'):
        os.remove(outdir+'image_only')
    # make sure the "no_image" file is reset everytime LIME runs
    if os.path.exists(outdir+'no_image'):
        os.remove(outdir+'no_image')
    if args['image_only']:
        if not os.path.exists(outdir+'grid5'):
            print('No appropriate grid file found.  Abort...')
            sys.exit()
        foo = open(outdir+'image_only', 'w')
        foo.close()
    # if not using the "image-only" mode, all files from previous runs of the current model will be deleted
    else:
        for f in glob.glob(outdir+'grid*'):
            os.remove(f)
        for f in glob.glob(outdir+'*pop'):
            os.remove(f)

    if args['no_image']:
        foo = open(outdir+'no_image', 'w')
        foo.close()

    # the gridding option.  Basically the same as "no_rte" except for changing the version of getDensity() to "gridding"
    if (args['gridding']):
        foo = open(outdir+'gridding', 'w')
        foo.close()

    if not args['dry_run']:
        # run pylime - if "image_only" file is presented, it will enter image-only mode
        print('Start running model '+str(m))
        log = open(outdir+'pylime.log','w')
        err = open(outdir+'pylime.err','w')
        run = call([pylime, 'model.py'], stdout=log, stderr=err, env=os.environ.copy())

        if not os.path.exists(outdir+'grid5'):
            print('grid files not found.  pylime probably failed.')
        if not os.path.exists(outdir+'image0.fits'):
            if not args['no_image']:
                print('Image file not found.  pylime probably failed.')
