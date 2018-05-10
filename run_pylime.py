# script for iterating pylime models
from astropy.io import ascii
import os
import shutil
from subprocess import Popen, call
import glob
import sys

import argparse
parser = argparse.ArgumentParser(description='Options for running LIME in batch mode')
parser.add_argument('--image_only', action='store_true',
                    help='only imaging the existing results of LIME')
args = vars(parser.parse_args())


model_list = ascii.read('/scratch/LIMEmods/pylime/YLY/lime_models/model_list.txt', comment='#')
outdir_base = '/scratch/LIMEmods/pylime/YLY/run/'
pylime = '/scratch/LIMEmods/pylime/lime/pylime'

for m in model_list['model_name']:

    # use the config file as the communication between model.py and user-defined model list
    foo = open('/scratch/LIMEmods/pylime/YLY/lime_models/lime_config.txt', 'w')

    # default parameters - the parameters that typically fixed and not defined in the model list
    p_names = ['mmw', 'g2d', 'dustfile', 'pIntensity', 'sinkPoints',
               'rtout', 'velfile', 'cs', 'age', 'rMin', 'rMax', 'distance']
    p_values = ['2.37', '100', '/scratch/LIMEmods/pylime/YLY/lime_models/dust_oh5.txt',
                '50000', '8000', '/scratch/LIMEmods/pylime/YLY/model12.rtout',
                '/scratch/LIMEmods/pylime/YLY/rho_v_env', '0.38', '36000',
                '0.2', '64973', '200.0']

    for i, (name, val) in enumerate(zip(p_names, p_values)):
        foo.write('{:<14s}  {:<s}\n'.format(name, val))

    # model parameters - only abundance now
    # the names of parameters will be the same as the ones in the header of model_list.txt
    outdir = outdir_base+'model'+str(m)+'/'
    p_names = ['outdir']
    p_names.extend(model_list.keys()[1:])
    p_values = [outdir]
    p_values.extend([str(model_list[p][model_list['model_name'] == m].data[0]) for p in model_list.keys()[1:]])

    for i, (name, val) in enumerate(zip(p_names, p_values)):
        foo.write('{:<14s}  {:<s}\n'.format(name, val))

    foo.close()

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # make a copy of config and model.py to the model directory
    shutil.copyfile('/scratch/LIMEmods/pylime/YLY/lime_models/lime_config.txt',
                    outdir+'lime_config.txt')
    shutil.copyfile('model.py', outdir+'model.py')

    
    # make sure the "image_only" file is reset everytime LIME run
    if os.path.exists(outdir+'image_only'):
        os.remove(outdir+'image_only')
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

    # run pylime - if "image_only" file is presented, it will enter image-only mode
    print('Start running model '+str(m))
    log = open(outdir+'pylime.log','w')
    err = open(outdir+'pylime.err','w')
    run = call(['pylime', 'model.py'], stdout=log, stderr=err)

    if not os.path.exists(outdir+'grid5'):
        print('grid files not found.  pylime probably failed.')
    if not os.path.exists(outdir+'image0.fits'):
        print('Image file not found.  pylime probably failed.')
