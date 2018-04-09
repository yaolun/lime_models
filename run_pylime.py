# script for iterating pylime models
from astropy.io import ascii
import os
import shutil
from subprocess import Popen, call

model_list = ascii.read('/scratch/LIMEmods/pylime/lime/YLY/lime_models/model_list.txt')
outdir_base = '/scratch/LIMEmods/pylime/lime/YLY/run/'
pylime = '/scratch/LIMEmods/pylime/lime/pylime'

for m in model_list['model_name']:
    # use the config file as the communication between model.py and user-defined model list
    foo = open('/scratch/LIMEmods/pylime/lime/YLY/lime_models/lime_config.txt', 'w')

    # default parameters - the parameters that typically fixed and not defined in the model list
    p_names = ['mmw', 'g2d', 'dustfile', 'pIntensity', 'sinkPoints',
               'rtout', 'velfile', 'cs', 'age', 'rMin', 'rMax', 'distance']
    p_values = ['2.37', '100', '/scratch/LIMEmods/pylime/lime/YLY/dust_oh5.txt',
                '50000', '8000', '/scratch/LIMEmods/pylime/lime/YLY/model12.rtout',
                '/scratch/LIMEmods/pylime/lime/YLY/rho_v_env', '0.38', '36000',
                '0.2', '64973', '200.0']

    for i, (name, val) in enumerate(zip(p_names, p_values)):
        foo.write('{:<14s}  {:<s}\n'.format(name, val))

    # model parameters - only abundance now
    outdir = outdir_base+'model'+str(m)+'/'
    p_names = ['outdir', 'a_params0', 'a_params1', 'a_params2']
    p_values = [outdir,
                str(model_list['a_params0'][model_list['model_name'] == m].data[0]),
                str(model_list['a_params1'][model_list['model_name'] == m].data[0]),
                str(model_list['a_params2'][model_list['model_name'] == m].data[0])]
    for i, (name, val) in enumerate(zip(p_names, p_values)):
        foo.write('{:<14s}  {:<s}\n'.format(name, val))

    foo.close()

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # make a copy of config to the model directory
    shutil.copyfile('/scratch/LIMEmods/pylime/lime/YLY/lime_models/lime_config.txt',
                    outdir+'lime_config.txt')

    # run pylime
    log = open(outdir+'pylime.log','w')
    err = open(outdir+'pylime.err','w')
    run = Popen([pylime, 'model.py'], stdout=log, stderr=err)
    run.communicate()

    print('Finish model '+str(m))
