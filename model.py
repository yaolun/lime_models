#!/usr/bin/env python
# Notes:
#	- The above pragma line is only required if you plan to run this module as a stand-alone script to run any test harnesses which may occur after "if __name__ == '__main__'".
#
#	- You should run lime with this script in the form
#		pylime model.py
#	  You will need the location of pylime in your PATH environment variable; also you need to have the location of the par_classes.py module in your PYTHONPATH environment variable.
# For definitions of the classes ModelParameters and ImageParameters:

import sys
sys.path.append('/scratch/LIMEmods/pylime/lime/python/')
sys.path.append('/scratch/LIMEmods/pylime/lime/YLY/lime_models/')
from par_classes import *

# import results from Hyperion and TSC calculations
import os
sys.path.append(os.path.expanduser('~')+'/anaconda/lib/python2.7/site-packages/')
from Hyperion2LIME import *
import astropy.constants as const
import astropy.io as io
mh = const.m_p.cgs.value + const.m_e.cgs.value
au = const.au.cgs.value
au_si = const.au.si.value
pc_si = const.pc.si.value

# TODO: read in the following parameters from files
config_file = io.ascii.read('lime_config.txt')
config = {}
for name, val in zip(config_file['col1'],config_file['col2']):
    config[name] = val

pIntensity = int(config['pIntensity'])
sinkPoints = int(config['sinkPoints'])
mmw = float(config['mmw'])
rtout = config['rtout']
velfile = config['velfile']
cs = float(config['cs'])
age = float(config['age'])
g2d = float(config['g2d'])  # gas-to-dust mass ratio
rMax = float(config['rMax'])*au_si
rMin = float(config['rMin'])*au_si # greater than zero to avoid a singularity at the origin.
a_params = [float(config['a_params0']), float(config['a_params1']), float(config['a_params2'])]
distance = float(config['distance'])*pc_si

# path
outdir = str(config['outdir'])
dustfile = str(config['dustfile'])
print('output directory = '+outdir)

# TODO: rmax, rmin, dustfile, distance, inclination
# truncate = 2000.0  # in unit of au
truncate = None
model = Hyperion2LIME(rtout, velfile, cs, age, rmin=rMin, g2d=g2d, mmw=mmw, truncate=truncate)



# Note that the useful macros defined in lime.h are also provided here in the dictionary 'macros' provided as an argument to each function below. See the example code at the end for the full list of macro values provided.

#.......................................................................
def input(macros):
    par = ModelParameters()

    # We give all the possible parameters here, but have commented out many which can be left at their defaults.

    # Parameters which must be set (they have no sensible defaults).
    #
    # TODO: review the choice of these parameters
    par.radius            = rMax
    par.minScale          = rMin
    par.pIntensity        = pIntensity  # number of model grid points
    par.sinkPoints        = sinkPoints   # grid points that are distributed randomly at surface of the model

    # Parameters which may be omitted (i.e. left at their default values) under some circumstances.
    #
    par.dust              = dustfile  # opacity per gram of dust.  This is only used for generating continuum image.
    # par.dust              = "jena_thin_e6.tab"  # default dust model, similar to OH5
    par.outputfile        = outdir+"populations.pop"
    par.binoutputfile     = outdir+"restart.pop"  # contains the grid, popilations, and molecular data in binary format for re-imaging
    par.gridfile          = outdir+"grid.vtk"
    #  par.pregrid           = "pregrid.asc"
    #  par.restart           = "restart.pop"
    # par.gridInFile        = ['', '', 'grid3','grid4','grid5']

    #    Setting elements of the following two arrays is optional. NOTE
    #    that, if you do set any of their values, you should set as many as
    #    the number of elements returned by your function density(). The
    #    ith element of the array in question will then be assumed to refer
    #    to the ith element in the density function return. The current
    #    maximum number of elements allowed is 7, which is the number of
    #    types of collision partner recognized in the LAMBDA database.
    #
    #    Note that there is no (longer) a hard connection between the
    #    number of density elements and the number of collision-partner
    #    species named in the moldata files. This means in practice that,
    #    if you set the values for par->collPartIds, you can, if you like,
    #    set some for which there are no transition rates supplied in the
    #    moldatfiles. This might happen for example if there is a molecule
    #    which contributes significantly to the total molecular density but
    #    for which there are no measured collision rates for the radiating
    #    species you are interested in.
    #
    #    You may also omit to mention in par->collPartIds a collision
    #    partner which is specified in the moldatfiles. In this case LIME
    #    will assume the density of the respective molecules is zero.
    #
    #    If you don't set any values for any or all of these arrays,
    #    i.e. if you omit any mention of them here (we preserve this
    #    possibility for purposes of backward compatibility), LIME will
    #    attempt to replicate the algorithms employed in version 1.5, which
    #    involve guessing which collision partner corresponds to which
    #    density element. Since this was not exactly a rigorous procedure,
    #    we recommend use of the arrays.
    #
    #    par->nMolWeights: this specifies how you want the number density
    #    of each radiating species to be calculated. At each grid point a
    #    sum (weighted by par->nMolWeights) of the density values is made,
    #    then this is multiplied by the abundance to return the number
    #    density.
    #
    #    Note that there are convenient macros defined in ../src/lime.h for
    #    7 types of collision partner.
    #
    #    Below is an example of how you might use these parameters:
    #
    par.collPartIds        = [macros["CP_H2"]] # must be a list, even when there is only 1 item.
    par.nMolWeights        = [1.0] # must be a list, even when there is only 1 item.

    #  par.collPartNames     = ["phlogiston"] # must be a list, even when there is only 1 item.
    par.collPartMolWeights = [mmw] # must be a list, even when there is only 1 item.  TODO: review the choice of mmw in the future

    #  par.gridDensMaxValues = [1.0] # must be a list, even when there is only 1 item.
    #  par.gridDensMaxLoc    = [[0.0,0.0,0.0]] # must be a list, each element of which is also a list with 3 entries (1 for each spatial coordinate).

    #  par.tcmb              = 2.72548
    #  par.lte_only          = False
    #  par.init_lte          = False
    par.samplingAlgorithm = 1  # TODO: may want to try "1" in the future to employ a faster sampling
    par.sampling          = 2  # Now only accessed if par.samplingAlgorithm==0 (the default).
    #  par.blend             = False
    #  par.polarization      = False
    # par.nThreads          = 1
    par.nSolveIters       = 17
    par.traceRayAlgorithm = 1
    #  par.resetRNG          = False
    #  par.doSolveRTE        = False
    # par.gridOutFiles      = ['', '', outdir+'grid3', outdir+'grid4', outdir+'grid5'] # must be a list with 5 string elements, although some or all can be empty.
    # par.gridInFiles       = ['', '', outdir+'grid3', outdir+'grid4', outdir+'grid5']
    # can use HDF5 format by adding USEHDF5="yes" to the make command
    par.moldatfile        = ["hco+@xpol.dat"] # must be a list, even when there is only 1 item.
    #  par.girdatfile        = ["myGIRs.dat"] # must be a list, even when there is only 1 item.

    if not os.path.exists(outdir+'grid5'):
        print('run in RTE-only mode')
        par.nThreads = 20
        par.doSolveRTE = True
        par.gridOutFiles = [outdir+'grid1', outdir+'grid2', outdir+'grid3', outdir+'grid4', outdir+'grid5']

    else:
        print('run in imaging-only mode')
        par.nThreads = 1
        par.gridInFiles = [outdir+'grid1', outdir+'grid2', outdir+'grid3', outdir+'grid4', outdir+'grid5']
        par.restart = outdir+"restart.pop"

        # Definitions for image #0. Add further similar blocks for additional images.
        #
        par.img.append(ImageParameters())
        # by default this list par.img has 0 entries. Each 'append' will add an entry.
        # The [-1] entry is the most recently added.
        # TODO: review the choice of imaging parameters

        par.img[-1].nchan             = 200            # Number of channels
        par.img[-1].trans             = 3              # zero-indexed J quantum number of the lower level
        #  par.img[-1].molI              = -1
        par.img[-1].velres            = 100.0          # Channel resolution in m/s
        par.img[-1].imgres            = 1.0            # Resolution in arc seconds
        par.img[-1].pxls              = 100            # Pixels per dimension
        par.img[-1].unit              = 0              # 0:Kelvin 1:Jansky/pixel 2:SI 3:Lsun/pixel 4:tau
        #  par.img[-1].freq              = -1.0
        #  par.img[-1].bandwidth         = -1.0
        par.img[-1].source_vel        = 0.0            # source velocity in m/s
        #  par.img[-1].theta             = 0.0
        #  par.img[-1].phi               = 0.0
        par.img[-1].incl              = 40.0            # TODO: read from file
        #  par.img[-1].posang            = 0.0
        #  par.img[-1].azimuth           = 0.0
        par.img[-1].distance          = distance         # source distance in m
        par.img[-1].doInterpolateVels = True
        par.img[-1].filename          = outdir+'image0.fits'  # Output filename
        #  par.img[-1].units             = "0,1"

        print(par.img[-1].filename)

    return par

#.......................................................................
#.......................................................................
# User-defined functions:

#.......................................................................
def density(macros, x, y, z):
    """
    number density of the collision partners

    The value returned should be a list, each element of which is a density
    (in molecules per cubic metre) of a molecular species (or electrons).
    The molecule should be one of the 7 types currently recognized in the
    LAMDA database - see

    http://home.strw.leidenuniv.nl/~moldata/

    Note that these species are expected to be the bulk constituent(s) of
    the physical system of interest rather than species which contribute
    significantly to spectral-line radiation. In LIME such species are often
    called 'collision partners'.

    The identity of each collision partner is provided via the list parameter
    par.collPartIds. If you do provide this, obviously it must have the same
    number and ordering of elements as the density list you provide here;
    if you don't include it, LIME will try to guess the identities of the
    species you provide density values for.
    """

    # rMin = 0.7*macros["AU"] # greater than zero to avoid a singularity at the origin.
    #
    # # Calculate radial distance from origin
    # #
    # r = math.sqrt(x*x+y*y+z*z)
    #
    # # Calculate a spherical power-law density profile
    # # (Multiply with 1e6 to go to SI-units)
    # #
    # if r>rMin:
    #   rToUse = r
    # else:
    #   rToUse = rMin # Just to prevent overflows at r==0!
    #
    # listOfDensities = [1.5e6*((rToUse/(300.0*macros["AU"]))**(-1.5))*1e6] # must be a list, even when there is only 1 item.


    listOfDensities = [model.getDensity(x, y, z)]


    return listOfDensities

#.......................................................................
def temperature(macros, x, y, z):
    """
    This function should return a tuple of 2 temperatures (in kelvin).
    The 2nd is optional, i.e. you can return None for it, and LIME will
    do the rest.
    """

    # # Array containing temperatures as a function of radial
    # # distance from origin (this is an example of a tabulated model)
    # #
    # rToTemp = [
    # [2.0e13, 5.0e13, 8.0e13, 1.1e14, 1.4e14, 1.7e14, 2.0e14, 2.3e14, 2.6e14, 2.9e14],
    # [44.777, 31.037, 25.718, 22.642, 20.560, 19.023, 17.826, 16.857, 16.050, 15.364]
    # ]
    #
    # # Calculate radial distance from origin
    # #
    # r = math.sqrt(x*x+y*y+z*z)
    #
    # # Linear interpolation in temperature input
    # #
    # xi = 0
    # if r>rToTemp[0][0] and r<rToTemp[0][9]:
    # for i in range(9):
    #   if r>rToTemp[0][i] and r<rToTemp[0][i+1]: xi=i
    #
    # # YLY: flat extrapolation beyond the defined r-region.
    # #      linear interpolation between the defined temperature-radius relation.
    # if r<rToTemp[0][0]:
    #     temp0 = rToTemp[1][0]
    # elif r>rToTemp[0][9]:
    #     temp0 = rToTemp[1][9]
    # else:
    #     temp0 = rToTemp[1][xi]+(r-rToTemp[0][xi])*(rToTemp[1][xi+1]-rToTemp[1][xi])\
    #           / (rToTemp[0][xi+1]-rToTemp[0][xi])

    #  return (temp0, None)

    temp_dust = model.getTemperature(x, y, z)

    return [temp_dust, temp_dust]

#.......................................................................
def abundance(macros, x, y, z):
    """
    This function should return a list of abundances (as fractions of
    the effective bulk density), 1 for each of the radiating species.
    Note that the number and identity of these species is set via the list of
    file names you provide in the par.moldatfile parameter, so make sure
    at least that the number of elements returned by abundance() is the same as
    the number in par.moldatfile!

    Note that the 'effective bulk density' mentioned just above is calculated
    as a weighted sum of the values returned by the density() function,
    the weights being provided in the par.nMolWeights parameter.
    """

    # Here we use a constant abundance. Could be a
    # function of (x,y,z).
    #
    # listOfAbundances = [1.0e-9] # must be a list, even when there is only 1 item.

    listOfAbundances = [model.getAbundance(x, y, z, a_params)]

    return listOfAbundances

#.......................................................................
def doppler(macros, x, y, z):
    """
    in m/s

    This function returns the Doppler B parameter, defined in terms of
    a Doppler-broadened Gaussian linewidth as follows:

                 ( -[v-v0]^2 )
    flux(v) = exp(-----------).
                 (    B^2    )

    Note that the present value refers only to the Doppler broadening
    due to bulk turbulence; LIME later adds in the temperature-dependent part
    (which also depends on molecular mass).
    """

    # 200 m/s as the doppler b-parameter. This
    # can be a function of (x,y,z) as well.
    # Note that *doppler is a pointer, not an array.
    # Remember the * in front of doppler.
    #
    dopplerBValue = 340.0
    # 0.34 km/s 1-D turbulent velocity from Yang+2017.
    # TODO: clarify if the broadening used here need 1-D turbulent velocity or 3-D.

    return dopplerBValue

#.......................................................................
def velocity(macros, x, y, z):
    """
    Gives the bulk gas velocity vector in m/s.
    """

    # rMin = 0.1*macros["AU"] # greater than zero to avoid a singularity at the origin.
    #
    # # Calculate radial distance from origin
    # #
    # r = math.sqrt(x*x+y*y+z*z)
    #
    # if r > rMin:
    #     rToUse = r
    # else:
    #     rToUse = rMin # Just to prevent overflows at r==0!
    #
    # # Free-fall velocity in the radial direction onto a central
    # # mass of 1.0 solar mass
    # #
    # ffSpeed = math.sqrt(2.0*macros["GRAV"]*1.989e30/rToUse)
    #
    # vel = [0,0,0] # just to initialize its size.
    # vel[0] = -x*ffSpeed/rToUse
    # vel[1] = -y*ffSpeed/rToUse
    # vel[2] = -z*ffSpeed/rToUse

    vel = model.getVelocity(x, y, z)

    # debug
    # foo = open('h2l.log', 'a')
    # foo.write('%e \t %e \t %e \t %f \t %f \t %f \n' % (x,y,z,vel[0],vel[1],vel[2]))
    # foo.close()

    return vel

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if __name__ == '__main__':
  # Put any private debugging tests here, which you can then run by calling the module directly from the unix command line.

  macros = {\
    "AMU"           :1.66053904e-27,\
    "CLIGHT"        :2.99792458e8,\
    "HPLANCK"       :6.626070040e-34,\
    "KBOLTZ"        :1.38064852e-23,\
    "GRAV"          :6.67428e-11,\
    "AU"            :1.495978707e11,\
    "LOCAL_CMB_TEMP":2.72548,\
    "PC"            :3.08567758e16,\
    "PI"            :3.14159265358979323846,\
    "SPI"           :1.77245385091,\
    "CP_H2"   :1,\
    "CP_p_H2" :2,\
    "CP_o_H2" :3,\
    "CP_e"    :4,\
    "CP_H"    :5,\
    "CP_He"   :6,\
    "CP_Hplus":7\
  }

  par = input(macros)

  x = par.radius*0.1
  y = par.radius*0.07
  z = par.radius*0.12

  print density(    macros, x, y, z)[0]
  print temperature(macros, x, y, z)[0]
  print doppler(    macros, x, y, z)
  print velocity(   macros, x, y, z)
