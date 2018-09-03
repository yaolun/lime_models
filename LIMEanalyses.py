import numpy as np
from astropy.io import fits, ascii
import astropy.constants as const

au_si = const.au.si.value
au_cgs = const.au.cgs.value
c = const.c.cgs.value
h = const.h.cgs.value
mh = const.m_p.cgs.value+const.m_e.cgs.value

class LIMEanalyses:
    """
    Analyses packages for LIME results.
    All output values are in CGS!
    Current functions:
        - grid inspection
    """

    def __init__(self, config=None):

        if config != None:
            config_file = ascii.read(config)
            config = {}
            for name, val in zip(config_file['col1'],config_file['col2']):
                config[name] = val

            self.config = config

    def unpackLIME(self, grid, gridtype):
        """
        this result include the sink points
        """
        self.grid = np.array([list(g) for g in fits.open(grid)[1].data])
        self.gridtype = gridtype

        if gridtype != 5:
            print('unsupported grid type', gridtype)
            return None

        grid = self.getGrid()
        sph_grid = self.Cart2Spherical(grid)
        velocity = self.getVelocity()
        density = self.getDensity()
        abundance = self.getAbundance()
        Tk = self.getTempK()

        output = {'x': grid[0], 'y': grid[1], 'z': grid[2], 'r': sph_grid[0], 'theta': sph_grid[1], 'phi': sph_grid[2],
                  'vx': velocity[0], 'vy': velocity[1], 'vz': velocity[2],
                  'density': density, 'abundance': abundance, 'Tk': Tk}
        return output

    def unpackLIMEpop(self, grid, gridtype, pop, velfile=None, rtout=None, recalVelo=False):
        """
        this result does not include the sink points
        it can now do velocity, but it is based on the assumption
        that the order in grid is the same as the order in populations.pop
        all outputs in cgs unit
        """
        popdata = ascii.read(pop)
        if gridtype != 5:
            print('unsupported grid type', gridtype)
            return None

        lime_grid = self.unpackLIME(grid, 5)
        v_grid = (lime_grid['vx'][lime_grid['r'] < lime_grid['r'].max()-0.05*au_cgs],
                  lime_grid['vy'][lime_grid['r'] < lime_grid['r'].max()-0.05*au_cgs],
                  lime_grid['vz'][lime_grid['r'] < lime_grid['r'].max()-0.05*au_cgs])

        sph_grid = self.Cart2Spherical((popdata['x']*1e2, popdata['y']*1e2, popdata['z']*1e2))

        # check if the velocity grid has the same length as the population grid
        # sometime LIME outputs these two arrays in different size
        if len(popdata['x']) < len(v_grid[0]):
            print('The length of population grid is smaller than the LIME grid\nRe-calculate the velocity with Hyperion2LIME.')
            # use the H2L velocity function to re-calculate the velocity
            if velfile == None:
                velfile = input('Where is the TSC velocity file?')
            if rtout == None:
                rtout = input('Where is the Hyperion output file?')
            recalVelo = True

        elif len(popdata['x']) > len(v_grid[0]):
            print('The population grid ({:<d}) is larger than the velocity grid ({:<d}).\nThis never happen, abort abort!'.format(len(v_grid[0]), len(popdata['x'])))
            return None

        if recalVelo:
            from Hyperion2LIME import Hyperion2LIME
            model = Hyperion2LIME(rtout, velfile, float(self.config['cs']), float(self.config['age']),
                                  rmin=float(self.config['rMin'])*au_si, g2d=float(self.config['g2d']), mmw=float(self.config['mmw']))
            v_grid = [[], [], []]
            for i, (x,y,z) in enumerate(zip(popdata['x'], popdata['y'], popdata['z'])):  # unit is meter here
                # v = model.getVelocity2(x,y,z)
                v = model.getVelocity(x,y,z)
                v_grid[0].append(v[0]*1e2)
                v_grid[1].append(v[1]*1e2)
                v_grid[2].append(v[2]*1e2)

        output = {'x': popdata['x']*1e2, 'y': popdata['y']*1e2, 'z': popdata['z']*1e2,
                  'r': sph_grid[0], 'theta': sph_grid[1], 'phi': sph_grid[2],
                  'vx': v_grid[0], 'vy': v_grid[1], 'vz': v_grid[2],
                  'density': popdata['H2_density']/1e6,
                  'abundance': popdata['molecular_abundance'],
                  'Tk': popdata['kinetic_gas_temperature']}
        return output, popdata

    def Cart2Spherical(self, grid):
        # convert arrays of x, y, z to r, theta, phi
        x_arr, y_arr, z_arr = grid

        r = (x_arr**2+y_arr**2+z_arr**2)**0.5
        t = np.arccos(z_arr/r)
        t[r == 0] = 0
        p = np.arctan2(y_arr, x_arr)

        return (r, t, p)

    def getGrid(self):
        # get the x,y,z grids of the sampled model
        x = self.grid[:,1]*1e2
        y = self.grid[:,2]*1e2
        z = self.grid[:,3]*1e2

        return (x,y,z)

    def getVelocity(self):
        # get vx, vy, vz
        vx = self.grid[:,7]*1e2
        vy = self.grid[:,8]*1e2
        vz = self.grid[:,9]*1e2

        return (vx,vy,vz)

    def getAbundance(self):
        # get abundance
        return self.grid[:,11]

    def getDensity(self):
        # get density
        # unit: cm-3  (I think it is number density although the FITS header says it's mass density)

        return self.grid[:,10]*1e-6

    def getTempK(self):
        # get the kinetic temperature of gas
        return self.grid[:,13]

    def getTempDust(self):
        # get the dust temperature
        return self.grid[:,14]

    def LIME2COLT(self, grid, gridtype, pop, auxdata, velfile=None, rtout=None, recalVelo=False):

        c = const.c.cgs.value
        h = const.h.cgs.value

        output, popdata = self.unpackLIMEpop(grid, gridtype, pop, velfile=velfile, rtout=rtout, recalVelo=recalVelo)
        n1 = output['density']*output['abundance']*popdata['pops_'+str(auxdata['trans_up']-1)]  # number density (1/cm3)
        n2 = output['density']*output['abundance']*popdata['pops_'+str(auxdata['trans_up'])]    # number density (1/cm3)
        # B21 = auxdata['EA']*c**3/(8*np.pi*h*auxdata['nu0']**3)
        B21 = auxdata['EA']*c**2/(2*h*auxdata['nu0']**3)
        B12 = B21*auxdata['degeneracy'][0]/auxdata['degeneracy'][1]   # (upper, lower)

        auxdata['r_max'] = float(self.config['rMax'])

        # gas
        jv_gas = h*auxdata['nu0']/(4*np.pi)*n2*auxdata['EA']
        av_gas = h*auxdata['nu0']/(4*np.pi)*(n1*B12-n2*B21)

        def Planck(nu, T):
            import astropy.constants as const
            h = const.h.cgs.value
            k = const.k_B.cgs.value
            c = const.c.cgs.value

            # for given temperature, calculate the corresponding B_v
            B_v = 2*h*nu**3/c**2*(np.exp(h*nu/k/T)-1)**-1

            return B_v

        # dust
        g2d = 100
        # kappa_v in cm2 per gram of dust
        av_dust = auxdata['kappa_v']*output['density']*float(self.config['mmw'])*mh/g2d
        jv_dust = -av_dust*Planck(auxdata['nu0'], output['Tk'])

        output['jv_gas'] = jv_gas
        output['av_gas'] = av_gas
        output['jv_dust'] = jv_dust
        output['av_dust'] = av_dust

        return output, auxdata
