import numpy as np
from astropy.io import fits
import astropy.constants as const

au_si = const.au.si.value

class LIMEanalyses:
    """
    Analyses packages for LIME results.
    All output values are in CGS!
    Current functions:
        - grid inspection
    """

    def __init__(self, grid, gridtype):
        self.grid = np.array([list(g) for g in fits.open(grid)[1].data])
        self.gridtype = gridtype
        if gridtype != 5:
            print('unsupported grid type', gridtype)
            return None

    def unpackLIME(self):

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

    def Cart2Spherical(self, grid):
        # convert arrays of x, y, z to r, theta, phi
        x_arr, y_arr, z_arr = grid

        r = (x_arr**2+y_arr**2+z_arr**2)**0.5
        t = np.arccos(z_arr/r)
        t[r == 0] = 0
        p = np.arctan(y_arr/x_arr)
        p[x_arr == 0] = np.sign(y_arr[x_arr == 0])*np.pi/2

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
