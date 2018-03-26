from hyperion.model import ModelOutput
import numpy as np
import astropy.io as io

class Hyperion2LIME:
    """
    Class for importing Hyperion result to LIME
    """

    def __init__(self, rtout, velfile):
        self.rtout = rtout
        self.velfile = velfile
        self.hyperion = ModelOutput(rtout)
        self.hy_grid = self.hyperion.get_quantities()

    def locateCell(self, coord, wall_grid):
        """
        return the indice of cell at given coordinates
        """
        r, t, p = coord
        r_wall, t_wall, p_wall = wall_grid

        r_ind = min(np.argsort(abs(r_wall-r))[:2])
        t_ind = min(np.argsort(abs(t_wall-t))[:2])
        p_ind = min(np.argsort(abs(p_wall-p))[:2])

        return (r_ind, t_ind, p_ind)

    def locateCell2d(self, coord, wall_grid):
        """
        return the indice of cell at given coordinates
        """
        r, t= coord
        r_wall, t_wall = wall_grid

        r_ind = min(np.argsort(abs(r_wall-r))[:2])
        t_ind = min(np.argsort(abs(t_wall-t))[:2])

        return (r_ind, t_ind)


    def getDensity(self, x, y, z):
        r_wall = self.hy_grid.r_wall
        t_wall = self.hy_grid.t_wall
        p_wall = self.hy_grid.p_wall
        self.rho = self.hy_grid.quantities['density'][0].T

        r_in = (x**2+y**2+z**2)**0.5
        if r_in != 0:
            t_in = np.arccos(z/r_in)
        else:
            t_in = 0
        if x != 0:
            p_in = np.arctan(y/x)  # the input phi is irrelevant in axisymmetric model
        else:
            p_in = np.sign(y)*np.pi/2

        indice = self.locateCell((r_in, t_in, p_in), (r_wall, t_wall, p_wall))

        return float(self.rho[indice])

    def getTemperature(self, x, y, z):
        r_wall = self.hy_grid.r_wall
        t_wall = self.hy_grid.t_wall
        p_wall = self.hy_grid.p_wall
        self.temp = self.hy_grid.quantities['temperature'][0].T

        r_in = (x**2+y**2+z**2)**0.5
        t_in = np.arccos(z/r_in)
        p_in = np.arctan(y/x)  # the input phi is irrelevant in axisymmetric model

        indice = self.locateCell((r_in, t_in, p_in), (r_wall, t_wall, p_wall))

        return float(self.temp[indice])

    def getVelocity(self, x, y, z, cs, age):
        """
        cs: effecitve sound speed in km/s;
        age: the time since the collapse began in year.
        """
        r_inf = cs*1e5*age*3600*24*365

        r_in = (x**2+y**2+z**2)**0.5
        t_in = np.arccos(z/r_in)
        p_in = np.arctan(y/x)  # the input phi is irrelevant in axisymmetric model

        # outside of infall radius, the envelope is static
        if r_in > r_inf:
            v_out = np.array([0,0,0])
            return v_out

        tsc = io.ascii.read(self.velfile)

        xr = np.unique(tsc['xr'])  # reduce radius: x = r/(a*t) = r/r_inf
        xr_wall = np.hstack(([2*xr[0]-xr[1]], (xr[:-1]+xr[1:])/2, [2*xr[-1]-xr[-2]]))
        theta = np.unique(tsc['theta'])
        theta_wall = np.hstack(([2*theta[0]-theta[1]],
                                (theta[:-1]+theta[1:])/2,
                                [2*theta[-1]-theta[-2]]))
        nxr = len(xr)
        ntheta = len(theta)

        # if the input radius is smaller than the minimum in xr array,
        # use the minimum in xr array instead.
        if r_in < xr_wall.min()*r_inf:
            r_in = xr.min()*r_inf
            # TODO: raise warning

        vr2d = tsc['ur'].reshape([nxr, ntheta]) * cs*1e5
        vtheta2d = tsc['utheta'].reshape([nxr, ntheta]) * cs*1e5
        vphi2d = tsc['uphi'].reshape([nxr, ntheta]) * cs*1e5

        ind = self.locateCell2d((r_in, t_in), (xr_wall*r_inf, theta_wall))
        v_out = map(float, [vr2d[ind], vtheta2d[ind], vphi2d[ind]])

        return v_out

    def getAbundance(self, x, y, z):
        return 1.0e-9
