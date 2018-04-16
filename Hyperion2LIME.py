from hyperion.model import ModelOutput
import numpy as np
import astropy.io as io
import astropy.constants as const
mh = const.m_p.cgs.value+const.m_e.cgs.value
au_cgs = const.au.cgs.value
au_si = const.au.si.value

class Hyperion2LIME:
    """
    Class for importing Hyperion result to LIME
    IMPORTANT: LIME uses SI units, while Hyperion uses CGS units.
    """

    def __init__(self, rtout, velfile, cs, age,
                 rmin=0, mmw=2.37, g2d=100, truncate=None):
        self.rtout = rtout
        self.velfile = velfile
        self.hyperion = ModelOutput(rtout)
        self.hy_grid = self.hyperion.get_quantities()
        self.rmin = rmin*1e2        # rmin defined in LIME, which use SI unit
        self.mmw = mmw
        self.g2d = g2d
        self.cs = cs
        self.age = age
        # option to truncate the sphere to be a cylinder
        # the value is given in au to specify the radius of the truncated cylinder viewed from the observer
        self.truncate = truncate

        # velocity grid construction
        self.tsc = io.ascii.read(self.velfile)
        self.xr = np.unique(self.tsc['xr'])  # reduce radius: x = r/(a*t) = r/r_inf
        self.xr_wall = np.hstack(([2*self.xr[0]-self.xr[1]],
                                 (self.xr[:-1]+self.xr[1:])/2,
                                 [2*self.xr[-1]-self.xr[-2]]))
        self.theta = np.unique(self.tsc['theta'])
        self.theta_wall = np.hstack(([2*self.theta[0]-self.theta[1]],
                                (self.theta[:-1]+self.theta[1:])/2,
                                [2*self.theta[-1]-self.theta[-2]]))
        self.nxr = len(self.xr)
        self.ntheta = len(self.theta)
        self.r_inf = self.cs*1e5*self.age*3600*24*365  # in cm

        self.vr2d = self.tsc['ur'].reshape([self.nxr, self.ntheta]) * self.cs*1e5
        self.vtheta2d = self.tsc['utheta'].reshape([self.nxr, self.ntheta]) * self.cs*1e5
        self.vphi2d = self.tsc['uphi'].reshape([self.nxr, self.ntheta]) * self.cs*1e5


    def Cart2Spherical(self, x, y, z, unit_convert=True):
        """
        if unit_convert, the inputs (x, y, z) are meter.
        The outputs are in cm.
        """
        if unit_convert:
            x, y, z = x*1e2, y*1e2, z*1e2

        r_in = (x**2+y**2+z**2)**0.5
        if r_in != 0:
            t_in = np.arccos(z/r_in)
        else:
            t_in = 0
        if x != 0:
            p_in = np.sign(y)*np.arctan(y/x)  # the input phi is irrelevant in axisymmetric model
        else:
            p_in = np.sign(y)*np.pi/2

        if r_in < self.rmin:
            r_in = self.rmin

        return (r_in, t_in, p_in)

    def Spherical2Cart_vector(self, coord_sph, v_sph):
        r, theta, phi = coord_sph
        vr, vt, vp = v_sph

        transform = np.matrix([[np.sin(theta)*np.cos(phi)  , np.cos(theta)*np.cos(phi), -np.sin(phi)],
                               [np.sin(theta)*np.sin(phi)  , np.cos(theta)*np.sin(phi), np.cos(phi)],
                               [np.cos(theta)              , -np.sin(theta)           , 0]])
        v_cart = transform.dot(np.array([vr, vt, vp]))

        return list(map(float, np.asarray(v_cart).flatten()))


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

        if self.truncate != None:
            if (y**2+z**2)**0.5 > self.truncate*au_si:
                return 0.0

        (r_in, t_in, p_in) = self.Cart2Spherical(x, y, z)

        indice = self.locateCell((r_in, t_in, p_in), (r_wall, t_wall, p_wall))

        # LIME needs molecule number density per cubic meter

        return float(self.rho[indice])*self.g2d/mh/self.mmw*1e6

    def getTemperature(self, x, y, z):
        r_wall = self.hy_grid.r_wall
        t_wall = self.hy_grid.t_wall
        p_wall = self.hy_grid.p_wall
        self.temp = self.hy_grid.quantities['temperature'][0].T

        if self.truncate != None:
            if (y**2+z**2)**0.5 > self.truncate*au_si:
                return 0.0

        (r_in, t_in, p_in) = self.Cart2Spherical(x, y, z)

        indice = self.locateCell((r_in, t_in, p_in), (r_wall, t_wall, p_wall))

        return float(self.temp[indice])

    def getVelocity(self, x, y, z):
        """
        cs: effecitve sound speed in km/s;
        age: the time since the collapse began in year.
        """

        (r_in, t_in, p_in) = self.Cart2Spherical(x, y, z)

        if self.truncate != None:
            if (y**2+z**2)**0.5 > self.truncate*au_si:
                v_out = [0.0, 0.0, 0.0]
                return v_out

        # outside of infall radius, the envelope is static
        if r_in > self.r_inf:
            v_out = [0.0, 0.0, 0.0]
            return v_out

        # if the input radius is smaller than the minimum in xr array,
        # use the minimum in xr array instead.
        if r_in < self.xr_wall.min()*self.r_inf:
            r_in = self.xr.min()*self.r_inf
            # TODO: raise warning

        ind = self.locateCell2d((r_in, t_in), (self.xr_wall*self.r_inf, self.theta_wall))
        v_sph = list(map(float, [self.vr2d[ind]/1e2, self.vtheta2d[ind]/1e2, self.vphi2d[ind]/1e2]))

        v_out = self.Spherical2Cart_vector((r_in, t_in, p_in), v_sph)

        return v_out

    def getAbundance(self, x, y, z, a_params, tol=10):
        # tol: the size (in AU) of the linear region between two steps
        # (try to avoid "cannot find cell" problem in LIME)

        # a_params = [abundance at outer region,
        #             fraction of outer abundance to the inner abundance,
        #             the ratio of the outer radius of the inner region to the infall radius]

        # abundances = [3.5e-8, 3.5e-9]  # inner, outer

        if self.truncate != None:
            if (y**2+z**2)**0.5 > self.truncate*au_si:
                return 0.0

        tol = tol*au_cgs

        (r_in, t_in, p_in) = self.Cart2Spherical(x, y, z)

        if (r_in - a_params[2]*self.r_inf) > tol/2:
            abundance = a_params[0]
        elif abs(r_in - a_params[2]*self.r_inf) <= tol/2:
            abundance = a_params[0] + \
                        (r_in-a_params[2]*self.r_inf+tol/2)*(a_params[1]-a_params[0])/tol
        else:
            abundance = a_params[0]

        # uniform abundance
        # abundance = 3.5e-9

        return abundance
