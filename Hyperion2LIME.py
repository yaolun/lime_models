from hyperion.model import ModelOutput
import numpy as np
import pandas as pd
import astropy.io as io
import astropy.constants as const
from astropy.convolution import convolve, Box1DKernel
from scipy.interpolate import interp2d
mh = const.m_p.cgs.value+const.m_e.cgs.value
MS = const.M_sun.cgs.value
G = const.G.cgs.value
au_cgs = const.au.cgs.value
au_si = const.au.si.value

class Hyperion2LIME:
    """
    Class for importing Hyperion result to LIME
    IMPORTANT: LIME uses SI units, while Hyperion uses CGS units.
    """

    def __init__(self, rtout, velfile, cs, age,
                 rmin=0, mmw=2.37, g2d=100, truncate=None, debug=False, load_full=True):
        self.rtout = rtout
        self.velfile = velfile
        if load_full:
            self.hyperion = ModelOutput(rtout)
            self.hy_grid = self.hyperion.get_quantities()
        self.rmin = rmin*1e2        # rmin defined in LIME, which use SI unit
        self.mmw = mmw
        self.g2d = g2d
        self.cs = cs
        self.age = age
        self.r_inf = self.cs*1e5*self.age*3600*24*365  # in cm

        # option to truncate the sphere to be a cylinder
        # the value is given in au to specify the radius of the truncated cylinder viewed from the observer
        self.truncate = truncate

        # debug option: print out every call to getDensity, getVelocity and getAbundance
        self.debug = debug

        # velocity grid construction
        if load_full:
            # ascii.read() fails for large file.  Use pandas instead
            self.tsc = pd.read_table(velfile, skiprows=1, delim_whitespace=True, header=None)
            self.tsc.columns = ['lp', 'xr', 'theta', 'ro', 'ur', 'utheta', 'uphi']

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

            self.vr2d = np.array(self.tsc['ur']).reshape([self.nxr, self.ntheta]) * self.cs*1e5
            self.vtheta2d = np.array(self.tsc['utheta']).reshape([self.nxr, self.ntheta]) * self.cs*1e5
            self.vphi2d = np.array(self.tsc['uphi']).reshape([self.nxr, self.ntheta]) * self.cs*1e5
            self.tsc2d = {'vr2d': self.vr2d, 'vtheta2d': self.vtheta2d, 'vphi2d': self.vphi2d}


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

        # if x != 0:
            # p_in = np.sign(y)*np.arctan(y/x)  # the input phi is irrelevant in axisymmetric model
        # else:
            # p_in = np.sign(y)*np.pi/2
        p_in = np.arctan2(y, x)

        if r_in < self.rmin:
            r_in = self.rmin

        return (r_in, t_in, p_in)

    def Spherical2Cart(self, r, t, p):
        """
        This is only valid for axisymmetric model
        """
        x = r*np.sin(t)*np.cos(p)
        y = r*np.sin(t)*np.sin(p)
        z = r*np.cos(t)

        return (x, y, z)

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

        if self.debug:
            foo = open('density.log', 'a')
            foo.write('%e \t %e \t %e \t %e\n' % (x,y,z,float(self.rho[indice])*self.g2d/mh/self.mmw*1e6))
            foo.close()

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

    def getVelocity(self, x, y, z, sph=False, unit_convert=True):
        """
        cs: effecitve sound speed in km/s;
        age: the time since the collapse began in year.
        """

        (r_in, t_in, p_in) = self.Cart2Spherical(x, y, z, unit_convert=unit_convert)

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
        # UPDATE (081518): return zero velocity instead
        if r_in < self.xr_wall.min()*self.r_inf:
            r_in = self.xr.min()*self.r_inf

            v_out = [0.0, 0.0, 0.0]
            return v_out

        ind = self.locateCell2d((r_in, t_in), (self.xr_wall*self.r_inf, self.theta_wall))
        v_sph = list(map(float, [self.vr2d[ind]/1e2, self.vtheta2d[ind]/1e2, self.vphi2d[ind]/1e2]))
        if sph:
            return v_sph

        v_out = self.Spherical2Cart_vector((r_in, t_in, p_in), v_sph)

        if self.debug:
            foo = open('velocity.log', 'a')
            foo.write('%e \t %e \t %e \t %f \t %f \t %f\n' % (x, y, z, v_out[0], v_out[1], v_out[2]))
            foo.close()

        return v_out

    def getSakaiVelocity(self, x, y, z, J, M, sph=False, unit_convert=True):
        """
        cs: effecitve sound speed in km/s;
        age: the time since the collapse began in year.
        """

        (r_in, t_in, p_in) = self.Cart2Spherical(x, y, z, unit_convert=unit_convert)

        if self.truncate != None:
            if (y**2+z**2)**0.5 > self.truncate*au_si:
                v_out = [0.0, 0.0, 0.0]
                return v_out

        # if the input radius is smaller than the minimum in xr array,
        # use the minimum in xr array instead.
        # UPDATE: return zero velocity instead
        if r_in < self.xr_wall.min()*self.r_inf:
            r_in = self.xr.min()*self.r_inf

            v_out = [0.0, 0.0, 0.0]
            return v_out

        # use the Sakai model
        M = M*MS
        # centrifugal barrier
        cb = J**2/(2*G*M)

        if 2*G*M/r_in - J**2/r_in**2 >= 0:
            vr = (2*G*M/r_in - J**2/r_in**2)**0.5
        else:
            vr = 0.0
        # let vk = vp at CB
        M_k = J**2/(G*cb)
        vp = J/r_in
        vk = (G*M_k/r_in)**0.5

        if r_in >= cb:
            v_sph = [-vr/1e2, 0.0, vp/1e2]
        else:
            v_sph = [-vr/1e2, 0.0, vk/1e2]

        if sph:
            return v_sph

        v_out = self.Spherical2Cart_vector((r_in, t_in, p_in), v_sph)

        if self.debug:
            foo = open('velocity.log', 'a')
            foo.write('%e \t %e \t %e \t %f \t %f \t %f\n' % (x, y, z, v_out[0], v_out[1], v_out[2]))
            foo.close()

        return v_out

    def getVelocity2(self, x, y, z, sph=False, unit_convert=True):
        """
        new method to interpolate the velocity
        cs: effecitve sound speed in km/s;
        age: the time since the collapse began in year.
        """

        (r_in, t_in, p_in) = self.Cart2Spherical(x, y, z, unit_convert=unit_convert)

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

        # r, t = 10*au, np.radians(30.)
        # print(r, t)
        r_corners = np.argsort(abs(r_in-self.xr*self.r_inf))[:2]
        theta_corners = np.argsort(abs(t_in-self.theta))[:2]

        # print(r_corners, theta_corners)

        # initialize the velocity vector in spherical coordinates
        # TODO: use scipy interp2d
        v_sph = []
        for k in ['vr2d', 'vtheta2d', 'vphi2d']:
            f = interp2d(self.xr[r_corners]*self.r_inf,
                         self.theta[theta_corners],
                         self.tsc2d[k][np.ix_(r_corners, theta_corners)])
            v_sph.append(float(f(r_in, t_in)/1e2))

        # v_r, v_theta, v_phi = 0.0, 0.0, 0.0
        # for rc in r_corners:
        #     for tc in theta_corners:
        #         v_r += self.vr2d[rc, tc]
        #         v_theta += self.vtheta2d[rc, tc]
        #         v_phi += self.vphi2d[rc, tc]
        # v_r = v_r/4
        # v_theta = v_theta/4
        # v_phi = v_phi/4
        #
        # v_sph = list(map(float, [v_r/1e2, v_theta/1e2, v_phi/1e2]))  # convert to SI unit (meter)
        if sph:
            return v_sph

        v_out = self.Spherical2Cart_vector((r_in, t_in, p_in), v_sph)

        if self.debug:
            foo = open('velocity.log', 'a')
            foo.write('%e \t %e \t %e \t %f \t %f \t %f\n' % (x, y, z, v_out[0], v_out[1], v_out[2]))
            foo.close()

        return v_out

    def getAbundance(self, x, y, z, config, tol=10):
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

        # single negative drop case
        # TODO: adopt a more generic model name, but keep backward compatability.
        if (config['a_model'] == 'neg_step1') or (config['a_model'] == 'step1'):
            a0 = float(config['a_params0'])
            a1 = float(config['a_params1'])
            a2 = float(config['a_params2'])

            if (r_in - a2*self.r_inf) > tol/2:
                abundance = a0
            elif abs(r_in - a2*self.r_inf) <= tol/2:
                abundance = a0*a1 + (r_in-(a2*self.r_inf-tol/2))*(a0-a0*a1)/tol
            else:
                abundance = a0*a1

        elif (config['a_model'] == 'neg_step2') or (config['a_model'] == 'step2'):
            a0 = float(config['a_params0'])
            a1 = float(config['a_params1'])
            a2 = float(config['a_params2'])
            a3 = float(config['a_params3'])
            a4 = float(config['a_params4'])

            if (r_in - a2*self.r_inf) > tol/2:
                abundance = a0
            # linear interpolation from the outer region to the first step
            elif abs(r_in - a2*self.r_inf) <= tol/2:
                abundance = a0*a1 + (r_in-(a2*self.r_inf-tol/2))*(a0-a0*a1)/tol
            # first step
            elif (r_in - a4*au_cgs) > tol/5/2 and (a2*self.r_inf - r_in) > tol/2:
                abundance = a0*a1
            # linear interpolation from the first step to the second step
            elif abs(r_in - a4*au_cgs) <= tol/5/2:
                abundance = a0*a3 + (r_in-(a4*au_cgs-tol/5/2))*(a0*a1-a0*a3)/(tol/5)
            else:
                abundance = a0*a3

        elif (config['a_model'] == 'drop'):
            a0 = float(config['a_params0'])
            a1 = float(config['a_params1'])
            a2 = float(config['a_params2'])
            a3 = float(config['a_params3'])
            a4 = float(config['a_params4'])

            if (r_in - a2*au_cgs) > tol/2:
                abundance = a0
            # linear interpolation from the outer region to the first step
            elif abs(r_in - a2*au_cgs) <= tol/2:
                abundance = a1 + (r_in-(a2*au_cgs-tol/2))*(a0-a1)/tol
            # first step
            elif (r_in - a4*au_cgs) > tol/5/2 and (a2*au_cgs - r_in) > tol/2:
                abundance = a1
            # linear interpolation from the first step to the second step
            elif abs(r_in - a4*au_cgs) <= tol/5/2:
                abundance = a3 + (r_in-(a4*au_cgs-tol/5/2))*(a1-a3)/(tol/5)
            else:
                abundance = a3

        elif (config['a_model'] == 'drop2'):
            a0 = float(config['a_params0'])
            a1 = float(config['a_params1'])
            a2 = float(config['a_params2'])
            a3 = float(config['a_params3'])
            a4 = float(config['a_params4'])

            if (r_in - a2*au_cgs) > tol/2:
                abundance = a0
            # linear interpolation from the outer region to the first step
            elif abs(r_in - a2*au_cgs) <= tol/2:
                abundance = a1 + (r_in-(a2*au_cgs-tol/2))*(a0-a1)/tol
            # first step
            elif (r_in - a4*au_cgs) > tol/5/2 and (a2*au_cgs - r_in) > tol/2:
                abundance = a1
            # linear interpolation from the first step to the second step
            elif abs(r_in - a4*au_cgs) <= tol/5/2:
                abundance = a3 + (r_in-(a4*au_cgs-tol/5/2))*(a1-a3)/(tol/5)
            elif r_in >= 13*au_cgs:
                abundance = a3
            else:
                abundance = 0.0

        elif (config['a_model'] == 'drop3'):
            a0 = float(config['a_params0'])  # undelepted abundance
            a1 = float(config['a_params1'])  # depleted abundance
            a2 = float(config['a_params2'])  # evaporation temperature (K)
            a3 = float(config['a_params3'])  # depletion density (cm-3)
            a4 = float(config['a_params4'])  # the temperature H2O starts to destory HCO+
            if a4 == -1:
                a4 = np.inf

            temp = self.getTemperature(x, y, z)
            density = self.getDensity(x, y, z)/1e6

            if (temp <= a2) and (density >= a3):
                abundance = a1
            elif (temp <= a4):
                abundance = a0
            else:
                abundance = 0.0


        elif config['a_model'] == 'uniform':
            abundance = float(config['a_params0'])

        elif config['a_model'] == 'lognorm':
            a0 = float(config['a_params0'])
            a1 = float(config['a_params1'])
            a2 = float(config['a_params2'])
            a3 = float(config['a_params3']) # r_in for power law decrease

            if r_in >= a2*self.r_inf:
                abundance = a0
            elif (r_in < a2*self.r_inf) & (r_in > a3*au_cgs):
                abundance = a0*a1+a0*(1-a1)/(np.log10(self.r_inf*a2)-np.log10(a3*au_cgs))*(np.log10(r_in)-np.log10(a3*au_cgs))
            else:
                abundance = a0*a1

        elif config['a_model'] == 'powerlaw':
            a0 = float(config['a_params0'])
            a1 = float(config['a_params1'])
            a2 = float(config['a_params2'])
            a3 = float(config['a_params3'])
            a4 = float(config['a_params4'])

            # re-define rMin
            # rmin = 100*au_cgs
            rmin = self.rmin

            if r_in >= a2*self.r_inf:
                abundance = a0
            elif (r_in >= rmin) and (r_in < a2*self.r_inf):
                # y = Ax^a3+B
                A = a0*(1-a1)/((a2*self.r_inf)**a3 - rmin**a3)
                B = a0-a0*(1-a1)*(a2*self.r_inf)**a3/((a2*self.r_inf)**a3 - rmin**a3)
                abundance = A*r_in**a3+B
            else:
                abundance = a0*a1

            # option to cap the maximum value of abundance
            if a4 > 0:
                if abundance > abs(a4):
                    abundance = abs(a4)

        elif config['a_model'] == 'powerlaw2':
            a0 = float(config['a_params0'])
            a1 = float(config['a_params1'])
            a2 = float(config['a_params2'])
            a3 = float(config['a_params3'])
            a4 = float(config['a_params4'])

            # re-define rMin
            # rmin = 100*au_cgs
            rmin = self.rmin

            if r_in >= a2*self.r_inf:
                abundance = a0
            elif (r_in >= rmin) and (r_in < a2*self.r_inf):
                # y = Ax^a3+B
                A = a0*(1-a1)/((a2*self.r_inf)**a3 - rmin**a3)
                B = a0-a0*(1-a1)*(a2*self.r_inf)**a3/((a2*self.r_inf)**a3 - rmin**a3)
                abundance = A*r_in**a3+B
            else:
                abundance = a0*a1

            # add the evaporation zone
            # TODO: parametrize the setup
            if (r_in <= 100*au_cgs) and (r_in >= 13*au_cgs):
                abundance = 1e-10

            # option to cap the maximum value of abundance
            if a4 > 0:
                if abundance > abs(a4):
                    abundance = abs(a4)

        elif config['a_model'] == 'chem':
            a0 = float(config['a_params0'])  # peak abundance
            a1 = float(config['a_params1'])  # inner abundace
            a2 = float(config['a_params2'])  # peak radius
            a3 = float(config['a_params3'])  # inner decrease power
            a4 = float(config['a_params4'])  # outer decrease power

            # radius of the evaporation front, determined by the extent of COM emission
            rCOM = 100*au_cgs

            if r_in >= a2*self.r_inf:
                # y = Ax^a, a < 0
                A_out = a0 / (a2*self.r_inf)**a4
                abundance = A_out * r_in**a4
            elif (r_in >= rCOM) and (r_in < a2*self.r_inf):
                # y = Ax^a, a > 0
                A_in = a0 / (a2*self.r_inf)**a3
                abundance = A_in * r_in**a3
            else:
                abundance = a1

        elif config['a_model'] == 'chem2':
            a0 = float(config['a_params0'])  # peak abundance
            a1 = float(config['a_params1'])  # inner abundace
            a2 = float(config['a_params2'])  # inner peak radius [AU]
            a3 = float(config['a_params3'])  # outer peak radius [AU]
            a4 = config['a_params4']  # inner/outer radius of the evaporation region

            # radius of the evaporation front, determined by the extent of COM emission
            if (a4 == '-1') or (a4 == '2.0/-2.0'):  # for backward compatability
                rCOM = 100*au_cgs
                rCen = 13*au_cgs
            else:
                rCen = float(a4.split(',')[0])*au_cgs
                rCOM = float(a4.split(',')[1])*au_cgs

            # innerExpo, outerExpo = [float(i) for i in config['a_params4'].split('/')]
            # fix the decreasing/increasing powers
            innerExpo = 2.0
            outerExpo = -2.0

            if r_in >= a3*au_cgs:
                # y = Ax^a, a < 0
                A_out = a0 / (a3*au_cgs)**outerExpo
                abundance = A_out * r_in**outerExpo
            elif (r_in < a3*au_cgs) and (r_in >= a2*au_cgs):
                abundance = a0
            elif (r_in >= rCOM) and (r_in < a2*au_cgs):
                # y = Ax^a, a > 0
                A_in = a0 / (a2*au_cgs)**innerExpo
                abundance = A_in * r_in**innerExpo
            elif (r_in >= rCen) and (r_in < rCOM):  # centrifugal radius
                abundance = a1
            else:
                abundance = 1e-20

        elif config['a_model'] == 'chem3':
            a0 = float(config['a_params0'])  # peak abundance
            a1 = float(config['a_params1'])  # inner abundace
            a2 = list(map(float, config['a_params2'].split(',')))  # inner/outer radius for the maximum abundance [AU]
            a3 = list(map(float, config['a_params3'].split(',')))  # inner/outer radius for the evaporation zone [AU]
            a4 = list(map(float, config['a_params4'].split(',')))  # inner/outer decreasing power
            # radius of the evaporation front, determined by the extent of COM emission
            rEvap_inner = a3[0]*au_cgs
            rEvap_outer = a3[1]*au_cgs

            # innerExpo, outerExpo = [float(i) for i in config['a_params4'].split('/')]
            # fix the decreasing/increasing powers
            innerExpo = a4[0]
            outerExpo = a4[1]

            if r_in >= a2[1]*au_cgs:
                # y = Ax^a, a < 0
                A_out = a0 / (a2[1]*au_cgs)**outerExpo
                abundance = A_out * r_in**outerExpo
            elif (r_in < a2[1]*au_cgs) and (r_in >= a2[0]*au_cgs):
                abundance = a0
            elif (r_in >= rEvap_outer) and (r_in < a2[0]*au_cgs):
                # y = Ax^a, a > 0
                A_in = a0 / (a2[0]*au_cgs)**innerExpo
                abundance = A_in * r_in**innerExpo
            elif (r_in >= rEvap_inner) and (r_in < rEvap_outer):  # centrifugal radius
                abundance = a1
            else:
                abundance = 1e-20

        if self.debug:
            foo = open('abundance.log', 'a')
            foo.write('%e \t %e \t %e \t %f\n' % (x, y, z, abundance))
            foo.close()

        # uniform abundance
        # abundance = 3.5e-9

        return float(abundance)

    def radialSmoothing(self, x, y, z, variable, kernel='boxcar',
                        smooth_factor=2, config=None):
        # convert the coordinates from Cartian to spherical
        (r_in, t_in, p_in) = self.Cart2Spherical(x, y, z)

        # r-array for smoothing
        smoothL = r_in/smooth_factor*au_cgs
        r_arr = np.arange(r_in-smoothL/2, r_in+smoothL/2, smoothL/50)  # 50 bins

        # setup the smoothing kernel
        # it is not really a smoothing kernel, more like a local mean
        def averageKernel(kernel, r, var):
            if kernel == 'boxcar':
                out = np.mean(var)
            return out

        # run the corresponding look-up function for the desired variable

        var_arr = np.empty_like(r_arr)
        for i, r in enumerate(r_arr):
            (xd, yd, zd) = self.Spherical2Cart(r, t_in, p_in)
            if variable == 'abundance':
                var_arr[i] = self.getAbundance(xd/1e2, yd/1e2, zd/1e2, config)
        var = averageKernel(kernel, r, var_arr)

        return float(var)
