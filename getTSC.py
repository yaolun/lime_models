def getTSC(age, cs, omega, velfile='none', max_rCell=0.001, TSC_dir='', outdir='', outname='', overwrite=False, **kwargs):
    import numpy as np
    import os
    import astropy.constants as const
    from scipy.interpolate import interp1d
    import h5py
    au = const.au.cgs.value
    yr = 3600.*24*365

    if (not os.path.exists(velfile)) or overwrite or (velfile == 'none'):
        if (len(TSC_dir) == 0) or (len(outdir) == 0):
            print('Paths to directories are not set completely.  Please revise... ')
            print('TSC_dir = {:<s}'.format(TSC_dir))
            print('outdir = {:<s}'.format(outdir))
            return False
        # run Fortran-TSC code
        print('Using the fortran binary to calculate the TSC model.')

        # write out the TSC parameter file
        tscpar = open(TSC_dir+'tsc.par', 'w')
        """
        vmin, vmax, delv, # of inclination, inclinatio, tau (=Omega * age)
        xrmin, xrmax, delxr, ntmax, npmax
        The inclination is for calculating the line-of-sight quantities, which is not used here.  Thus, set to 90 degree.
        """
        nr = 300
        ntheta = 400
        nphi = 50
        tscpar.write('0.1 3.0 0.1 1 90 %.3f\n' % (omega*age*yr) )
        tscpar.write('0.0001 1.0 0.0001 %d %d' % (ntheta/2, nphi) )
        tscpar.close()

        os.chdir(TSC_dir)
        os.system(TSC_dir+'ncofrac_update')

        # read in the TSC output
        tsc2d_coarse = loadTSC(TSC_dir+'rho_v_env', age, cs, omega, **kwargs)

        tscpar = open(TSC_dir+'tsc.par', 'w')
        tscpar.write('0.1 3.0 0.1 1 90 %.3f\n' % (omega*age*yr) )
        tscpar.write('0.00001 0.01 0.00001 %d %d' % (ntheta/2, nphi) )
        tscpar.close()

        os.chdir(TSC_dir)
        os.system(TSC_dir+'ncofrac_update')

        # read in the TSC output
        tsc2d_fine = loadTSC(TSC_dir+'rho_v_env', age, cs, omega, **kwargs)

        # reduce the total file size and interpolate onto a log-grid
        # create the log-linear grid cap at 0.01 for the reduced radius
        r_in = 0.1*au/(cs*1e5*age*yr)
        ri           = r_in * (1.0/r_in)**(np.arange(nr+1).astype(dtype='float')/float(nr))
        ri           = np.hstack((0.0, ri))
        # Keep the constant cell size in r-direction at large radii
        ri_cellsize = ri[1:-1]-ri[0:-2]
        ind = np.where(ri_cellsize > max_rCell)[0][0]       # The largest cell size is 100 AU
        ri = np.hstack((ri[0:ind], ri[ind]+np.arange(np.ceil((1.0-ri[ind])/max_rCell))*max_rCell, 1.0))
        rc = (ri[1:]+ri[:-1])/2

        thetac = tsc2d_fine['thetac']

        vr2d = np.empty((len(rc), len(thetac)))
        vtheta2d = np.empty((len(rc), len(thetac)))
        vphi2d = np.empty((len(rc), len(thetac)))
        rho2d = np.empty((len(rc), len(thetac)))

        tsc_theta_wall = np.hstack(([2*tsc2d_fine['thetac'][0]-tsc2d_fine['thetac'][1]],
                                    (tsc2d_fine['thetac'][:-1]+tsc2d_fine['thetac'][1:])/2,
                                    [2*tsc2d_fine['thetac'][-1]-tsc2d_fine['thetac'][-2]]))

        tsc_xr = np.hstack((tsc2d_fine['xrc'], tsc2d_coarse['xrc']))
        # interpolate the tsc kinematics onto the log-linear grid
        for it, t in enumerate(thetac):
            vr = np.hstack((tsc2d_fine['vr2d'][:,it], tsc2d_coarse['vr2d'][:,it]))
            vtheta = np.hstack((tsc2d_fine['vtheta2d'][:,it], tsc2d_coarse['vtheta2d'][:,it]))
            vphi = np.hstack((tsc2d_fine['vphi2d'][:,it], tsc2d_coarse['vphi2d'][:,it]))
            rho = np.hstack((tsc2d_fine['rho2d'][:,it], tsc2d_coarse['rho2d'][:,it]))
            f_vr = interp1d(tsc_xr, vr)
            f_vtheta = interp1d(tsc_xr, vtheta)
            f_vphi = interp1d(tsc_xr, vphi)
            f_rho = interp1d(tsc_xr, rho)

            for ir, r in enumerate(rc):
                vr2d[ir, it] = f_vr(r)
                vtheta2d[ir, it] = f_vtheta(r)
                vphi2d[ir, it] = f_vphi(r)
                rho2d[ir, it] = f_rho(r)

        tsc2d = {'vr2d': vr2d, 'vtheta2d':vtheta2d, 'vphi2d':vphi2d, 'rho2d':rho2d,
                 'xrc':rc, 'thetac': thetac, 'xr_wall': ri, 'theta_wall': tsc_theta_wall}

        # Saving the objects:
        with h5py.File(outdir+outname+'.h5', 'w') as f:
            f.attrs['age'] = age
            f.attrs['cs'] = cs
            f.attrs['omega'] = omega
            for k in tsc2d.keys():
                f.create_dataset(k, data=tsc2d[k])
        # with open(outdir+outname+'.pkl', 'wb') as f:
        #     pickle.dump(tsc2d, f)
    else:
        try:
            tsc2d = {}
            tsc2d_keys = ['vr2d', 'vtheta2d', 'vphi2d', 'rho2d',
                          'xrc', 'thetac', 'xr_wall', 'theta_wall']
            with h5py.File(velfile, 'r') as f:
                for k in tsc2d_keys:
                    if '2d' in k:
                        tsc2d[k] = f[k][:,:]
                    else:
                        tsc2d[k] = f[k][:]
            # with open(velfile, 'rb') as f:
            #     tsc2d = pickle.load(f)
        except:
            print(velfile)
            tsc2d = loadTSC(velfile, age, cs, omega, **kwargs)

    return tsc2d

def loadTSC(velfile, age, cs, omega, fix_tsc=True, hybrid_tsc=False):
    import numpy as np
    import pandas as pd
    import astropy.constants as const
    G = const.G.cgs.value
    mh = const.m_p.cgs.value + const.m_e.cgs.value
    mmw = 2.37
    yr = 3600.*24*365
    # ascii.read() fails for large file.  Use pandas instead
    tsc = pd.read_csv(velfile, skiprows=1, delim_whitespace=True, header=None)
    tsc.columns = ['lp', 'xr', 'theta', 'ro', 'ur', 'utheta', 'uphi']

    xr = np.unique(tsc['xr'])  # reduce radius: x = r/(a*t) = r/r_inf
    xr_wall = np.hstack(([2*xr[0]-xr[1]],
                          (xr[:-1]+xr[1:])/2,
                          [2*xr[-1]-xr[-2]]))
    theta = np.unique(tsc['theta'])
    theta_wall = np.hstack(([2*theta[0]-theta[1]],
                            (theta[:-1]+theta[1:])/2,
                            [2*theta[-1]-theta[-2]]))
    nxr = len(xr)
    ntheta = len(theta)

    # the output of TSC fortran binary is in mass density
    rho2d = 1/(4*np.pi*G*(age*yr)**2)/mh/mmw * np.array(tsc['ro']).reshape([nxr, ntheta])

    # in unit of km/s
    vr2d = np.reshape(tsc['ur'].to_numpy(), (nxr, ntheta)) * np.float64(cs)
    vtheta2d = np.reshape(tsc['utheta'].to_numpy(), (nxr, ntheta)) * np.float64(cs)
    vphi2d = np.reshape(tsc['uphi'].to_numpy(), (nxr, ntheta)) * np.float64(cs)

    if fix_tsc:
        # fix the discontinuity in v_r
        # vr = vr + offset * log(xr)/log(xr_break)  for xr >= xr_break
        for i in range(ntheta):
            dvr = abs((vr2d[1:,i] - vr2d[:-1,i])/vr2d[1:,i])
            break_pt = xr[1:][(dvr > 0.05) & (xr[1:] > 1e-3) & (xr[1:] < 1-5e-3)]
            if len(break_pt) > 0:
                offset = vr2d[(xr < break_pt),i].max() - vr2d[(xr > break_pt),i].min()
                vr2d[(xr >= break_pt),i] = vr2d[(xr >= break_pt),i] + offset*np.log10(xr[xr >= break_pt])/np.log10(break_pt)
        # fix the discontinuity in v_phi
        for i in range(ntheta):
            dvr = abs((vphi2d[1:,i] - vphi2d[:-1,i])/vphi2d[1:,i])
            break_pt = xr[1:][(dvr > 0.1) & (xr[1:] > 1e-3) & (xr[1:] < 1-2e-3)]
            if len(break_pt) > 0:
                offset = vphi2d[(xr < break_pt),i].min() - vphi2d[(xr > break_pt),i].max()
                vphi2d[(xr >= break_pt),i] = vphi2d[(xr >= break_pt),i] + offset*np.log10(xr[xr >= break_pt])/np.log10(break_pt)

    # hybrid TSC kinematics that switches to angular momentum conservation within the centrifugal radius
    if hybrid_tsc:
        from scipy.interpolate import interp1d
        for i in range(ntheta):
            rCR = omega**2 * G**3 * (0.975*(cs*1e5)**3/G*(age*3600*24*365))**3 * np.sin(theta[i])**4 / (16*(cs*1e5)**8)
            if rCR/r_inf >= xr.min():
                f_vr = interp1d(xr, vr2d[:,i])
                vr_rCR = f_vr(rCR/r_inf)
                f_vphi = interp1d(xr, vphi2d[:,i])
                vphi_rCR = f_vphi(rCR/r_inf)

                # radius in cylinderical coordinates
                wCR = np.sin(theta[i]) * rCR
                J = vphi_rCR * wCR
                M = (vr_rCR**2 + vphi_rCR**2) * wCR / (2*G)

                w = xr*np.sin(theta[i])*r_inf
                vr2d[(xr <= rCR/r_inf), i] = -( 2*G*M/w[xr <= rCR/r_inf] - J**2/(w[xr <= rCR/r_inf])**2 )**0.5
                vphi2d[(xr <= rCR/r_inf), i] = J/(w[xr <= rCR/r_inf])

    tsc2d = {'vr2d': vr2d, 'vtheta2d': vtheta2d, 'vphi2d': vphi2d, 'rho2d': rho2d,
             'xrc':xr, 'thetac':theta}

    return tsc2d
