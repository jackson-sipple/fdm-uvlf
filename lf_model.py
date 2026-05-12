import utils
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from dust_correction import DustCorrector
from collections import defaultdict
import scipy
from astropy import constants as const, units as u
import warnings
import sys
sys.path.append('../')
import mass_function


class LFMeasurements:
    def __init__(self, fn, dc=True, **shift_args):
        self.dc = dc
        self.fn = fn
        self.points = dict(np.load(fn))
        if dc:
            self.points = DustCorrector().dodc(self.points)
        self.z_vals = self.points['z_vals']
        self.mags = self.points['mags']
        self.phis = np.array(self.points['phis'])
        self.sig_minuses = np.array(self.points['sig_minuses'])
        self.sig_pluses = np.array(self.points['sig_pluses'])
        self.sigmas = 2*self.sig_pluses*self.sig_minuses / (self.sig_pluses+self.sig_minuses)
        self.sigma_primes = (self.sig_pluses-self.sig_minuses) / (self.sig_pluses+self.sig_minuses)
        self.sources = self.points['sources']
        self.unique_z = np.unique(self.z_vals)
        self.n_points = len(self.mags)
        if len(shift_args) > 0 and not all(val is None for val in shift_args.values()):
            self.shift_all(**shift_args)
        self.mags_and_zs = list(zip(self.mags, self.z_vals))
        self.zipped = list(zip(self.z_vals, self.mags, self.phis,
                               self.sig_minuses, self.sig_pluses, self.sources))

    def shift_all(self, biases, sources, redshift_bins=None):
        if redshift_bins is None:
            redshift_bins = len(biases)*[[None]]
        for bias, source, redshifts in zip(biases, sources, redshift_bins):
            for i in range(self.n_points):
                if self.sources[i] == source:
                    if redshifts is None or self.z_vals[i] in redshifts:
                        # shift upper error bar and keep the rest consistent with zero
                        if self.phis[i] == 0:
                            self.sig_pluses[i] += bias*self.sig_pluses[i]
                        else:
                            amt = bias*self.phis[i]
                            # phi is at least zero, lower error bar is at most phi
                            self.phis[i] = max(self.phis[i] + amt, 0)
                            self.sig_minuses[i] = min(
                                self.sig_minuses[i], self.phis[i])
                        self.sources[i] += ' shifted'
        np.savez('test_shift.npz', **dict(self.points))
        

    def shift_B22(self, sgn=1):
        for i in range(self.n_points):
            if self.sources[i] == 'B22':
                bias = 0.2
                if self.z_vals[i] == 5:
                    bias = 0.22
                bias = bias * sgn
                # shift upper error bar and keep the rest consistent with zero
                if self.phis[i] == 0:
                    self.sig_pluses[i] += bias*self.sig_pluses[i]
                else:
                    amt = bias*self.phis[i]
                    # phi is at least zero, lower error bar is at most phi
                    self.phis[i] = max(self.phis[i] + amt, 0)
                    self.sig_minuses[i] = min(
                        self.sig_minuses[i], self.phis[i])
                self.sources[i] += ' shifted'

    #ONLY VALID FOR GIVING LOWER LIMITS (UPPER LIMIT MEASURMENTS EXCLUDED)
    def phi_tot_loglike(self, z, tot, non_neg=True, use_minimize=True, method='trust-constr', minimize_maxiter=10000, eps=0):
        mask = (self.z_vals == z) & (self.phis != 0)
        plus_z = self.sig_pluses[mask]
        minus_z = self.sig_minuses[mask]
        phis_z = self.phis[mask]
        x_is = phis_z + self.deltas_given_u_asym(z, u=tot-np.sum(phis_z), non_neg=non_neg, use_minimize=use_minimize, method=method, minimize_maxiter=minimize_maxiter, eps=eps)
        return np.sum(utils.log_like(xhat=phis_z, x=x_is, sigma_plus=plus_z, sigma_minus=minus_z))

    def phi_tot_012_sigma(self, z, naive=False, also3=False):
        mask = (self.z_vals == z) & (self.phis != 0)
        sigmas = self.sigmas[mask]
        sigma_primes = self.sigma_primes[mask]
        plus_z = self.sig_pluses[mask]
        minus_z = self.sig_minuses[mask]
        phis_z = self.phis[mask]
        x_hat = np.sum(phis_z)
        if naive:
            if also3:
                return x_hat, x_hat - np.sqrt(np.sum(minus_z)), x_hat - 2*np.sqrt(np.sum(minus_z)), x_hat - 3*np.sqrt(np.sum(minus_z))
            return x_hat, x_hat - np.sqrt(np.sum(minus_z)), x_hat - 2*np.sqrt(np.sum(minus_z))
        min_range = x_hat-2.5*np.sqrt(np.sum(minus_z**2))
        print(min_range)
        min_range = max(min_range, 0)
        max_range = x_hat
        tots = np.linspace(min_range, max_range, 10)
        loglikes = [self.phi_tot_loglike(z, tot) for tot in tots]
        tots_from_loglikes = interp1d(loglikes, tots, fill_value='extrapolate')
        print(loglikes)
        print(x_hat)
        if also3:
            return tots_from_loglikes(0), tots_from_loglikes(-0.5), tots_from_loglikes(-2), tots_from_loglikes(-9/2)
        return tots_from_loglikes(0), tots_from_loglikes(-0.5), tots_from_loglikes(-2)


    def deltas_given_u_asym(self, z, u, max_iter=1000, rtol=1e-6, non_neg=True, use_minimize=True, method='trust-constr', minimize_maxiter=10000, eps=0):
        mask = (self.z_vals == z) & (self.phis != 0)
        sigmas = self.sigmas[mask]
        sigma_primes = self.sigma_primes[mask]
        plus_z = self.sig_pluses[mask]
        minus_z = self.sig_minuses[mask]
        phis_z = self.phis[mask]
        atol = rtol * 3*(plus_z+minus_z)
        deltas = np.zeros_like(sigmas)
        dif_between_iters = np.inf * np.ones_like(deltas)
        def score(xval):
            return np.sum(utils.log_like(xhat=phis_z, x=xval, sigma_plus=plus_z, sigma_minus=minus_z))
        i = 0
        while np.any(dif_between_iters > atol) and i < max_iter:  
            term1 =  (sigmas + sigma_primes*deltas)**3 / sigmas
            term2 = u / np.sum((sigmas + sigma_primes*deltas)**3 / sigmas)
            new_deltas = term1 * term2
            dif_between_iters = np.abs(deltas-new_deltas)
            deltas = new_deltas
            i += 1
        if i == max_iter:
            deltas = np.zeros_like(sigmas)
        def equations(deltas):
            term1 =  (sigmas + sigma_primes*deltas)**3 / sigmas
            term2 = u / np.sum((sigmas + sigma_primes*deltas)**3 / sigmas)
            return term1 * term2 - deltas
        def jacobian(deltas):
            A = np.sum((sigmas + sigma_primes*deltas)**3 / sigmas)
            B = 3*(sigmas + sigma_primes*deltas)**2*sigma_primes / sigmas
            jacobian = np.outer(-((sigmas+sigma_primes*deltas)**3/sigmas), u*B/A**2)
            diag_term = u*B/A - 1
            np.fill_diagonal(jacobian, jacobian.diagonal()+diag_term)
            return jacobian
        fsolve_deltas, infodict, ier, message = scipy.optimize.fsolve(equations, deltas, fprime=jacobian, full_output=1, xtol=1e-12)
        # if ier != 1:
        #     print(message)
        # else:
        deltas = fsolve_deltas
        def neg_scaled_score(xval, scale=1):
            return -np.sum(utils.log_like(xhat=phis_z*scale, x=xval*scale, sigma_plus=plus_z*scale, sigma_minus=minus_z*scale))
        if use_minimize:
            #deltas = np.zeros_like(sigmas)
            bounds = None
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - u - np.sum(phis_z)})
            if non_neg:
                #bounds = scipy.optimize.Bounds(lb=0*np.ones_like(phis_z), ub=np.inf*np.ones_like(phis_z), keep_feasible=True)
                constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - u - np.sum(phis_z)}, {'type': 'ineq', 'fun': lambda x: x})
            x0 = np.clip(phis_z+deltas,0,np.inf)
            #x0 = np.zeros_like(deltas)
            #x0 = phis_z
            #x0 = np.clip(phis_z + u/len(phis_z), eps, np.inf)
            #print(x0)
            jac = None
            # if non_neg:
            #     bounds = [[0,np.inf], [0,np.inf], [0,np.inf], [0,np.inf], [0,np.inf]]
            #     # THANK YOU TO STACKOVERFLOW USER https://stackoverflow.com/questions/52208363/scipy-minimize-violates-given-bounds
            #     def gradient_respecting_bounds(bounds, fun, eps=1e-8):
            #         """bounds: list of tuples (lower, upper)"""
            #         def gradient(x):
            #             fx = fun(x)
            #             grad = np.zeros(len(x))
            #             for k in range(len(x)):
            #                 d = np.zeros(len(x))
            #                 d[k] = eps if x[k] + eps <= bounds[k][1] else -eps
            #                 grad[k] = (fun(x + d) - fx) / d[k]
            #             return grad
            #         return gradient
            #     jac = gradient_respecting_bounds(bounds, neg_scaled_score)
            jac = 'cs'
            if method == 'SLSQP':
                optimum = scipy.optimize.minimize(fun=neg_scaled_score, x0=x0, bounds=bounds, jac=jac, constraints=constraints, method=method, tol=1e-8, options={'maxiter':minimize_maxiter})
            elif method == 'trust-constr':
                passed = False
                n_loops = 0
                while not passed:
                    optimum = scipy.optimize.minimize(fun=neg_scaled_score, x0=x0, bounds=bounds, jac=jac, constraints=constraints, method=method, tol=1e-8, options={'maxiter':minimize_maxiter,'gtol':1e-8,'xtol':1e-8,})
                    passed = optimum.success
                    x0 = optimum.x
                    if n_loops > 1:
                        print("LOOPED ENOUGH")
                        break
                    n_loops += 1
                # t = optimum
                # print('result', optimum.x)
                # deltas = optimum.x - phis_z
                # print(deltas)
                # print('result after sub and add', phis_z+deltas)
                # print('constr vals', t.constr, 'constr violation', t.constr_violation, 'constr penalty', t.constr_penalty, 'bar tol',t.barrier_tolerance, 'msg', t.message, 'status', t.status, 'cg stop cond', t.cg_stop_cond)
            if not optimum.success:
                print(optimum.message)
                if method == 'trust-constr':
                    t = optimum
                    #print('constr vals', t.constr, 'constr violation', t.constr_violation, 'constr penalty', t.constr_penalty, 'bar tol',t.barrier_tolerance, 'msg', t.message, 'status', t.status, 'cg stop cond', t.cg_stop_cond, '# func evals', t.nfev)
                deltas = optimum.x - phis_z
            else:
                #print(optimum.nit)
                deltas = optimum.x - phis_z
                #deltas = fsolve_deltas
        #print('tot', u+np.sum(phis_z), 'attempt', np.sum(phis_z+deltas), 'values', phis_z+deltas)
        #print(np.sum(utils.log_like(xhat=phis_z, x=phis_z+deltas, sigma_plus=plus_z, sigma_minus=minus_z)))
        return deltas


class LFModel:
    def __init__(self, meas_fn, params, dc=True, name=None, f_esc=0.2):
        self.meas_fn = meas_fn
        self.meas = LFMeasurements(self.meas_fn, dc=dc)
        self.name = name
        self.dc = dc
        self.dof = self.meas.n_points - len(params)
        self.sigma = lambda M: np.log(10)*0.16
        if not callable(f_esc):
            f_esc_constant = f_esc
            def f_esc(M): return f_esc_constant
        self.f_esc = f_esc
        self.params = params
        self.mass_fn = utils.load_mf()
        self.chi_dict = {}
        self.predictions = defaultdict(list)
        self.param_names = []
        self.xi_func = None
        self.n_dot_ion_func = None
        self.n_dot_ratio_func = None
        self.M_m = None

    def reinitialize(self):
        self.__init__(self.meas_fn, self.params, self.dc, self.name, self.f_esc)

    def shift_meas(self, biases, sources, redshift_bins=None):
        self.meas = LFMeasurements(
            self.meas_fn, dc=False, biases=biases, sources=sources, redshift_bins=redshift_bins)

    def dn_dM(self, M, z):
        ''' Halo mass function.'''
        return self.mass_fn[z](M)
    
    def dlogn_dlogM(self, M, z):
        log_M = np.log(M)
        del_log = 0.01 * np.log(M)
        x = [np.exp(log_M - del_log), np.exp(log_M+del_log)]
        y = [self.dn_dM(M_val, z) for M_val in x]
        return utils.log_derivative(x=x, y=y)[0]
    
    def L_c(self, M, z):
        ''' Median luminosity for a halo of mass M.'''
        pass

    def L_c_max(self, M, z):
        ''' L_c such that it implies SFE=1'''
        return self.M_dot_acc(M,z)/utils.KAPPA_UV

    def m_c(self, M, z):
        ''' L_c in m_UV.'''
        return utils.L_to_m(self.L_c(M, z))
    
    def m_mean(self, M, z):
        return utils.L_to_m(self.L_mean(M, z))

    def L_mean(self, M, z):
        ''' Mean luminosity of lognormal distribution in terms of median.'''
        return np.exp(self.sigma(M)**2/2) * self.L_c(M, z)
    
    #@utils.debug_function
    def M_from_m(self, m, z):
        if self.M_m is None:
            lo, hi = 8, 15
            M_vals = np.logspace(lo, hi, 1000)
            self.M_m = {}
            for redshift in np.arange(5,21):
                m_from_M = [self.m_mean(M, redshift) for M in M_vals]
                self.M_m[redshift] = utils.linlog_interp(m_from_M, M_vals)
        return self.M_m[z](m)

    def min_max_mass(self, z):
        ''' Convert faintest/brightest luminosities in measurements to masses.'''
        mags_and_zs = list(zip(self.meas.mags, self.meas.z_vals))
        mags = [mag for (mag, z_val) in mags_and_zs if z_val == z]
        if len(mags) == 0:
            return np.nan, np.nan
        mag1, mag2 = min(mags), max(mags)
        M_min, M_max = self.M_from_m(mag2, z), self.M_from_m(mag1, z)
        if not np.isfinite(M_max) or M_max < M_min or M_max > 1e15:
            M_max = 1e14
            warnings.warn(f"M_max undefined, setting to {M_max}", UserWarning)
        return M_min, M_max

    def phi_L_given_M(self, L, M, z):
        ''' Luminosity function conditioned on halo mass.'''
        prefactor = 1/(np.sqrt(2*np.pi)*self.sigma(M)*L)
        with np.errstate(divide='ignore'): # because sometimes L_c=0 ... warning is annoying but should correctly make exponent=-inf
            exponent = -np.log(L/self.L_c(M, z))**2 / (2*self.sigma(M)**2)
        return prefactor * np.exp(exponent)

    def phi_L(self, L, z):
        ''' Conditional luminosity function.'''
        lo, hi = 1e8, 1e15
        def integrand(M): return self.dn_dM(M, z) * self.phi_L_given_M(L, M, z)
        return utils.trapz_integrate(integrand, lo, hi, logspace=True)

    def phi_m(self, m, z):
        ''' Conditional luminosity function in m_UV'''
        L = utils.m_to_L(m)
        return utils.dL_dm(m) * self.phi_L(L, z)

    def M_dot_acc(self, M, z, delta=utils.M_DOT_ACC_DELTA, eta=2.5, to_Gyr=False):
        ''' Mass accretion rate in M_sol / yr (or / Gyr).'''
        retval = 3 * (M/1e10)**delta * ((1+z)/7)**eta  # In M_sun/yr
        if to_Gyr:
            retval *= 1e9
        return retval

    def sfr(self, M, z):
        ''' Star formation rate.'''
        L = self.L_mean(M, z)
        return utils.KAPPA_UV * L

    def f_star(self, M, z):
        ''' Star formation efficiency.'''
        return self.sfr(M, z) / self.M_dot_acc(M, z)
    
    def f_star_const(self, z, M_min=1e8):
        lo, hi = 1e8, 1e15
        def integrand(M): return self.dn_dM(M, z) * self.f_star(M, z)
        return utils.trapz_integrate(integrand, lo, hi, logspace=True)

    def n_dot_ion(self, z, observed_only=False, M_min=1e8, f_esc=None):
        ''' Ionization rate w/o reionizations'''
        if f_esc is None:
            f_esc = self.f_esc
        elif not callable(f_esc):
            f_esc_constant = f_esc
            def f_esc(M): return f_esc_constant
        if True:  # self.n_dot_ion_func is None:
            n_dot_ion_at_z = []
            z_vals = self.meas.unique_z
            for z_eval in z_vals:
                lo, hi = M_min, 1e15
                if observed_only:
                    lo, hi = self.min_max_mass(z_eval)
                prefactor = utils.A_HE * utils.F_GAMMA * \
                    (utils.OMEGA_M/utils.OMEGA_B) / utils.RHO_M.value

                def integrand(M):
                    return f_esc(M) * self.f_star(M, z_eval) * self.M_dot_acc(M, z_eval, to_Gyr=True) \
                        * self.dn_dM(M, z_eval)
                integral = utils.trapz_integrate(
                    integrand, lo, hi, logspace=True)
                n_dot_ion_at_z.append(prefactor * integral)
            self.n_dot_ion_func = utils.linlog_interp(
                z_vals, n_dot_ion_at_z)
        return self.n_dot_ion_func(z)

    def dn_dot_ion_dlnM(self, M, z):
        prefactor = utils.A_HE * utils.F_GAMMA * \
            (utils.OMEGA_M/utils.OMEGA_B) / utils.RHO_M.value
        return prefactor * M * self.f_esc(M) * self.dn_dM(M, z) * self.f_star(M, z) * self.M_dot_acc(M, z, to_Gyr=True)

    def dlnn_dot_ion_dlnM(self, M, z):
        return self.dn_dot_ion_dlnM(M, z) / self.n_dot_ion(z)

    def n_dot_ion_obs_vs_tot(self, z, M_min=1e8):
        ''' Ionization rate w/o reionizations'''
        if True:
            z_vals = self.meas.unique_z
            # will be zero at edge of bin
            z_vals = np.array([*z_vals, z_vals[-1]+0.5])
            n_dot_ratio_at_z = []
            for z_eval in z_vals:
                lo, hi = self.min_max_mass(z_eval)
                if np.isnan(lo) or np.isnan(hi):
                    n_dot_ratio_at_z.append(0)
                else:
                    prefactor = utils.A_HE * utils.F_GAMMA * \
                        (utils.OMEGA_M/utils.OMEGA_B) / utils.RHO_M.value

                    def integrand(M):
                        return self.f_esc(M) * self.f_star(M, z_eval) * self.M_dot_acc(M, z_eval, to_Gyr=True) \
                            * self.dn_dM(M, z_eval)
                    integral = utils.trapz_integrate(
                        integrand, lo, hi, logspace=True)
                    n_dot_ion_at_z = prefactor * integral
                    n_dot_ratio_at_z.append(
                        n_dot_ion_at_z / self.n_dot_ion(z_eval, M_min=M_min))
            self.n_dot_ratio_func = interp1d(
                z_vals, n_dot_ratio_at_z, fill_value=0, bounds_error=False)
        return self.n_dot_ratio_func(z)

    def big_n_dot_ion(self, z):
        retval_in_inv_Gyr = self.n_dot_ion(z)*utils.RHO_M.value*utils.OMEGA_B/utils.OMEGA_M \
            * (1/utils.M_H.to('M_sun').value) * u.Gyr**-1
        return retval_in_inv_Gyr.to('s^-1').value

    def t_rec(self, z, T_e=1e4):
        '''Recombination time.'''
        alpha_B = 2.6e-13 * (T_e/1e4)**0.76 * u.cm**3 * u.s**-1
        return (1/(alpha_B * utils.C_HII * (1+z)**3 * utils.N_H0)).to('Gyr').value

    def dxi_dz(self, z, x, observed_only=False, f_esc=None):
        '''Rate of change of ionization fraction.'''
        return utils.dt_dz(z) * (self.n_dot_ion(z, observed_only=observed_only, f_esc=f_esc) - x/self.t_rec(z))

    def xi(self, z, observed_only=False, f_esc=None):
        '''Ionization fraction.'''
        if self.xi_func is None:
            lo, hi = 0, 100

            def fun(z_prime, x, observed_only=observed_only):
                retval = self.dxi_dz(
                    z_prime, x[0], observed_only=observed_only, f_esc=f_esc)
                return -1 * retval  # i don't understand this part fully? -1?
            z_span = [hi, lo]
            z_eval = np.flip(np.linspace(lo, hi, 1000))
            x0 = [0]
            # have to do the reverse direction because initial condition must come first...
            xi_z_eval = solve_ivp(fun=fun, t_span=z_span,
                                  y0=x0, t_eval=z_eval).y[0]
            self.xi_func = utils.linlog_interp(
                np.flip(z_eval), np.clip(np.flip(xi_z_eval), 0, 1))

            #self.xi_func = lambda z: xi_z_eval
        return self.xi_func(z)
    
    def reionization_z(self, observed_only=False, f_esc=None):
        z_vals = np.linspace(0, 10, 1000)
        idx = np.where(np.array([self.xi(z, observed_only, f_esc) for z in z_vals]) < 1)[0][0]
        return z_vals[idx]

    def tau_e(self, z, f_esc=None):
        ''' Optical depth from electron scattering.'''
        numer = 3 * utils.H0 * utils.OMEGA_B * const.c * const.sigma_T
        denom = 8 * np.pi * const.G * const.m_p
        prefactor = (numer/denom).to('')  # Cancel out units
        def integrand(z, N_He):
            #N_He = 1 if z < 3 else 2
            num = self.xi(z, f_esc=f_esc) * (1+z)**2 + \
                (1-utils.Y_P) + N_He*utils.Y_P/4
            den = np.sqrt(utils.OMEGA_M * (1+z)**3 + utils.OMEGA_L)
            return num/den
        eps = np.finfo(float).eps  # loglog interpolation breaks at 0
        z = max(z, eps)
        integral = utils.trapz_integrate(
            lambda z: integrand(z, N_He=2), eps, min(z, 3))
        if z > 3:
            integral += utils.trapz_integrate(
                lambda z: integrand(z, N_He=1), 3, z)  # ,n_intervals=1000)
        return prefactor * integral

    def rho_UV(self, z, M_min=1e8, observed=False):
        M_max = 1e15
        rho_at_z = []
        for z_eval in self.meas.unique_z:
            if observed:
                M_min, M_max = self.min_max_mass(z_eval)
            def integrand(M): return self.dn_dM(
                M, z_eval) * self.L_mean(M, z_eval)
            rho_at_z.append(utils.trapz_integrate(
                integrand, M_min, M_max, logspace=True))
        return utils.linlog_interp(self.meas.unique_z, rho_at_z)(z)
    
    def epsilon(self, z, M_min=1e8):
        return self.rho_UV(z, M_min=M_min) * utils.KAPPA_UV / (8e-42)

    def epsilon_N(self, z):
        return 0.45 * self.big_n_dot_ion(z) * (const.h * utils.NU_H_ALPHA * u.Hz).to('erg').value

    def I_nu(self, z, M_min=1e8, nu_r=utils.NU_H_ALPHA):
        in_erg_s_Hz_Mpc2 = 1/(4 * np.pi * nu_r) * (const.c/utils.cosmo.H(z)).to('Mpc').value * self.epsilon(z, M_min=M_min) / (1+z)**2
        in_Jy = (in_erg_s_Hz_Mpc2 * u.erg * u.s**-1 * u.Hz**-1 * u.Mpc**-2).to('Jy').value
        return in_Jy
    
    def WF_z(self, M_min=1e8, observed=False, g_006=1):
        z_vals = np.linspace(5, 20, 1000)
        rho = np.array([self.rho_UV(z, M_min, observed) for z in z_vals])
        wf = np.array([utils.rho_UV_WF(z, g_006)[0] for z in z_vals])
        idx = np.where(rho-wf < 0)[0][0]
        return z_vals[idx]

    def chi_sq_of_model(self, at_each_point=False):
        if True: #if len(self.chi_dict) == 0:
            mags_and_zs = list(zip(self.meas.mags, self.meas.z_vals))
            phis_errs_and_zs = list(zip(
                self.meas.phis, self.meas.sig_minuses, self.meas.sig_pluses, self.meas.z_vals))
            for z in self.meas.unique_z:
                mags = [mag for (mag, z_val) in mags_and_zs if z_val == z]
                predicted_phis = [self.phi_m(mag, z) for mag in mags]
                phis = [phi for (phi, sm, sp, z_val)
                        in phis_errs_and_zs if z_val == z]
                sms = [sm for (phi, sm, sp, z_val)
                       in phis_errs_and_zs if z_val == z]
                sps = [sp for (phi, sm, sp, z_val)
                       in phis_errs_and_zs if z_val == z]
                chi_input = list(zip(phis, predicted_phis, sps, sms))
                self.predictions['phis'].extend(predicted_phis)
                self.predictions['mags'].extend(mags)
                self.predictions['z_vals'].extend([z]*len(mags))
                if at_each_point:
                    # Sort the array based on the sorted order of mags
                    chis_not_in_order = [utils.chi_squared(*args) for args in chi_input]
                    chis_by_sorted_mags = utils.sort_x_by_order_of_y(x=chis_not_in_order, y=mags)
                    self.chi_dict.update({z : chis_by_sorted_mags})
                else:
                    self.chi_dict.update(
                        {z: sum([utils.chi_squared(*args) for args in chi_input])})
        if at_each_point:
            return self.chi_dict
        return sum(self.chi_dict.values())

    def Delta_AIC(self, zero_point=None):
        k = len(self.params)
        n = self.meas.n_points
        if zero_point is None:
            return 2 * k + n * np.log(self.chi_sq_of_model()/n)
        else:
            return zero_point - 2 * k + n * np.log(self.chi_sq_of_model()/n)
    
    def red_chi_sq(self):
        return self.chi_sq_of_model()/self.dof

    def get_predictions(self):
        if len(self.predictions) == 0:
            self.chi_sq_of_model()
        return self.predictions
    
    def dM_dz(self, z, M, delta=utils.M_DOT_ACC_DELTA, eta=2.5, to_Gyr=True):
        return -self.M_dot_acc(M=M, z=z, delta=delta, to_Gyr=to_Gyr) * utils.dt_dz(z) * utils.OMEGA_M / utils.OMEGA_B
    
    def dM_star_dz(self, z, M, M_z):
        return self.dM_dz(z=z, M=M_z(z)[0]) * self.f_star(M=M_z(z)[0], z=z) * (utils.OMEGA_B/utils.OMEGA_M)
    
    def reversed_dM_star_dz(self, z, M, M_z):
        return -1*self.dM_star_dz(z, M, M_z)
    
    def M_of_z_func(self, M_init, z_init, M_min=1e8, delta=utils.M_DOT_ACC_DELTA):
        z_span = [z_init, z_init+50]
        func = lambda z, M: self.dM_dz(z=z, M=M, delta=delta)
        sol = solve_ivp(func, z_span, [M_init], dense_output=True)
        #print(sol.message)
        return sol.sol

    def M_star_of_z_func(self, M_init, z_init, M_min=1e8):
        z_min = self.z_of_M_min(M_init=M_init, z_init=z_init, M_min=M_min)
        M_z = self.M_of_z_func(M_init=M_init, z_init=z_init, M_min=M_min)
        z_span = [z_min, 5]
        func = lambda z, M: self.dM_star_dz(z=z, M=M, M_z=M_z)
        sol = solve_ivp(func, z_span, [0], dense_output=True, t_eval=np.linspace(*z_span, 1000))
        #print(sol.message)
        return sol.sol
    
    def z_of_M_min(self, M_init, z_init, M_min=1e8):
        M_func = self.M_of_z_func(M_init, z_init)
        z_vals = np.linspace(z_init, z_init+30, 10000)
        M_vals = np.array([M_func(z)[0] for z in z_vals])
        idx = np.argmin(np.abs(M_vals-M_min))
        return z_vals[idx]
    
class FiducialCLF(LFModel):
    N_PARAMS = 5
    MCMC_X0 = [1.24, 1, 1.5, -21.91, np.log10(1.2e12)] #B15
    MCMC_PRIOR = [[0.8, 2.5], [0.8, 2.5], [0, 2.5], [-25, -20], [10, 14]]

    def __init__(self, meas_fn, params, dc=True, name=None, f_esc=0.2, log_input=True):
        super().__init__(meas_fn, params, dc, name, f_esc)
        self.param_names = [r'$p$', r'$q$', r'$r$',
                            r'$M_{\rm UV,0}$', r'$\log_{10}(M_1/$M$_\odot)$']
        self.p, self.q, self.r, self.L0, self.M1 = params[:5]
        if log_input:
            self.L0 = utils.m_to_L(self.L0)
            self.M1 = 10**self.M1

    def to_log_input(self):
        return np.array([self.p, self.q, self.r, utils.L_to_m(self.L0), np.log10(self.M1)])

    def L_c(self, M, z):
        return self.L0 * (M/self.M1)**self.p/(1+(M/self.M1)**self.q) * ((1+z)/7)**self.r
    
class SharpkCLF(FiducialCLF):
    N_PARAMS = 5
    MCMC_X0 = [1.24, 1, 1.5, -21.91, np.log10(1.2e12)] #B15
    MCMC_PRIOR = [[0.8, 2.5], [0.8, 2.5], [0, 2.5], [-25, -20], [10, 14]]

    def __init__(self, meas_fn, params, dc=True, name=None, f_esc=0.2, log_input=True):
        super().__init__(meas_fn, params, dc, name, f_esc)
        self.mass_fn = utils.load_mf('../simplify/mass_fns_sharpk.npz')

class FlatteningCLF(LFModel):
    N_PARAMS = 6
    #MCMC_X0 = [1.24, 1, 1.5, -21.91, np.log10(1.2e12), np.log10(1e9)]
    MCMC_X0 = [1.99, 1.44, 1.08, -22.27, 11.55, 10.41]
    MCMC_PRIOR = [[0.8, 2.5], [0.8, 2.5], [0, 2.5], [-25, -20], [10, 14], [8, 14]]

    def __init__(self, meas_fn, params, dc=True, name=None, f_esc=0.2, log_input=True):
        super().__init__(meas_fn, params, dc, name, f_esc)
        self.param_names = [r'$p$', r'$q$', r'$r$',
                            r'$M_{\rm UV,0}$', r'$\log_{10}(M_1/$M$_\odot)$', r'$\log_{10}(M_{flat}/$M$_\odot)$']
        self.p, self.q, self.r, self.L0, self.M1, self.M_flat = params
        if log_input:
            self.L0 = utils.m_to_L(self.L0)
            self.M1 = 10**self.M1
            self.M_flat = 10**self.M_flat

    def to_log_input(self):
        return np.array([self.p, self.q, self.r, utils.L_to_m(self.L0), np.log10(self.M1), np.log10(self.M_flat)])

    def L_c(self, M, z):
        p = np.where(M > self.M_flat, self.p, utils.M_DOT_ACC_DELTA)
        # Normalization changes
        L0 = np.where(M > self.M_flat, self.L0, self.L0 *
                      (self.M_flat/self.M1)**(self.p-utils.M_DOT_ACC_DELTA))
        return L0 * (M/self.M1)**p/(1+(M/self.M1)**self.q) * ((1+z)/7)**self.r


class ShallowingCLF(LFModel):
    N_PARAMS = 7
    #MCMC_X0 = [1.24, 1, 1.5, -21.91, np.log10(1.2e12), 9, 1.24-utils.M_DOT_ACC_DELTA]
    MCMC_X0 = [2.09, 1.43, 1.12, -21.88, 11.43, 10.56, 0.27]
    MCMC_PRIOR = [[0.8, 2.5], [0.8, 2.5], [0, 2.5], [-25, -20], [10, 14], [8, 14], [-0.5, 1.5]]

    def __init__(self, meas_fn, params, dc=True, name=None, f_esc=0.2, log_input=True):
        super().__init__(meas_fn, params, dc, name, f_esc)
        self.param_names = [r'$p$', r'$q$', r'$r$',
                            r'$M_{\rm UV,0}$', r'$\log_{10}(M_1/$M$_\odot)$', r'$\log_{10}(M_2/$M$_\odot)$', r's']
        self.p, self.q, self.r, self.L0, self.M1, self.M2, self.s = params
        if log_input:
            self.L0 = utils.m_to_L(self.L0)
            self.M1 = 10**self.M1
            self.M2 = 10**self.M2

    def to_log_input(self):
        return np.array([self.p, self.q, self.r, utils.L_to_m(self.L0), np.log10(self.M1), np.log10(self.M2), self.s])

    def L_c(self, M, z):
        retval = self.L0 * (M/self.M1)**self.p/(1+(M/self.M1)**self.q) * ((1+z)/7)**self.r
        return np.where(M > self.M2, retval, retval * (M/self.M2)**(utils.M_DOT_ACC_DELTA + self.s - self.p))


class TruncatingCLF(LFModel):
    N_PARAMS = 6
    #MCMC_X0 = [1.24, 1, 1.5, -21.91, np.log10(1.2e12), np.log10(1e9)]
    MCMC_X0 = [1.69, 1.58, 1.12, -23.33, 11.93, 9]
    MCMC_PRIOR = [[0.8, 2.5], [0.8, 2.5], [0, 2.5], [-25, -20], [10, 14], [8, 14]]

    def __init__(self, meas_fn, params, dc=True, name=None, f_esc=0.2, log_input=True):
        super().__init__(meas_fn, params, dc, name, f_esc)
        self.param_names = [r'$p$', r'$q$', r'$r$',
                            r'$M_{\rm UV,0}$', r'$\log_{10}(M_1/$M$_\odot)$', r'$\log_{10}(M_{trunc}/$M$_\odot)$']
        self.p, self.q, self.r, self.L0, self.M1, self.M_trunc = params
        if log_input:
            self.L0 = utils.m_to_L(self.L0)
            self.M1 = 10**self.M1
            self.M_trunc = 10**self.M_trunc

    def to_log_input(self):
        return np.array([self.p, self.q, self.r, utils.L_to_m(self.L0), np.log10(self.M1), np.log10(self.M_trunc)])

    def L_c(self, M, z):
        return np.where(M > self.M_trunc, self.L0 * (M/self.M1)**self.p/(1+(M/self.M1)**self.q) * ((1+z)/7)**self.r, 0)
    
class TruncateAtCLF(LFModel):
    N_PARAMS = 5
    MCMC_X0 = [1.24, 1, 1.5, -21.91, np.log10(1.2e12)] #B15
    MCMC_PRIOR = [[0.8, 3.5], [0.8, 2.5], [0, 2.5], [-25, -20], [10, 14]]

    def __init__(self, meas_fn, params, dc=True, name=None, f_esc=0.2, log_input=True, M_trunc=None):
        super().__init__(meas_fn, params, dc, name, f_esc)
        self.param_names = [r'$p$', r'$q$', r'$r$',
                            r'$M_{\rm UV,0}$', r'$\log_{10}(M_1/$M$_\odot)$', r'$\log_{10}(M_{trunc}/$M$_\odot)$']
        self.p, self.q, self.r, self.L0, self.M1 = params
        self.M_trunc = M_trunc
        if log_input:
            self.L0 = utils.m_to_L(self.L0)
            self.M1 = 10**self.M1
            self.M_trunc = 10**self.M_trunc

    def to_log_input(self):
        return np.array([self.p, self.q, self.r, utils.L_to_m(self.L0), np.log10(self.M1), np.log10(self.M_trunc)])

    def L_c(self, M, z):
        return np.where(M > self.M_trunc, self.L0 * (M/self.M1)**self.p/(1+(M/self.M1)**self.q) * ((1+z)/7)**self.r, 0)
    
class TruncateAtFlatScaleCLF(TruncateAtCLF):
    def __init__(self, meas_fn, params, dc=True, name=None, f_esc=0.2, log_input=True):
        super().__init__(meas_fn, params, dc, name, f_esc, log_input, M_trunc=10.41775119) # best fit M_flat

class TruncateAt5e9CLF(TruncateAtCLF):
    def __init__(self, meas_fn, params, dc=True, name=None, f_esc=0.2, log_input=True):
        super().__init__(meas_fn, params, dc, name, f_esc, log_input, M_trunc=np.log10(5e9)) 



class PiecewiseZEvolutionCLF(LFModel):
    N_PARAMS = 6

    def __init__(self, meas_fn, params, dc=True, name=None, f_esc=0.2, log_input=True):
        super().__init__(meas_fn, params, dc, name, f_esc)
        self.param_names = [r'$p$', r'$q$', r'$r$',
                            r'$M_{\rm UV,0}$', r'$\log_{10}(M_1/$M$_\odot)$', r'$r\'$']
        self.p, self.q, self.r, self.L0, self.M1, self.r_prime = params
        if log_input:
            self.L0 = utils.m_to_L(self.L0)
            self.M1 = 10**self.M1

    def to_log_input(self):
        return np.array([self.p, self.q, self.r, utils.L_to_m(self.L0), np.log10(self.M1), self.r_prime])

    def L_c(self, M, z):
        z_crit = 8
        r = np.where(z <= z_crit, self.r, self.r_prime)
        # Normalization changes
        L0 = np.where(z <= z_crit, self.L0, self.L0 *
                      ((z_crit+1)/7)**(self.r-self.r_prime))
        retval = L0 * (M/self.M1)**self.p / \
            (1+(M/self.M1)**self.q) * ((1+z)/7)**r
        return retval


class ChangeScatterCLF(LFModel):
    N_PARAMS = 6

    def __init__(self, meas_fn, params, dc=True, name=None, f_esc=0.2, log_input=True):
        super().__init__(meas_fn, params, dc, name, f_esc)
        self.param_names = [r'$p$', r'$q$', r'$r$',
                            r'$M_{\rm UV,0}$', r'$\log_{10}(M_1/$M$_\odot)$', r'$\sigma$']
        self.p, self.q, self.r, self.L0, self.M1, sigma = params
        # np.where(M < 1e10, sigma, np.log(10)*0.16)
        self.sigma = lambda M: sigma
        if log_input:
            self.L0 = utils.m_to_L(self.L0)
            self.M1 = 10**self.M1

    def to_log_input(self):
        return np.array([self.p, self.q, self.r, utils.L_to_m(self.L0), np.log10(self.M1), self.sigma])

    def L_c(self, M, z):
        return self.L0 * (M/self.M1)**self.p/(1+(M/self.M1)**self.q) * ((1+z)/7)**self.r


class StepScatterCLF(LFModel):
    N_PARAMS = 7
    #MCMC_X0 = [1.24, 1, 1.5, -21.91, np.log10(1.2e12), 0.16*np.log(10), 9]
    MCMC_X0 = [1.85, 1.46, 1.03, -22.79, 11.71, 1.52, 9.71]
    MCMC_PRIOR = [[0.8, 2.5], [0.8, 5], [0, 2.5], [-25, -20], [10, 14], [0.05, 2.5], [8, 14]]

    def __init__(self, meas_fn, params, dc=True, name=None, f_esc=0.2, log_input=True):
        super().__init__(meas_fn, params, dc, name, f_esc)
        self.param_names = [r'$p$', r'$q$', r'$r$',
                            r'$M_{\rm UV,0}$', r'$\log_{10}(M_1/$M$_\odot)$', r'$\sigma$', r'$\log_{10}(M_2/$M$_\odot)$']
        self.p, self.q, self.r, self.L0, self.M1, sigma, self.M2 = params
        # np.where(M < 1e10, sigma, np.log(10)*0.16)
        self.sigma = lambda M: np.where(M < self.M2, sigma, 0.16*np.log(10))
        if log_input:
            self.L0 = utils.m_to_L(self.L0)
            self.M1 = 10**self.M1
            self.M2 = 10**self.M2

    def to_log_input(self):
        return np.array([self.p, self.q, self.r, utils.L_to_m(self.L0), np.log10(self.M1), self.sigma, np.log10(self.M2)])

    def L_c(self, M, z):
        return self.L0 * (M/self.M1)**self.p/(1+(M/self.M1)**self.q) * ((1+z)/7)**self.r


class EvolvingScatterCLF(LFModel):
    N_PARAMS = 7

    def __init__(self, meas_fn, params, dc=True, name=None, f_esc=0.2, log_input=True):
        super().__init__(meas_fn, params, dc, name, f_esc)
        self.param_names = [r'$p$', r'$q$', r'$r$',
                            r'$M_{\rm UV,0}$', r'$\log_{10}(M_1/$M$_\odot)$', r'$\sigma_1$', r'$s$']
        self.p, self.q, self.r, self.L0, self.M1, self.sigma1, self.s = params
        self.sigma = lambda M: self.sigma1 * (M/self.M1)**self.s
        if log_input:
            self.L0 = utils.m_to_L(self.L0)
            self.M1 = 10**self.M1

    def to_log_input(self):
        return np.array([self.p, self.q, self.r, utils.L_to_m(self.L0), np.log10(self.M1), self.sigma1, self.s])

    def L_c(self, M, z):
        return self.L0 * (M/self.M1)**self.p/(1+(M/self.M1)**self.q) * ((1+z)/7)**self.r


class FuzzyCLF(LFModel):
    N_PARAMS = 6
    MCMC_X0 = [1.24, 1, 1.5, -21.91, np.log10(1.2e12), 1]
    MCMC_PRIOR = [[0.8, 3.5], [0.8, 2.5], [0, 2.5], [-25, -20], [10, 14], [-1, 2]]

    def __init__(self, meas_fn, params, dc=True, name=None, f_esc=0.2, log_input=True):
        super().__init__(meas_fn, params, dc, name, f_esc)
        self.param_names = [r'$p$', r'$q$', r'$r$',
                            r'$M_{\rm UV,0}$', r'$\log_{10}(M_1/$M$_\odot)$', r'$\log_{10}m_{22}$']
        self.p, self.q, self.r, self.L0, self.M1, self.m_FDM22 = params
        if log_input:
            self.L0 = utils.m_to_L(self.L0)
            self.M1 = 10**self.M1
            self.m_FDM22 = 10**self.m_FDM22

        self.M0_schive = 1.6e10 * self.m_FDM22**(-4/3)

    def dn_dM(self, M, z):
        return self.mass_fn[z](M) * (1 + (M/self.M0_schive)**-1.1)**-2.2

    def to_log_input(self):
        return np.array([self.p, self.q, self.r, utils.L_to_m(self.L0), np.log10(self.M1), np.log10(self.m_FDM22)])

    def L_c(self, M, z):
        return np.clip(self.L0 * (M/self.M1)**self.p/(1+(M/self.M1)**self.q) * ((1+z)/7)**self.r, a_max=self.L_c_max(M,z), a_min=0)

class FuzzyJefferysCLF(LFModel):
    N_PARAMS = 6
    MCMC_X0 = [1.24, 1, 1.5, -21.91, np.log10(1.2e12), 1]

    def __init__(self, meas_fn, params, dc=True, name=None, f_esc=0.2, log_input=True):
        super().__init__(meas_fn, params, dc, name, f_esc)
        self.param_names = [r'$p$', r'$q$', r'$r$',
                            r'$M_{\rm UV,0}$', r'$\log_{10}(M_1/$M$_\odot)$', r'$\log_{10}m_{22}$']
        self.p, self.q, self.r, self.L0, self.M1, self.m_FDM22 = params
        if log_input:
            self.L0 = utils.m_to_L(self.L0)
            self.M1 = 10**self.M1
            self.m_FDM22 = 10**self.m_FDM22

        self.M0_schive = 1.6e10 * self.m_FDM22**(-4/3)

    def dn_dM(self, M, z):
        return self.mass_fn[z](M) * (1 + (M/self.M0_schive)**-1.1)**-2.2

    def to_log_input(self):
        return np.array([self.p, self.q, self.r, utils.L_to_m(self.L0), np.log10(self.M1), np.log10(self.m_FDM22)])

    def L_c(self, M, z):
        return np.clip(self.L0 * (M/self.M1)**self.p/(1+(M/self.M1)**self.q) * ((1+z)/7)**self.r, a_max=self.L_c_max(M,z), a_min=0)

    def MCMC_PRIOR(x):
        MCMC_FLAT_PRIOR = [[0.8, 3.5], [0.8, 2.5], [0, 2.5], [-25, -20], [10, 14]]
        for i, param in enumerate(x[:-1]):
            if param < min(MCMC_FLAT_PRIOR[i]) or param > max(MCMC_FLAT_PRIOR[i]) or np.isnan(param):
                return -np.inf
        arr = np.load('jefferys_prior_logspace.npz')
        prior, logvals = arr['prior'], arr['logvals']
        prior_interp = interp1d(logvals, prior, fill_value='extrapolate')
        return np.log(prior_interp(x[-1]))

class ReciprocalFuzzyCLF(LFModel):
    N_PARAMS = 6
    MCMC_X0 = [1.24, 1, 1.5, -21.91, np.log10(1.2e12), 0.1]
    MCMC_PRIOR = [[0.8, 3.5], [0.8, 2.5], [0, 2.5], [-25, -20], [10, 14], [0, 2]]

    def __init__(self, meas_fn, params, dc=True, name=None, f_esc=0.2, log_input=True):
        super().__init__(meas_fn, params, dc, name, f_esc)
        self.param_names = [r'$p$', r'$q$', r'$r$',
                            r'$M_{\rm UV,0}$', r'$\log_{10}(M_1/$M$_\odot)$', r'$1/m_{22}$']
        self.p, self.q, self.r, self.L0, self.M1, self.one_over_m_FDM22 = params
        if log_input:
            self.L0 = utils.m_to_L(self.L0)
            self.M1 = 10**self.M1
        
        self.m_FDM22 = 1/self.one_over_m_FDM22
        self.M0_schive = 1.6e10 * self.m_FDM22**(-4/3)

    def dn_dM(self, M, z):
        return self.mass_fn[z](M) * (1 + (M/self.M0_schive)**-1.1)**-2.2

    def to_log_input(self):
        return np.array([self.p, self.q, self.r, utils.L_to_m(self.L0), np.log10(self.M1), self.one_over_m_FDM22])

    def L_c(self, M, z):
        return np.clip(self.L0 * (M/self.M1)**self.p/(1+(M/self.M1)**self.q) * ((1+z)/7)**self.r, a_max=self.L_c_max(M,z), a_min=0)
    

class ReciprocalFuzzyTwiceChiCLF(ReciprocalFuzzyCLF):
    N_PARAMS = 6
    MCMC_X0 = [1.24, 1, 1.5, -21.91, np.log10(1.2e12), 0.1]
    MCMC_PRIOR = [[0.8, 3.5], [0.8, 2.5], [0, 2.5], [-25, -20], [10, 14], [0, 2]]

    def __init__(self, meas_fn, params, dc=True, name=None, f_esc=0.2, log_input=True):
        super().__init__(meas_fn, params, dc, name, f_esc)
        self.param_names = [r'$p$', r'$q$', r'$r$',
                            r'$M_{\rm UV,0}$', r'$\log_{10}(M_1/$M$_\odot)$', r'$1/m_{22}$']
        self.p, self.q, self.r, self.L0, self.M1, self.one_over_m_FDM22 = params
        if log_input:
            self.L0 = utils.m_to_L(self.L0)
            self.M1 = 10**self.M1
        
        self.m_FDM22 = 1/self.one_over_m_FDM22
        self.M0_schive = 1.6e10 * self.m_FDM22**(-4/3)

    def dn_dM(self, M, z):
        return self.mass_fn[z](M) * (1 + (M/self.M0_schive)**-1.1)**-2.2

    def to_log_input(self):
        return np.array([self.p, self.q, self.r, utils.L_to_m(self.L0), np.log10(self.M1), self.one_over_m_FDM22])

    def L_c(self, M, z):
        return np.clip(self.L0 * (M/self.M1)**self.p/(1+(M/self.M1)**self.q) * ((1+z)/7)**self.r, a_max=self.L_c_max(M,z), a_min=0)
    
    def chi_sq_of_model(self, at_each_point=False):
        return 2*super().chi_sq_of_model(at_each_point=at_each_point)

class ReciprocalFuzzyJefferysCLF(LFModel):
    N_PARAMS = 6
    MCMC_X0 = [1.24, 1, 1.5, -21.91, np.log10(1.2e12), 0.1]
    

    def __init__(self, meas_fn, params, dc=True, name=None, f_esc=0.2, log_input=True):
        super().__init__(meas_fn, params, dc, name, f_esc)
        self.param_names = [r'$p$', r'$q$', r'$r$',
                            r'$M_{\rm UV,0}$', r'$\log_{10}(M_1/$M$_\odot)$', r'$1/m_{22}$']
        self.p, self.q, self.r, self.L0, self.M1, self.one_over_m_FDM22 = params
        if log_input:
            self.L0 = utils.m_to_L(self.L0)
            self.M1 = 10**self.M1
        
        self.m_FDM22 = 1/self.one_over_m_FDM22
        self.M0_schive = 1.6e10 * self.m_FDM22**(-4/3)

    def dn_dM(self, M, z):
        return self.mass_fn[z](M) * (1 + (M/self.M0_schive)**-1.1)**-2.2

    def to_log_input(self):
        return np.array([self.p, self.q, self.r, utils.L_to_m(self.L0), np.log10(self.M1), self.one_over_m_FDM22])

    def L_c(self, M, z):
        return np.clip(self.L0 * (M/self.M1)**self.p/(1+(M/self.M1)**self.q) * ((1+z)/7)**self.r, a_max=self.L_c_max(M,z), a_min=0)
    
    def MCMC_PRIOR(x):
        MCMC_FLAT_PRIOR = [[0.8, 3.5], [0.8, 2.5], [0, 2.5], [-25, -20], [10, 14]]
        for i, param in enumerate(x[:-1]):
            if param < min(MCMC_FLAT_PRIOR[i]) or param > max(MCMC_FLAT_PRIOR[i]) or np.isnan(param):
                return -np.inf
        arr = np.load('jefferys_prior_inv.npz')
        prior, inv_vals = arr['prior'], arr['inv_m22s']
        prior_interp = interp1d(inv_vals, prior, fill_value='extrapolate')
        return np.log(prior_interp(x[-1]))
    
class ReciprocalFuzzy20CLF(LFModel):
    N_PARAMS = 6
    MCMC_X0 = [1.24, 1, 1.5, -21.91, np.log10(1.2e12), 10]
    MCMC_PRIOR = [[0.8, 3.5], [0.8, 2.5], [0, 2.5], [-25, -20], [10, 14], [0, 1000]]

    def __init__(self, meas_fn, params, dc=True, name=None, f_esc=0.2, log_input=True):
        super().__init__(meas_fn, params, dc, name, f_esc)
        self.param_names = [r'$p$', r'$q$', r'$r$',
                            r'$M_{\rm UV,0}$', r'$\log_{10}(M_1/$M$_\odot)$', r'$1/m_{20}$']
        self.p, self.q, self.r, self.L0, self.M1, self.one_over_m_FDM20 = params
        if log_input:
            self.L0 = utils.m_to_L(self.L0)
            self.M1 = 10**self.M1
        
        self.m_FDM20 = 1/self.one_over_m_FDM20
        self.m_FDM22 = self.m_FDM20 * 100
        self.M0_schive = 1.6e10 * self.m_FDM22**(-4/3)

    def dn_dM(self, M, z):
        return self.mass_fn[z](M) * (1 + (M/self.M0_schive)**-1.1)**-2.2

    def to_log_input(self):
        return np.array([self.p, self.q, self.r, utils.L_to_m(self.L0), np.log10(self.M1), self.one_over_m_FDM20])

    def L_c(self, M, z):
        return np.clip(self.L0 * (M/self.M1)**self.p/(1+(M/self.M1)**self.q) * ((1+z)/7)**self.r, a_max=self.L_c_max(M,z), a_min=0)


class ReciprocalFuzzyAtCLF(LFModel):
    N_PARAMS = 5
    MCMC_X0 = [1.24, 1, 1.5, -21.91, np.log10(1.2e12)]
    MCMC_PRIOR = [[0.8, 3.5], [0.8, 2.5], [0, 2.5], [-25, -20], [10, 14]]

    def __init__(self, meas_fn, params, dc=True, name=None, f_esc=0.2, log_input=True, one_over_m_FDM22=None):
        super().__init__(meas_fn, params, dc, name, f_esc)
        self.param_names = [r'$p$', r'$q$', r'$r$',
                            r'$M_{\rm UV,0}$', r'$\log_{10}(M_1/$M$_\odot)$']
        self.p, self.q, self.r, self.L0, self.M1 = params
        self.one_over_m_FDM22 = one_over_m_FDM22
        if log_input:
            self.L0 = utils.m_to_L(self.L0)
            self.M1 = 10**self.M1
        
        self.m_FDM22 = 1/self.one_over_m_FDM22
        self.M0_schive = 1.6e10 * self.m_FDM22**(-4/3)

    def dn_dM(self, M, z):
        return self.mass_fn[z](M) * (1 + (M/self.M0_schive)**-1.1)**-2.2

    def to_log_input(self):
        return np.array([self.p, self.q, self.r, utils.L_to_m(self.L0), np.log10(self.M1), self.one_over_m_FDM22])

    def L_c(self, M, z):
        return np.clip(self.L0 * (M/self.M1)**self.p/(1+(M/self.M1)**self.q) * ((1+z)/7)**self.r, a_max=self.L_c_max(M,z), a_min=0)
    
class FuzzyAt_1(ReciprocalFuzzyAtCLF):
    def __init__(self, meas_fn, params, dc=True, name=None, f_esc=0.2, log_input=True):
        super().__init__(meas_fn, params, dc, name, f_esc, log_input, one_over_m_FDM22=1/1)

class FuzzyAt_5(ReciprocalFuzzyAtCLF):
    def __init__(self, meas_fn, params, dc=True, name=None, f_esc=0.2, log_input=True):
        super().__init__(meas_fn, params, dc, name, f_esc, log_input, one_over_m_FDM22=1/5)

class FuzzyAt_10(ReciprocalFuzzyAtCLF):
    def __init__(self, meas_fn, params, dc=True, name=None, f_esc=0.2, log_input=True):
        super().__init__(meas_fn, params, dc, name, f_esc, log_input, one_over_m_FDM22=1/10)

class ShallowFuzzyCLF(LFModel):
    N_PARAMS = 8
    MCMC_X0 = [2.09, 1.43, 1.12, -21.88, 11.43, 10.56, -2, 1]
    #MCMC_PRIOR = [[0.8, 2.5], [0.8, 2.5], [0, 2.5], [-25, -20], [10, 14], [8, 14], [-5, 1.5], [0, 3]] --- commented out 12-18-24
    MCMC_PRIOR = [[0.8, 3.5], [0.8, 2.5], [0, 2.5], [-25, -20], [10, 14], [8, 14], [-3.5, 3.5], [0, 2]] 

    def __init__(self, meas_fn, params, dc=True, name=None, f_esc=0.2, log_input=True):
        super().__init__(meas_fn, params, dc, name, f_esc)
        self.param_names = [r'$p$', r'$q$', r'$r$',
                            r'$M_{\rm UV,0}$', r'$\log_{10}(M_1/$M$_\odot)$', r'$\log_{10}(M_2/$M$_\odot)$', r'$p_2$', r'$1/m_{22}$']
        self.p, self.q, self.r, self.L0, self.M1, self.M2, self.p2, self.one_over_m_FDM22 = params
        if log_input:
            self.L0 = utils.m_to_L(self.L0)
            self.M1 = 10**self.M1
            self.M2 = 10**self.M2

        self.m_FDM22 = 1/self.one_over_m_FDM22
        self.M0_schive = 1.6e10 * self.m_FDM22**(-4/3)

    def dn_dM(self, M, z):
        return self.mass_fn[z](M) * (1 + (M/self.M0_schive)**-1.1)**-2.2

    def to_log_input(self):
        return np.array([self.p, self.q, self.r, utils.L_to_m(self.L0), np.log10(self.M1), np.log10(self.M2), self.p2, self.one_over_m_FDM22])

    def L_c(self, M, z):
        retval = self.L0 * (M/self.M1)**self.p/(1+(M/self.M1)**self.q) * ((1+z)/7)**self.r
        corrected_retval = np.where(M > self.M2, retval, retval * (M/self.M2)**(self.p2 - self.p))
        return np.clip(corrected_retval, a_max=self.L_c_max(M,z), a_min=0)

class ShallowFuzzy20CLF(LFModel):
    N_PARAMS = 8
    MCMC_X0 = [2.09, 1.43, 1.12, -21.88, 11.43, 10.56, -2, 10]
    MCMC_PRIOR = [[0.8, 2.5], [0.8, 2.5], [0, 2.5], [-25, -20], [10, 14], [8, 14], [-5, 1.5], [0, 1000]]

    def __init__(self, meas_fn, params, dc=True, name=None, f_esc=0.2, log_input=True):
        super().__init__(meas_fn, params, dc, name, f_esc)
        self.param_names = [r'$p$', r'$q$', r'$r$',
                            r'$M_{\rm UV,0}$', r'$\log_{10}(M_1/$M$_\odot)$', r'$\log_{10}(M_2/$M$_\odot)$', r's', r'$1/m_{20}$']
        self.p, self.q, self.r, self.L0, self.M1, self.M2, self.s, self.one_over_m_FDM20 = params
        if log_input:
            self.L0 = utils.m_to_L(self.L0)
            self.M1 = 10**self.M1
            self.M2 = 10**self.M2

        self.one_over_m_FDM22 = 1/(self.one_over_m_FDM20/100)
        self.m_FDM20 = 1/self.one_over_m_FDM20
        self.m_FDM22 = self.m_FDM20 * 100
        self.M0_schive = 1.6e10 * self.m_FDM22**(-4/3)

    def dn_dM(self, M, z):
        return self.mass_fn[z](M) * (1 + (M/self.M0_schive)**-1.1)**-2.2

    def to_log_input(self):
        return np.array([self.p, self.q, self.r, utils.L_to_m(self.L0), np.log10(self.M1), np.log10(self.M2), self.s, self.one_over_m_FDM20])

    def L_c(self, M, z):
        retval = self.L0 * (M/self.M1)**self.p/(1+(M/self.M1)**self.q) * ((1+z)/7)**self.r
        corrected_retval = np.where(M > self.M2, retval, retval * (M/self.M2)**(utils.M_DOT_ACC_DELTA + self.s - self.p))
        return np.clip(corrected_retval, a_max=self.L_c_max(M,z), a_min=0)
    

class ShallowFuzzyMarshCLF(ShallowFuzzyCLF):
    def __init__(self, meas_fn, params, dc=True, name=None, f_esc=0.2, log_input=True):
        super().__init__(meas_fn, params, dc, name, f_esc, log_input)
        self.sample_fn = '../simplify/lf/marsh_mf/inv_log_spacing100.npz'
        self.mass_fn = self.create_marsh_mf()

    def create_marsh_mf_manually(self, z):
        EPS = 1e-100
        inv_mass = self.one_over_m_FDM22
        Mvals = np.logspace(7,16,1000)
        if inv_mass == 0:
            mf_obj = mass_function.MassFunction()
            func = 'ST'
        else:
            mf_obj = mass_function.MassFunction(m_FDM=1/inv_mass * 10**-22, is_marsh=True)
            func = 'FDM_Marsh'
        dndM_z = np.array([mf_obj.dndM(M=M, z=z, func=func) for M in Mvals]) + EPS
        self.mass_fn[z] = utils.loglog_interp(Mvals, dndM_z)

    def create_marsh_mf(self):
        return self.weighted_avg_interp()
    
    def load_marsh_mf(self):
        filename = self.sample_fn
        arr = np.load(filename)
        Mvals = arr['Mvals']
        inv_masses = arr['inv_masses']
        zvals = arr['zvals']
        mass_fns = defaultdict(list)
        for i, mf_z in enumerate(arr['mf']):
            z = zvals[i]
            for mf_z_fdm in mf_z:
                dndM_interp = utils.loglog_interp(Mvals, mf_z_fdm)
                mass_fns[z].append(dndM_interp)
        return np.array(Mvals), np.array(inv_masses), mass_fns

    def weighted_avg_interp(self):
            inv_mass = self.one_over_m_FDM22
            Mvals_samp, inv_masses_samp, mass_fns_interp_samp, = self.load_marsh_mf()
            Mvals = Mvals_samp
            zvals_samp = list(mass_fns_interp_samp.keys())
            close_ids = utils.closest_ids(x=inv_mass, arr=inv_masses_samp)
            mf_dict = defaultdict(list)
            for i, z in enumerate(zvals_samp):
                mf_samp_z = mass_fns_interp_samp[z]
                if close_ids[0] == close_ids[1]:
                    values_exp = utils.blog(mf_samp_z[close_ids[0]](Mvals))
                else:
                    delta = np.abs(utils.blog(inv_masses_samp[close_ids[1]]) - utils.blog(inv_masses_samp[close_ids[0]]))
                    w_lo = np.abs(utils.blog(inv_mass) - utils.blog(inv_masses_samp[close_ids[1]])) / delta
                    w_hi = 1 - w_lo
                    values_exp = w_lo * utils.blog(mf_samp_z[close_ids[0]](Mvals)) + w_hi * utils.blog(mf_samp_z[close_ids[1]](Mvals)) 
                interp_mf = utils.loglog_interp(Mvals_samp, np.exp(values_exp))
                mf_dict[z] = interp_mf
            return mf_dict
    
    def dn_dM(self, M, z):
        ''' Halo mass function.'''
        try:
            return self.mass_fn[z](M)
        except TypeError:
            self.create_marsh_mf_manually(z)
            return self.mass_fn[z](M)
        
class MarshCLF(LFModel):
    N_PARAMS = 6
    MCMC_X0 = [1.24, 1, 1.5, -21.91, np.log10(1.2e12), 0.1]
    MCMC_PRIOR = [[0.8, 3.5], [0.8, 2.5], [0, 2.5], [-25, -20], [10, 14], [0, 2]]

    def __init__(self, meas_fn, params, dc=True, name=None, f_esc=0.2, log_input=True):
        super().__init__(meas_fn, params, dc, name, f_esc)
        self.param_names = [r'$p$', r'$q$', r'$r$',
                            r'$M_{\rm UV,0}$', r'$\log_{10}(M_1/$M$_\odot)$', r'$1/m_{22}$']
        self.p, self.q, self.r, self.L0, self.M1, self.one_over_m_FDM22 = params
        if log_input:
            self.L0 = utils.m_to_L(self.L0)
            self.M1 = 10**self.M1
        
        self.m_FDM22 = 1/self.one_over_m_FDM22
        self.sample_fn = '../simplify/lf/marsh_mf/inv_log_spacing500_to_z20.npz'
        self.mass_fn = self.create_marsh_mf()

    def to_log_input(self):
        return np.array([self.p, self.q, self.r, utils.L_to_m(self.L0), np.log10(self.M1), self.one_over_m_FDM22])

    def L_c(self, M, z):
        return np.clip(self.L0 * (M/self.M1)**self.p/(1+(M/self.M1)**self.q) * ((1+z)/7)**self.r, a_max=self.L_c_max(M,z), a_min=0)

    def create_marsh_mf_manually(self, z):
        EPS = 1e-100
        inv_mass = self.one_over_m_FDM22
        Mvals = np.logspace(7,16,1000)
        if inv_mass == 0:
            mf_obj = mass_function.MassFunction()
            func = 'ST'
        else:
            mf_obj = mass_function.MassFunction(m_FDM=1/inv_mass * 10**-22, is_marsh=True)
            func = 'FDM_Marsh'
        dndM_z = np.array([mf_obj.dndM(M=M, z=z, func=func) for M in Mvals]) + EPS
        self.mass_fn[z] = utils.loglog_interp(Mvals, dndM_z)

    def create_marsh_mf(self):
        return self.weighted_avg_interp()
    
    def load_marsh_mf(self):
        filename = self.sample_fn
        arr = np.load(filename)
        Mvals = arr['Mvals']
        inv_masses = arr['inv_masses']
        zvals = arr['zvals']
        mass_fns = defaultdict(list)
        for i, mf_z in enumerate(arr['mf']):
            z = zvals[i]
            for mf_z_fdm in mf_z:
                dndM_interp = utils.loglog_interp(Mvals, mf_z_fdm)
                mass_fns[z].append(dndM_interp)
        return np.array(Mvals), np.array(inv_masses), mass_fns

    def weighted_avg_interp(self):
            inv_mass = self.one_over_m_FDM22
            Mvals_samp, inv_masses_samp, mass_fns_interp_samp, = self.load_marsh_mf()
            Mvals = Mvals_samp
            zvals_samp = list(mass_fns_interp_samp.keys())
            close_ids = utils.closest_ids(x=inv_mass, arr=inv_masses_samp)
            mf_dict = defaultdict(list)
            for i, z in enumerate(zvals_samp):
                mf_samp_z = mass_fns_interp_samp[z]
                if close_ids[0] == close_ids[1]:
                    values_exp = utils.blog(mf_samp_z[close_ids[0]](Mvals))
                else:
                    delta = np.abs(utils.blog(inv_masses_samp[close_ids[1]]) - utils.blog(inv_masses_samp[close_ids[0]]))
                    w_lo = np.abs(utils.blog(inv_mass) - utils.blog(inv_masses_samp[close_ids[1]])) / delta
                    w_hi = 1 - w_lo
                    try:
                        values_exp = w_lo * utils.blog(mf_samp_z[close_ids[0]](Mvals)) + w_hi * utils.blog(mf_samp_z[close_ids[1]](Mvals)) 
                    except TypeError:
                        values_exp = w_lo * utils.blog(mf_samp_z[close_ids[0][0]](Mvals)) + w_hi * utils.blog(mf_samp_z[close_ids[1][0]](Mvals))
                interp_mf = utils.loglog_interp(Mvals_samp, np.exp(values_exp))
                mf_dict[z] = interp_mf
            return mf_dict
    
    def dn_dM(self, M, z):
        ''' Halo mass function.'''
        try:
            return self.mass_fn[z](M)
        except TypeError:
            self.create_marsh_mf_manually(z)
            return self.mass_fn[z](M)
        
class FuzzySharpkCLF(LFModel):
    N_PARAMS = 6
    MCMC_X0 = [1.24, 1, 1.5, -21.91, np.log10(1.2e12), 0.1]
    MCMC_PRIOR = [[0.8, 3.5], [0.8, 2.5], [0, 2.5], [-25, -20], [10, 14], [0, 2]]

    def __init__(self, meas_fn, params, dc=True, name=None, f_esc=0.2, log_input=True):
        super().__init__(meas_fn, params, dc, name, f_esc)
        self.param_names = [r'$p$', r'$q$', r'$r$',
                            r'$M_{\rm UV,0}$', r'$\log_{10}(M_1/$M$_\odot)$', r'$1/m_{22}$']
        self.p, self.q, self.r, self.L0, self.M1, self.one_over_m_FDM22 = params
        if log_input:
            self.L0 = utils.m_to_L(self.L0)
            self.M1 = 10**self.M1
        
        self.m_FDM22 = 1/self.one_over_m_FDM22
        self.sample_fn = '../simplify/lf/sharpk_mf/sharpk_inv_log_spacing500_to_z20.npz'
        self.mass_fn = self.create_sharpk_mf()

    def to_log_input(self):
        return np.array([self.p, self.q, self.r, utils.L_to_m(self.L0), np.log10(self.M1), self.one_over_m_FDM22])

    def L_c(self, M, z):
        return np.clip(self.L0 * (M/self.M1)**self.p/(1+(M/self.M1)**self.q) * ((1+z)/7)**self.r, a_max=self.L_c_max(M,z), a_min=0)

    def create_sharpk_mf_manually(self, z):
        raise NotImplementedError

    def create_sharpk_mf(self):
        return self.weighted_avg_interp()
    
    def load_sharpk_mf(self):
        filename = self.sample_fn
        arr = np.load(filename)
        Mvals = arr['Mvals']
        inv_masses = arr['inv_masses']
        zvals = arr['zvals']
        mass_fns = defaultdict(list)
        for i, mf_z in enumerate(arr['mf']):
            z = zvals[i]
            for mf_z_fdm in mf_z:
                dndM_interp = utils.loglog_interp(Mvals, mf_z_fdm)
                mass_fns[z].append(dndM_interp)
        return np.array(Mvals), np.array(inv_masses), mass_fns

    def weighted_avg_interp(self):
            inv_mass = self.one_over_m_FDM22
            Mvals_samp, inv_masses_samp, mass_fns_interp_samp, = self.load_sharpk_mf()
            Mvals = Mvals_samp
            zvals_samp = list(mass_fns_interp_samp.keys())
            close_ids = utils.closest_ids(x=inv_mass, arr=inv_masses_samp)
            mf_dict = defaultdict(list)
            for i, z in enumerate(zvals_samp):
                mf_samp_z = mass_fns_interp_samp[z]
                if close_ids[0] == close_ids[1]:
                    values_exp = utils.blog(mf_samp_z[close_ids[0]](Mvals))
                else:
                    delta = np.abs(utils.blog(inv_masses_samp[close_ids[1]]) - utils.blog(inv_masses_samp[close_ids[0]]))
                    w_lo = np.abs(utils.blog(inv_mass) - utils.blog(inv_masses_samp[close_ids[1]])) / delta
                    w_hi = 1 - w_lo
                    try:
                        values_exp = w_lo * utils.blog(mf_samp_z[close_ids[0]](Mvals)) + w_hi * utils.blog(mf_samp_z[close_ids[1]](Mvals)) 
                    except TypeError:
                        values_exp = w_lo * utils.blog(mf_samp_z[close_ids[0][0]](Mvals)) + w_hi * utils.blog(mf_samp_z[close_ids[1][0]](Mvals)) 
                interp_mf = utils.loglog_interp(Mvals_samp, np.exp(values_exp))
                mf_dict[z] = interp_mf
            return mf_dict
    
    def dn_dM(self, M, z):
        ''' Halo mass function.'''
        try:
            return self.mass_fn[z](M)
        except TypeError:
            self.create_sharpk_mf_manually(z)
            return self.mass_fn[z](M)


    
class ShallowFuzzyMarsh20CLF(ShallowFuzzy20CLF):
    def __init__(self, meas_fn, params, dc=True, name=None, f_esc=0.2, log_input=True):
        super().__init__(meas_fn, params, dc, name, f_esc, log_input)
        self.sample_fn = '../simplify/lf/marsh_mf/inv_log_spacing.npz'
        self.mass_fn = self.create_marsh_mf()

    def create_marsh_mf(self):
        return self.weighted_avg_interp()
    
    def load_marsh_mf(self):
        filename = self.sample_fn
        arr = np.load(filename)
        Mvals = arr['Mvals']
        inv_masses = arr['inv_masses']
        zvals = arr['zvals']
        mass_fns = defaultdict(list)
        for i, mf_z in enumerate(arr['mf']):
            z = zvals[i]
            for mf_z_fdm in mf_z:
                dndM_interp = utils.loglog_interp(Mvals, mf_z_fdm)
                mass_fns[z].append(dndM_interp)
        return np.array(Mvals), np.array(inv_masses), mass_fns

    def weighted_avg_interp(self):
            inv_mass = self.one_over_m_FDM22
            Mvals_samp, inv_masses_samp, mass_fns_interp_samp, = self.load_marsh_mf()
            Mvals = Mvals_samp
            zvals_samp = list(mass_fns_interp_samp.keys())
            close_ids = utils.closest_ids(x=inv_mass, arr=inv_masses_samp)
            mf_dict = defaultdict(list)
            for i, z in enumerate(zvals_samp):
                mf_samp_z = mass_fns_interp_samp[z]
                if close_ids[0] == close_ids[1]:
                    values_exp = utils.blog(mf_samp_z[close_ids[0]](Mvals))
                else:
                    delta = np.abs(utils.blog(inv_masses_samp[close_ids[1]]) - utils.blog(inv_masses_samp[close_ids[0]]))
                    w_lo = np.abs(utils.blog(inv_mass) - utils.blog(inv_masses_samp[close_ids[1]])) / delta
                    w_hi = 1 - w_lo
                    values_exp = w_lo * utils.blog(mf_samp_z[close_ids[0]](Mvals)) + w_hi * utils.blog(mf_samp_z[close_ids[1]](Mvals)) 
                interp_mf = utils.loglog_interp(Mvals_samp, np.exp(values_exp))
                mf_dict[z] = interp_mf
            return mf_dict
    
    def dn_dM(self, M, z):
        ''' Halo mass function.'''
        return self.mass_fn[z](M)

class DutyCycleCLF(LFModel):
    N_PARAMS = 8

    def __init__(self, meas_fn, params, dc=True, name=None, f_esc=0.2, log_input=True):
        super().__init__(meas_fn, params, dc, name, f_esc)
        self.param_names = [r'$p$', r'$q$', r'$r$',
                            r'$M_{\rm UV,0}$', r'$\log_{10}(M_1/$M$_\odot)$', r'$f_{\rm duty,10}$', r'$\alpha_{\rm duty}$', r'$\gamma_{\rm duty}$']
        self.p, self.q, self.r, self.L0, self.M1, self.f_duty_10_6, self.alpha_duty, self.gamma_duty = params
        if log_input:
            self.L0 = utils.m_to_L(self.L0)
            self.M1 = 10**self.M1

    def to_log_input(self):
        return np.array([self.p, self.q, self.r, utils.L_to_m(self.L0), np.log10(self.M1), self.f_duty_10_6, self.alpha_duty, self.gamma_duty])

    def L_c(self, M, z):
        return self.L0 * (M/self.M1)**self.p/(1+(M/self.M1)**self.q) * ((1+z)/7)**self.r

    def duty_cycle(self, M, z):
        return np.minimum(self.f_duty_10_6 * (M/1e10)**self.alpha_duty * (1+z)**self.gamma_duty, 1)

    def phi_L_given_M(self, L, M, z):
        return self.duty_cycle(M, z) * super().phi_L_given_M(L, M, z)


class StepDutyCLF(LFModel):
    N_PARAMS = 7
    #MCMC_X0 = [1.24, 1, 1.5, -21.91, np.log10(1.2e12), 1, 9]
    MCMC_X0 = [1.41, 1.61, 0.98, -23.93, 12.21, 0.49, 11.08]
    MCMC_PRIOR = [[0.8, 2.5], [0, 5], [0, 2.5], [-26, -20], [10, 14], [0, 1], [8,14]]

    def __init__(self, meas_fn, params, dc=True, name=None, f_esc=0.2, log_input=True):
        super().__init__(meas_fn, params, dc, name, f_esc)
        self.param_names = [r'$p$', r'$q$', r'$r$',
                            r'$M_{\rm UV,0}$', r'$\log_{10}(M_1/$M$_\odot)$', r'$f_{\rm duty}$', r'$\log_{10}(M_2/$M$_\odot)$']
        self.p, self.q, self.r, self.L0, self.M1, self.f_duty, self.M_duty = params
        self.f_duty = min(self.f_duty, 1)
        if log_input:
            self.L0 = utils.m_to_L(self.L0)
            self.M1 = 10**self.M1
            self.M_duty = 10**self.M_duty

    def to_log_input(self):
        return np.array([self.p, self.q, self.r, utils.L_to_m(self.L0), np.log10(self.M1), self.f_duty, np.log10(self.M_duty)])

    def L_c(self, M, z):
        return self.L0 * (M/self.M1)**self.p/(1+(M/self.M1)**self.q) * ((1+z)/7)**self.r

    def duty_cycle(self, M, z):
        return np.where(M < self.M_duty, self.f_duty, 1)

    def phi_L_given_M(self, L, M, z):
        return self.duty_cycle(M, z) * super().phi_L_given_M(L, M, z)
    
    def f_star(self, M, z):
        return self.duty_cycle(M, z) * super().f_star(M, z)
