import numpy as np
from astropy import constants as const, units as u
from astropy.cosmology import Planck18 as cosmo
from scipy.misc import derivative
import camb

import utils

from scipy.integrate import quad
import scipy.signal as sps
import scipy.special

class MassFunction2:
    def __init__(self, camb_fn='../simplify/P_k.npz', load=True, m22=0, is_marsh=False):
        if not load:
            raise NotImplementedError
        arr = np.load(camb_fn)
        k, Pk_0, s8 = arr['k'], arr['Pk_0'], arr['sigma_8']
        self.kmin, self.kmax = k[0], k[-1]
        self.default_f_nu = 'ST'
        self.m22 = m22
        assert self.m22 >= 0
        if self.m22 > 0:
            if is_marsh:
                self.default_f_nu = 'Marsh'
            self.kj_eq = 9.11 * self.m22**(1/2)
            Pk_0 = Pk_0 * self.fdm_transfer_fn(k)**2
        self.Pk_0 = utils.loglog_interp(k, Pk_0)
        self.Delta2_0 = utils.loglog_interp(k, k**3 * Pk_0 / (2*np.pi**2))
        z_vals = np.linspace(0, 21, 1000)
        self.Dg_0 = self.growth_factor(0, normalized=False)
        Dg = np.array([self.growth_factor(z)for z in z_vals])
        self.Dg = utils.linlog_interp(z_vals, Dg)
        M_vals = np.logspace(7, 16, 2000)

    def fdm_transfer_fn(self, k):
        x = 1.61 * self.m22**(1/18) * k / self.kj_eq
        return np.cos(x**3) / (1+x**8)
    
    def growth_factor(self, z, normalized=True):
        prefactor = 5*utils.OMEGA_M*utils.COSMO.H(z)/(2*utils.H0)
        if normalized:
            prefactor /= self.Dg_0
        def integrand(z_prime): return (1+z_prime) / \
            (utils.COSMO.H(z_prime)/utils.H0)**3
        integral = utils.trapz_integrate(integrand, z, 100, logspace=False, n_intervals=100000)
        return prefactor * integral
    
    def tophat_window(self, k, R):
        return 3 * (np.sin(k*R) - k*R*np.cos(k*R)) / (k*R)**3
    
    def sharp_k_window(self, k, R, alpha=2.5):
        k0 = alpha / R
        k = np.array(k)
        return np.where(k > k0, 1, 0)
    
    def Sigma_R(self, z, R):
        pass

    def Sigma_M(self, M):
        pass
        


class MassFunction:
    def __init__(self, camb_fn='../simplify/P_k.npz', load=True, m_FDM=0, is_marsh=False, window='top hat', fdm_growth=False):
        if not load:
            pars = camb.CAMBparams()
            #This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency
            ombh2 = utils.OMEGA_B*utils.LITTLE_H**2
            omch2 = (utils.OMEGA_M-utils.OMEGA_B)*utils.LITTLE_H**2
            pars.set_cosmology(H0=utils.H0.value, ombh2=ombh2,
                               omch2=omch2, omk=0.0)
            pars.InitPower.set_params(As=2.147656261896007e-9, ns=utils.N_S)
            # Default lmax in web tool
            pars.set_for_lmax(2200, lens_potential_accuracy=1)

            pars.set_matter_power(redshifts=[0], kmax=2000)
            pars.NonLinear = camb.model.NonLinear_none
            results = camb.get_results(pars)
            kh, z, pk = results.get_matter_power_spectrum(
                minkh=1e-6, maxkh=100, npoints=100000)
            k = kh * utils.LITTLE_H
            Pk_0 = pk[0] / utils.LITTLE_H**3
            s8 = results.get_sigma8()
            np.savez('../simplify/P_k.npz', k=k, Pk_0=Pk_0, sigma_8=s8)
        else:
            arr = np.load(camb_fn)
            k, Pk_0, s8 = arr['k'], arr['Pk_0'], arr['sigma_8']

        ##############print('sigma8 is:', s8)
        self.kmin, self.kmax = k[0], k[-1]
        self.default_mf_func = "ST"
        self.window = window
        self.fdm_growth = fdm_growth
        assert m_FDM >= 0
        if m_FDM > 0:
            if is_marsh:
                self.default_mf_func = "FDM_Marsh"
            self.m_22 = m_FDM / 10**-22
            self.kj_eq = 9.11 * self.m_22**(1/2) #TODO: CHECK UNITS (this is Mpc^-1)
            if True:#not is_marsh:
                Pk_0 = [
                    Pk_0[i] * self.FDM_correction(k[i]) for i in range(len(k))]
        self.Pk_0 = utils.loglog_interp(k, Pk_0)
        self.Delta_2_0 = utils.loglog_interp(k, k**3 * Pk_0 / (2*np.pi**2))
        z_vals = np.linspace(0, 21, 1000)
        if fdm_growth:
            kvals = np.logspace(self.kmin, self.kmax, 1000)
            Dg_0 = [self.growth_factor(z=0, k=k, normalized=False) for k in kvals]
            self.Dg_0 = utils.linlog_interp(kvals, Dg_0, fill_value='extrapolate')
            self.Dg = self.growth_factor
        else:
            self.Dg_0 = lambda k: self.growth_factor(z=0, normalized=False)
            Dg = np.array([self.growth_factor(z) for z in z_vals])
            self.Dg = lambda z, k: utils.linlog_interp(z_vals, Dg)(z)
        
        M_vals = np.logspace(7, 16, 1000)
        sigma_M = [self.Sigma_M(M) for M in M_vals]
        self.sigma_M = utils.loglog_interp(M_vals, sigma_M)
        if window == 'sharp k':
            dlogsigma_dlogM_fn = lambda M: -1/(2*3*self.sigma_M(M)**2) * self.Delta_2_0(2.5/self.R_M(M))
            dlogsigma_dlogM = [dlogsigma_dlogM_fn(M) for M in M_vals[:-1]]
        else:
            dlogsigma_dlogM = self.dlogSigma_dlogM(M_vals, sigma_M)
        # a little bit of error here from derivative sorta being evaluated BETWEEN M_vals
        self.dlogSigma_dlogM = utils.loglin_interp(
            M_vals[:-1], dlogsigma_dlogM)
        #self.dlogSigma_dlogM = lambda M: -1*neg_dlogSigma_dlogM(M)
        n_M_zvals = np.arange(0, 21)
        self.n_M = []
        for z in n_M_zvals:
            n_M = np.array([self.dndM(M, z=z) for M in M_vals])
            self.n_M.append(utils.loglog_interp(M_vals, n_M))

        self.marsh_M_j = None

    def Pk_z(self, k, z):
        return self.Pk_0(k) * self.Dg(z,k=k)

    def growth_factor_integrand(self, z_prime):
        return (1+z_prime) / (utils.COSMO.H(z_prime)/utils.H0)**3

    def growth_factor_prefactor(self, z):
        return 5*utils.OMEGA_M*utils.COSMO.H(z)/(2*utils.H0)

    def growth_factor(self, z, k=None, normalized=True):
        if self.fdm_growth:
            a = 1/(1+z)
            a_i = 1#1/(1+100)
            arg_prefactor1 = const.hbar * (k*u.Mpc**(-1))**2
            arg_prefactor2 = self.m_22 * 10**22 * u.eV * (1/const.c**2) * utils.H0
            arg_prefactor = arg_prefactor1 / arg_prefactor2
            arg_prefactor = arg_prefactor.decompose()
            #print(arg_prefactor)
            term1 = (a_i/a)**(1/4)
            term2 = scipy.special.jv(-5/2, arg_prefactor/np.sqrt(a))
            term3 = scipy.special.jv(-5/2, arg_prefactor/np.sqrt(a_i))
            retval = term1 *  term2 / term3
        else:
            prefactor = 5*utils.OMEGA_M*utils.COSMO.H(z)/(2*utils.H0)  
            def integrand(z_prime): return (1+z_prime) / \
            (utils.COSMO.H(z_prime)/utils.H0)**3
            integral = utils.trapz_integrate(integrand, z, 100, logspace=False)
            retval = prefactor * integral
            if normalized:
                retval = retval / self.Dg_0(k)
        return retval

    def Delta_2_z(self, k, z):
        return self.Delta_2_0(k) * self.Dg(z,k=k)

    def Theta_R(self, k, R):
        k = k / u.Mpc
        volume = 4/3 * np.pi * R**3
        result = 4*np.pi/(volume * k**3) * \
            (-k*R*np.cos(k*R*u.rad) + np.sin(k*R*u.rad))
        return result
    
    def sharp_k_window(self, k, R, alpha=2.5):
        k = k / u.Mpc
        k0 = alpha / R
        return np.where(k <= k0, 1, 0)

    def sigma_R_integrand(self, z, R, kval):
        R = R * u.Mpc
        a = np.abs(self.Theta_R(kval, R=R))**2
        b = 1/kval
        c = self.Delta_2_z(kval, z=z)
        return a * b * c

    def Sigma_R(self, z, R):
        klo, khi = self.kmin, self.kmax
        if self.window == 'top hat':
            def integrand(kval): return np.abs(self.Theta_R(
                kval, R=R))**2 * 1/kval * self.Delta_2_z(kval, z=z)
        elif self.window == 'sharp k':
            khi = 2.5/R * u.Mpc
            def integrand(kval): return  1/kval * self.Delta_2_z(kval, z=z)
            # def integrand(kval): return np.abs(self.sharp_k_window(
            #     kval, R=R))**2 * 1/kval * self.Delta_2_z(kval, z=z)
        else:
            raise NotImplementedError
        integral = utils.trapz_integrate(
            integrand, klo, khi, logspace=True, n_intervals=10000)
        sigmaR = np.sqrt(integral)
        return sigmaR

    def R_M(self, M):
        return np.cbrt(3*M/(4*np.pi*utils.RHO_M*u.Mpc**3/u.solMass))

    def Sigma_M(self, M):
        if type(M).__module__ != 'astropy.units.quantity':
            M = M * u.solMass
        R = np.cbrt(3*M/(4*np.pi*utils.RHO_M))
        return self.Sigma_R(0, R)

    def dlogSigma_dlogM(self, M_vals, sigma_M):
        return utils.log_derivative(x=M_vals, y=sigma_M)
    
    def init_marsh_M_j(self, a1):
        self.marsh_M_j = a1 * 1e8 * self.m_22**(-3/2) * (utils.OMEGA_M*utils.LITTLE_H**2/0.14)**(1/4) / utils.LITTLE_H

    def marsh_barrier(self, M, a2, a3, a4, a5, a6):
        def marsh_h(x, a2=a2):
            return 0.5 * (1 - np.tanh(self.marsh_M_j*(x-a2)))
        x = M / self.marsh_M_j
        h = marsh_h(x)
        return h * np.exp(a3*x**(-a4)) + (1-h)*np.exp(a5*x**(-a6))

    def marsh_barrier_FIXED(self, M, a2, a3, a4, a5, a6):
        def marsh_h(x, a2=a2):
            return 0.5 * np.tanh(utils.LITTLE_H*self.marsh_M_j*(x - a2))
        x = M / self.marsh_M_j
        h = marsh_h(x)
        return h * np.exp(a3*x**(-a4)) + (1-h)*np.exp(a5*x**(-a6))

    def marsh_barrier_FIXED_BUT_PAPER_TANH(self, M, a2, a3, a4, a5, a6):
        def marsh_h(x, a2=a2):
            return 0.5 * (1 - np.tanh(utils.LITTLE_H*self.marsh_M_j*(x - a2)))
        x = M / self.marsh_M_j
        h = marsh_h(x)
        return h * np.exp(a3*x**(-a4)) + (1-h)*np.exp(a5*x**(-a6))

    def hmf_f(self, sigma, func, M):
        delta_c = utils.DELTA_C
        if self.window == 'sharp k':
            delta_c = delta_c * 1.195
        if func == "PS":
            f = np.sqrt(2/np.pi) * (delta_c/sigma) * \
                np.exp(-delta_c**2/(2*sigma**2))
        elif func == "ST":
            A = 0.322
            a = 0.707
            p = 0.3
            f = A * np.sqrt(2*a/np.pi) * (1 + (sigma**2/(a*delta_c**2))**p) * \
                (delta_c/sigma) * np.exp(-a*delta_c**2/(2*sigma**2))
        elif func == "FDM_Marsh_old":
            self.init_marsh_M_j(a1=3.4)
            delta = delta_c * self.marsh_barrier(M, a2=1.0, a3=1.8, a4=0.5, a5=1.7, a6=0.9)
            A = 0.322
            a = 0.707
            p = 0.3
            f = A * np.sqrt(2*a/np.pi) * (1 + (sigma**2/(a*delta**2))**p) * \
                (delta/sigma) * np.exp(-a*delta**2/(2*sigma**2))
        elif func == "FDM_Marsh":
            self.init_marsh_M_j(a1=3.4)
            delta = delta_c * self.marsh_barrier_FIXED(M, a2=1.0, a3=1.8, a4=0.5, a5=1.7, a6=0.9)
            A = 0.322
            a = 0.707
            p = 0.3
            f = A * np.sqrt(2*a/np.pi) * (1 + (sigma**2/(a*delta**2))**p) * \
                (delta/sigma) * np.exp(-a*delta**2/(2*sigma**2))
        elif func == "FDM_Marsh_FIXED_BUT_PAPER_TANH":
            self.init_marsh_M_j(a1=3.4)
            delta = delta_c * self.marsh_barrier_FIXED_BUT_PAPER_TANH(M, a2=1.0, a3=1.8, a4=0.5, a5=1.7, a6=0.9)
            A = 0.322
            a = 0.707
            p = 0.3
            f = A * np.sqrt(2*a/np.pi) * (1 + (sigma**2/(a*delta**2))**p) * \
                (delta/sigma) * np.exp(-a*delta**2/(2*sigma**2))
        else:
            raise NotImplementedError
        return f

    def dndM(self, M, z, func=None):
        if func is None:
            func = self.default_mf_func
        sigma_z = self.sigma_M(M) * self.Dg(z,k=2.5/self.R_M(M))
        dlogsigma_dlogM_z = self.dlogSigma_dlogM(M)
        prefactor = -(utils.RHO_M.value/M**2) * dlogsigma_dlogM_z
        return np.nan_to_num(prefactor * self.hmf_f(sigma_z, func, M), nan=1e-100)

    def FDM_x(self, k):
        return 1.61 * self.m_22**(1/18) * k/self.kj_eq

    def FDM_correction(self, k):
        x = self.FDM_x(k)
        return (np.cos(x**3)/(1+x**8))**2
    
    def k_half_FDM(self, z):
        k_half =  4.5 * self.m_22**(4/9)
        two_P_k_half = 2*self.Pk_z(k=k_half, z=z)
        return k_half, two_P_k_half
