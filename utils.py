import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import csv
import sys

from scipy.integrate import trapz
from scipy.interpolate import interp1d, interp2d
import scipy.stats as sps
from astropy import constants as const, units as u
from astropy.cosmology import Planck18 as cosmo

LINESTYLE_ARR = ['-', '--', ':', '-.', (0, (3, 1, 1, 1, 1, 1)), (5, (10, 3)), *(10*['-'])]
MARKER_ARR = ['o', 's', 'D', 'x']
ONE_SIGMA = sps.norm.cdf(-1)

import inspect

def debug_function(func):
    def wrapper(*args, **kwargs):
        # Print function name
        print("Function name:", func.__name__)
        
        # Print argument names and values
        signature = inspect.signature(func)
        bound_arguments = signature.bind(*args, **kwargs).arguments
        
        for arg_name, arg_value in bound_arguments.items():
            print(f"{arg_name} = {arg_value}")
        
        # Call the original function
        return func(*args, **kwargs)
    
    return wrapper

def sigma_to_p_value(sigma):
    return sps.norm.cdf(-1*sigma)

#"bounded" log
def blog(x, base=np.e, lower_bound=-10):
    return np.log(np.maximum(x, lower_bound)) / np.log(base)


# Sort the phis array based on the sorted order of mags
def sort_x_by_order_of_y(x, y):
    x,y = np.array(x), np.array(y)
    sorted_indices = np.argsort(y)
    sorted_x = x[sorted_indices]
    return sorted_x


def load_csv(fn, delim=' '):
    x = []
    y = []
    with open(fn) as f:
        reader = csv.reader(f, delimiter=delim)
        for row in reader:
            x.append(float(row[0]))
            y.append(float(row[1]))
    return to_array(x), to_array(y)


def my_mpl():
    plt.rc('font', family='serif', size=20)
    plt.rc('axes', grid=True)
    plt.rc('lines', lw=4)
    ts = 8
    plt.rc('xtick.minor', size=ts-2)
    plt.rc('ytick.minor', size=ts-2)
    plt.rc('xtick.major', size=ts)
    plt.rc('ytick.major', size=ts)
    plt.rc('figure', figsize=[16, 9])


MF_FN = "../np_files/mass_fns.npz"
MF_FN_NEW = "../simplify/mass_fns.npz"

# def load_mf(fn=MF_FN_NEW):
#     mf = np.load(fn)
#     Mvals = mf['Mvals']
#     mass_fns = {}
#     for z in mf.files:
#         if z[1:].isnumeric():  # skip mf['Mvals']
#             if z[1:] == '1375':
#                 new_z = 13.75
#             else:
#                 new_z = int(z[1:])
#         elif z[1:] == '13.75':
#             new_z = float(z[1:])
#         else:
#             continue
#         mass_fns[new_z] = loglog_interp(Mvals, mf[z])
#     return mass_fns

def load_mf(fn=MF_FN_NEW):
    mf = dict(np.load(fn))
    mass_fns = dict()
    Mvals = mf['Mvals']
    for z, dn_dM_z in mf.items():
        if z == "Mvals":
            continue
        if int(z) == 14:
            ####TODO: THIS SHOULD BE CHANGED
            mass_fns[13.75] = loglog_interp(Mvals, dn_dM_z)
        mass_fns[int(z)] = loglog_interp(Mvals, dn_dM_z)
    return mass_fns

    


def log_derivative(x, y):
    dlogx = np.diff(np.log(np.abs(x)))
    dlogy = np.diff(np.log(np.abs(y)))
    return dlogy / dlogx


def to_array(x, extend=1):
    try:
        return np.array(extend*list(iter(x)))
    except TypeError:
        return np.array(extend*[x])

NU_H_ALPHA = ((const.c / (6563 * u.Angstrom)).to('Hz')).value

COSMO = cosmo
A_HE = 1.22
Y_P = (4/3) * (1 - 1/1.22)
C_HII = 3
F_GAMMA = 4000
OMEGA_M = cosmo.Om0
OMEGA_L = cosmo.Ode0
OMEGA_B = cosmo.Ob0
H0 = cosmo.H0
LITTLE_H = H0.value/100
RHO_CRIT = (cosmo.critical_density0).to('Msun/Mpc^3')
RHO_M = OMEGA_M * RHO_CRIT
RHO_B = OMEGA_B * RHO_CRIT
M_H = const.m_p
N_H0 = ((1-Y_P) * OMEGA_B * RHO_CRIT / M_H).to('cm^-3')
KAPPA_UV = 1.15e-28  # u.Msun / u.yr / (u.erg * u.s**-1 * u.Hz**-1)
M_DOT_ACC_DELTA = 1.127
N_S = 0.9649
SIGMA_8 = 0.818
DELTA_C = 1.686


def dt_dz(z):
    return 1/(cosmo.H(z).to('Gyr^-1').value*(1+z))


def m_to_L(m):
    log10_L = 0.4 * (51.6 - m)
    return 10**log10_L


def L_to_m(L):
    return 51.6 - 2.5 * np.log10(L)


def dL_dm(m):
    return 0.4*np.log(10) * m_to_L(m)


def loglog_interp(x, y, fill_value='extrapolate'):
    logx = np.log(x)
    logy = np.log(y)
    lin_interpolator = interp1d(
        logx, logy, bounds_error=False, fill_value=fill_value)

    def loglog_interpolator(z): return np.exp(lin_interpolator(np.log(z)))
    return loglog_interpolator


def loglin_interp(x, y, fill_value='extrapolate'):

    # if sgn_x != sgn_y:
    #    raise NotImplementedError
    logx = np.log(x)
    lin_interpolator = interp1d(
        logx, y, bounds_error=False, fill_value=fill_value)

    def loglin_interpolator(z): return lin_interpolator(np.log(z))
    return loglin_interpolator


def linlog_interp(x, y, fill_value='extrapolate'):
    x = to_array(x)
    y = to_array(y) 

    # if sgn_x != sgn_y:
    #    raise NotImplementedError
    logy = np.log(y)
    lin_interpolator = interp1d(
        x, logy, bounds_error=False, fill_value=fill_value)

    def linlog_interpolator(z): return np.exp(lin_interpolator(z))
    return linlog_interpolator

def logloglin_interp(x, y, z, fill_value='extrapolate'):
    logx = np.log(x)
    logy = np.log(y)
    lin_interpolator = interp1d(
        logx, logy, bounds_error=False, fill_value=fill_value)

    def loglog_interpolator(x1, x2): return np.exp(lin_interpolator(x1, x2))
    return loglog_interpolator

def logloglog_interp(x, y, z, fill_value='extrapolate'):
    logx = np.log(x)
    logy = np.log(y)
    logz = np.log(z)
    lin_interpolator = interp2d(
        logx, logy, logz, bounds_error=False, fill_value=fill_value)

    def logloglog_interpolator(x1, y1): return np.exp(np.ravel(lin_interpolator(np.log(x1), np.log(y1))))
    return logloglog_interpolator


def trapz_integrate(integrand, lo, hi, n_intervals=1000, logspace=False):
    integrand_fn = integrand
    if logspace:
        lo, hi = np.log(lo), np.log(hi)
        def integrand_fn(logx): return np.exp(logx) * integrand(np.exp(logx))
    x = np.linspace(lo, hi, n_intervals)
    y = integrand_fn(x)
    dx = x[1] - x[0]
    return trapz(y=y, x=x, dx=dx)

def log_like(xhat, x, sigma_plus, sigma_minus=None):
    return -0.5 * chi_squared(measured=xhat, theory=x, sigma_plus=sigma_plus, sigma_minus=sigma_minus)

def chi_squared(measured, theory, sigma_plus, sigma_minus=None):
    sq_dif = (measured-theory)**2
    #if sigma_minus is None:  # symmetric error
    #    sigma_minus = sigma_plus
    sigma_minus = np.where(sigma_minus is None, sigma_plus, sigma_minus)
    #if sigma_minus == 0:  # upper bound only so assume half-gaussian
    #    half_normal_factor = 1/np.sqrt(2)
    #    return sq_dif / (half_normal_factor * sigma_plus)**2
    half_normal_factor = 1/np.sqrt(2)
    sigma_plus = np.where(sigma_minus==0, sigma_plus*half_normal_factor, sigma_plus)
    sigma_minus = np.where(sigma_minus==0, sigma_plus, sigma_minus)
    sigma = 2 * sigma_plus * sigma_minus / (sigma_plus+sigma_minus)
    sigma_prime = (sigma_plus-sigma_minus)/(sigma_plus+sigma_minus)
    denominator = (sigma + sigma_prime*(theory-measured))**2
    return sq_dif / denominator

# def closest_ids(x, arr):
#     return [max(np.where(x>=arr)[0]), min(np.where(x<=arr)[0])]

def closest_ids(x, arr):
    idx_equal = np.where(arr == x)[0]
    if idx_equal.size > 0:
        return [idx_equal[0], idx_equal[0]]
    if x <= arr[0]:
        return [0, 0]
    if x >= arr[-1]:
        return [len(arr) - 1, len(arr) - 1]    
    idx_greater = np.searchsorted(arr, x, side='right')
    idx_smaller = idx_greater - 1
    return [idx_smaller, idx_greater]

def rho_UV_WF(z, g_006=1):
    low_bound = 10**24.52 * (1/g_006) * np.sqrt(18/(1+z))
    upper_minimal_coupling = 10**25 * (1/g_006) * np.sqrt(18/(1+z))
    return low_bound, upper_minimal_coupling
