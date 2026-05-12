from multiprocessing import Pool
import os
import lf_model
import emcee
import numpy as np
import scipy.optimize
import utils

B15_X0 = [1.24, 1, 1.5, -21.91, np.log10(1.2e12)]
DEFAULT_PRIOR = [[0.8, 2.5], [
    0.8, 2.5], [0, 2.5], [-24, -20], [10, 14]]
BEST_X0 = [1.68, 1.57, 1.07, -23.33, 11.91]
TEST_X0 = [1.78862447,   1.50849723,   1.29258025, -23.21602454,
        11.88000283]

B15_X0_FT = [1.24, 1, 1.5, -21.91, np.log10(1.2e12), np.log10(1e9)]

B15_X0_DUTY = [1.24, 1, 1.5, -21.91, np.log10(1.2e12), 1, 0, 0]
BEST_X0_DUTY = [1.68, 1.57, 1.07, -23.33, 11.91, 1, 0, 0]
DUTY_PRIOR = [[0.8, 2.5], [
    0.8, 2.5], [0, 2.5], [-24.5, -20], [10, 14], [0, 1], [-3, 3], [-1, 5]]

B15_X0_DUTY2 = [1.24, 1, 1.5, -21.91, np.log10(1.2e12), 1, 9]
BEST_X0_DUTY2 = [1.68, 1.57, 1.07, -23.33, 11.91, 1, 9]
DUTY_PRIOR2 = [[0.8, 2.5], [
    0.8, 2.5], [0, 2.5], [-24.5, -20], [10, 14], [0, 1], [8,14]]

B15_X0_SHALLOW = [1.24, 1, 1.5, -21.91, np.log10(1.2e12), 9, 1.24]
BEST_X0_SHALLOW = [1.68, 1.57, 1.07, -23.33, 11.91, 9, 1.68-utils.M_DOT_ACC_DELTA]
SHALLOW_PRIOR = [[0.8, 2.5], [
    0.8, 2.5], [0, 2.5], [-24.5, -20], [10, 14], [8, 14], [0.8, 2.5]]

BEST_X0_FT = [1.68, 1.57, 1.07, -23.33, 11.91, 9]
BEST_X0_SH = [1.68, 1.57, 1.07, -23.33, 11.91, 9, utils.M_DOT_ACC_DELTA]
MCMC_X0_FT = [1.98, 1.43, 1.03, -22.27, 11.52, 10.39]
FT_PRIOR = [[0.8, 2.5], [0.8, 2.5],
            [0, 2.5], [-24, -20], [10, 14], [8, 14]]
SCATTER_PRIOR = [[0.8, 2.5], [0.8, 5],
                 [0, 2.5], [-24, -20], [10, 14], [0.1, 1], [-2, 2]]
BEST_X0_SCATTER = [1.68, 1.57, 1.07, -23.33, 11.91, 0.16*np.log(10), 0]

BEST_X0_SCATTER2 = [1.6822196,   3.92029909,   1.20652811, -23.02269523,
                    11.83665656,   0.65787342,   0.97946327]

B15_X0_SCATTER = [1.24, 1, 1.5, -21.91, np.log10(1.2e12), 0.16*np.log(10), 9]
BEST_X0_SCATTER3 = [1.68, 1.57, 1.07, -23.33, 11.91, 0.16*np.log(10), 9]
STEP_SCATTER_PRIOR = [[0.8, 2.5], [0.8, 5],
                 [0, 2.5], [-24, -20], [10, 14], [0.05, 2.5], [8, 14]]

B15_X0_FDM = [1.24, 1, 1.5, -21.91, np.log10(1.2e12), 2]
BEST_X0_FDM = [1.68, 1.57, 1.07, -23.33, 11.91, 2]
BEST_X0_FDM_BIGP = [3, 1.57, 1.07, -23.33, 11.91, 2]
FDM_PRIOR = [[0.8, 4], [0.8, 2.5],
             [0, 2.5], [-24, -20], [10, 14], [-2, 3]]


class LFOptimizer:
    def __init__(self, meas_fn, ModelClass, x0=B15_X0, prior=DEFAULT_PRIOR, minimize_method='Nelder-Mead', **minimize_kwargs):
        self.meas = lf_model.LFMeasurements(meas_fn)
        self.meas_fn = meas_fn
        self.ModelClass = ModelClass
        self.x0 = x0
        self.prior = prior
        self.scipy_output = None
        self.minimize_method = minimize_method
        self.minimize_kwargs = minimize_kwargs

    def get_chi_sq(self, params):
        model = self.ModelClass(self.meas_fn, params)
        chi_sq = model.chi_sq_of_model()
        return chi_sq

    def optimize(self):
        self.scipy_output = scipy.optimize.minimize(
            self.get_chi_sq, self.x0, bounds=self.prior, method=self.minimize_method, options=self.minimize_kwargs)
        return self.scipy_output

    def optimize_loop(self, n_iter=3):
        for i in range(n_iter):
            opt_output = self.optimize()
            self.x0 = self.optimize().x
        return opt_output

    def log_prior(self, x):
        if self.prior is None:
            return 0
        if callable(self.prior):
            return self.prior(x)
        for i, param in enumerate(x):
            if param < min(self.prior[i]) or param > max(self.prior[i]) or np.isnan(param):
                return -np.inf
        return 0
    
    def log_prob(self, x):
        ln_pri = self.log_prior(x)
        chi_sq = self.get_chi_sq(x)
        ln_prob = -0.5 * chi_sq + ln_pri
        if np.isnan(ln_prob):
            return -np.inf
        return ln_prob

    def log_prob_and_blob(self, x):
        ln_prob = self.log_prob(x)
        return ln_prob, x


def mcmc(optimizer, backend_fn, n_steps=5000, n_walkers=96, init_sigma=0.1, reset=True):
    n_dim = optimizer.ModelClass.N_PARAMS
    os.environ['OMP_NUM_THREADS'] = '1'
    backend = emcee.backends.HDFBackend(backend_fn)
    if reset:
        backend.reset(n_walkers, n_dim)
        emcee_x0 = np.random.normal(
            loc=0, scale=init_sigma, size=(n_walkers, n_dim)) + optimizer.x0
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(
            n_walkers, n_dim, optimizer.log_prob_and_blob, pool=pool, backend=backend, moves=[(emcee.moves.StretchMove(), 0.2), (emcee.moves.WalkMove(), 0.2), (emcee.moves.KDEMove(), 0.2), (emcee.moves.DEMove(), 0.2), (emcee.moves.DESnookerMove(), 0.2)])
        if reset:
            sampler.run_mcmc(emcee_x0, n_steps, progress=True)
        else:
            sampler.run_mcmc(None, n_steps, progress=True)

    print(np.mean(sampler.acceptance_fraction))

@utils.debug_function
def run_optimize(directories, ModelClass=lf_model.FiducialCLF, **optimizer_kwargs):
    for d in directories:
        meas_fn = d + '/meas.npz'
        optimizer = LFOptimizer(meas_fn, ModelClass, **optimizer_kwargs)
        output = optimizer.optimize()
        print(output.message, output.fun, output.x)
        if output.success:
            np.save(d + '/best.npy', output.x)
            if len(directories) == 1:
                return output.x
        else:
            print("FAILED", output)


def run_mcmc(directories, ModelClass=lf_model.FiducialCLF, reset=True, n_steps=5000, **optimizer_kwargs):
    for d in directories:
        meas_fn = d + '/meas.npz'
        optimizer = LFOptimizer(meas_fn, ModelClass, **optimizer_kwargs)
        mcmc(optimizer, d + '/mcmc.h5', n_steps=n_steps, reset=reset)
