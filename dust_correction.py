import numpy as np

class DustCorrector:
    def __init__(self, delta_m=0.5):
        self.sig_beta = 0.34
        self.c0 = 4.54
        self.c1 = 2.07
        self.c = -2.33
        self.m0 = -19.5
        self.beta_m0 = {
            5 : -1.91,
            6 : -2,
            7 : -2.05,
            8 : -2.13,
            }
        self.dbeta_dm0 = {
            5 : -0.14,
            6 : -0.2,
            7 : -0.2,
            8 : -0.15,
            }
        self.delta_m_old = delta_m ### TODO: NOT ALWAYS 0.5!!

    def beta(self, m, z):
        return self.dbeta_dm0[z] * (m + 19.5) + self.beta_m0[z]

    def A_uv(self, m, z):
        try:
            beta_val = self.beta(m, z)
        except KeyError:
            return 0
        A_val = (self.c0 + 0.2*np.log(10)*self.sig_beta**2*self.c1**2
                + self.c1*beta_val)
        return max(A_val, 0)

    def m_new(self, m, z):
        return m - self.A_uv(m, z)

    def delta_m_new(self, m, z):
        return (self.delta_m_old + self.A_uv(m - self.delta_m_old/2, z)
                - self.A_uv(m + self.delta_m_old/2, z))

    def phi_new(self, phi, m, z):
        return phi * self.delta_m_old / self.delta_m_new(m, z)

    def sigma_new(self, sigma, m, z):
        return sigma * self.delta_m_old / self.delta_m_new(m, z)

    def dodc(self, points):
        pzs = points['z_vals']
        pms = points['mags']
        pvs = points['phis']
        pls = points['sig_minuses']
        phs = points['sig_pluses']
        points['mags'] = [self.m_new(pm, pz) for pm, pz in zip(pms, pzs)]
        points['phis'] = [self.phi_new(pv, pm, pz) for pv, pm, pz in zip(pvs, pms, pzs)]
        points['sig_minuses'] = [self.sigma_new(pl, pm, pz) for pl, pm, pz in zip(pls, pms, pzs)]
        points['sig_pluses'] = [self.sigma_new(ph, pm, pz) for ph, pm, pz in zip(phs, pms, pzs)]
        return points

