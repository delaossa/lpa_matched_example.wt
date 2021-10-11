import numpy as np
from scipy.constants import e, m_e, c, epsilon_0, mu_0, pi
from scipy.integrate import solve_ivp
from scipy.optimize import root

def linear_ramp(z, zp):
    return np.where(z<=0., 0., np.where(z<zp, z/zp, 1.))

# Plasma ramp - e.g. a cosine - of length zp, starting at z=0
def cosine_ramp(z, zp):
    return np.where(z<=0., 0., np.where(z<zp, 0.5*(1.-np.cos(np.pi * z/zp)), 1.))

def find_matching(rm=23e-6, lambda0=800e-9, zp=3e-3, ramp=cosine_ramp):

    # Laser wave number
    k = 2 * np.pi / lambda0

    # Laser waist evolution in vacuum
    def R_vac(z, zf, R0):
        zr = k/2 * R0**2
        return R0 * np.sqrt(1 + ((z-zf)/zr)**2)

    # derivative of vacuum evolution
    def R_vac_p(z, zf, R0):
        zr = k/2 * R0**2
        return R0 * (z-zf) / (zr**2 * np.sqrt(1 + ((z-zf)/zr)**2))


    # Laser waist equation (in SI units):
    #
    #   d^2 R / dz^2 = 4/(k^2 R^3) * (1 - (R/rm)^4)
    #
    # here written as two first-order diff. eqs. in the form requried for scipy's solver
    def waist_eq(z, R):
        return [R[1], 4./(k**2 * R[0]**3) * (1. - ramp(z, zp) * (R[0] / rm)**4)]

    # solve the differential equation backwards, assuming matching (R=rm, R'=0) behind the ramp
    sol = solve_ivp(waist_eq, (zp+1e-3, -1e-3), [rm, 0.], max_step=1e-5)

    # find the required focus position and laser waist by comparing the ODE results in front
    # of the ramp to the vacuum evolution 
    rootsol = root(lambda arg: [R_vac(sol.t[-1], arg[0], arg[1])-sol.y[0,-1], R_vac_p(sol.t[-1], arg[0], arg[1])-sol.y[1,-1]], [zp/2., rm])

    # ideal focus position
    zf_id = rootsol.x[0]
    # ideal laser waist
    w_id = rootsol.x[1]

    return zf_id, w_id