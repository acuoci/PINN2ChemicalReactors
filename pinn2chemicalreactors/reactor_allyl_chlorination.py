import numpy as np
from math import exp, log

def reactor_allyl_chlorination(u,V, params):

    # Unknowns
    FCl2, FP, FA, FHCl, FD = u

    # Parameters/Constants
    teta, T, P = params
    A1 = exp(teta[0])
    A2 = exp(teta[1])
    E1 = teta[2]
    E2 = teta[3]

    # Total flow rate
    Ftot = FCl2 + FP + FA + FHCl + FD

    # Partial pressures
    pP = FP/Ftot*P
    pCl2 = FCl2/Ftot*P

    # Kinetic constants
    k1 = A1*np.exp(-E1/8.314/T)
    k2 = A2*np.exp(-E2/8.314/T)

    # Reaction rates
    r1 = k1*pP*pCl2
    r2 = k2*pP*pCl2

    # Formation rates
    RCl2 = -r1-r2
    RP   = -r1-r2
    RA   = r1
    RHCl = r1
    RD   = r2

    # Equations
    return [ RCl2, RP, RA, RHCl, RD]
