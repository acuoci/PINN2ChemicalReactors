"""
Author:  Alberto Cuoci
Problem: Identification of kinetic parameters, CSTR
         Acetoacetylation, 3 reactions, 5 species (A,B,C,D,E), 5 measured species (A,B,C,D,E)
         Reactions:
         (1) A+B -> C  r1=k1*CA*CB*Ccat  k1=0.053 (l2/mol2/min)
         (2) 2B -> D   r2=k2*CB*CB*Ccat  k2=0.128 (l2/mol2/min)
         (3) B -> E    r3=k3*CB          k3=0.028 (1/min)
"""

def reactor_aceto_acetylation(u,t, params):

    CA, CB, CC, CD, CE = u
    k, Cin, V, Qin, Ccat = params

    r1 = k[0]*CA*CB*Ccat
    r2 = k[1]*CB*CB*Ccat
    r3 = k[2]*CB

    RA = -r1
    RB = -r1-2.*r2-r3
    RC = r1
    RD = r2
    RE = r3

    tau = V/Qin

    return [ (Cin[0]-CA)/tau+RA,  (Cin[1]-CB)/tau+RB, (Cin[2]-CC)/tau+RC, (Cin[3]-CD)/tau+RD, (Cin[4]-CE)/tau+RE]
