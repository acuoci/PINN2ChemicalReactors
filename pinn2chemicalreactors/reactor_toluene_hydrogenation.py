def reactor_toluene_hydrogenation(u,t, params):

    CA, CB, CC = u
    kappa = params[0]

    # Assigned parameters
    kH1 = kappa[0]
    kD1 = kappa[1]
    k2 = kappa[2]
    KrelA = kappa[3]
    KrelB = 1
    KrelC = kappa[4]

    # Surface coverage
    denominator = KrelA*CA + KrelB*CB + KrelC*CC
    tetaA = KrelA*CA/denominator
    tetaB = KrelB*CB/denominator

    # Reaction rates
    rH1 = kH1*tetaA
    rD1 = kD1*tetaB
    r2 = k2*tetaB

    RA = -rH1+rD1
    RB =  rH1-rD1-r2
    RC =  r2

    return [ RA,  RB, RC ]
