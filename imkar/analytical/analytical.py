import numpy as np
import pyfar as pf

from imkar import scattering

def rectangular(frequency_vector, phi_vector, width, length, height_to_length):
    """
    This function computes the scattering coefficient of periodic rectangular
    profiles, assuming that the extent is infinite.

    This function is based on Ducourneau's method described by Prof. Embrechts
    [#]_.

    Parameters
    ----------
    frequency_vector : array, double
        Vector of frequencies the scattering coefficient is computed at.
    phi_vector : array, double
        Vector of incidences phi the scattering coefficient is computed at.
    width : double
        Width of the rectangulars.
    length : double
        Length of the regtangulars.
    height_to_length : double
        Height to length ratio of the rectangulars.

    Returns
    -------
    s : FrequencyData
        The analytical solution for the scattering coefficient of a 
        rectangular profile at the given incidences and frequencies.
    s_rand : Frequency Data
        The analytical solution for the scattering coefficient of a
        rectangular profile at a random incidence.

    References
    ----------
    .. [#]J. Embrechts and A. Billon, "Theoretical Determination of the Random-
    Incidence Scattering Coefficients of Infinite Rigid Surfaces with a 
    Periodic Rectangular Roughness Profile", Acta Acustica united with Acustica,
    Bd. 97, Nr. 4, S. 607-617, Juli 2011, doi:10.3813/AAA.918441

    Comment
    -------
    In this solution the parameter r in equation (9) is also limited to 
    -N <= r <= N. The width of the well is described by the width parameter and
    replaces the parameter 2A.

    """
    # Initialization
    phi_vector = phi_vector*np.pi/180 #from degree to radiant
    c = 343.901 # ITA constant
    h = height_to_length*length #height of the profile
    #vector of corresponding wavelength in air
    lambda_air = c/frequency_vector[:] 
    k = 2*np.pi/lambda_air #vector of wavenumbers

    eps = 1E-3 # stop criterion for infinite sum over u
    n_max = 50 # truncation parameter for numbers of outgoing waves
    n_max2 = 2*n_max+1 #counter from -N to N including zero

    x_u = np.zeros((1, n_max),dtype=complex)
    n = np.arange(-n_max,n_max+1)[:, np.newaxis]
    #index_0 = 51 # index for R_0
    previous_frequency = 0

    n_frequency = frequency_vector.size # number of computed frequencies
    n_phi = phi_vector.size # number of computed incident angles
    s = np.zeros((n_frequency, n_phi))

    #calculation
    for iFrequencies in range(1, (n_frequency+1)):
        for iPhi in range(1, (n_phi+1)):
            phi_0 = phi_vector[iPhi-1]
            alpha_n = np.cos(phi_0) + n*lambda_air[iFrequencies-1]/length
            beta_n = np.conj(-(1j*np.emath.sqrt(alpha_n**2-1)))
            #np.sin(np.emath.arccos(alpha_n))            

            #helping variables
            k_frequency = k[iFrequencies-1]
            k_alpha_n = (k_frequency*alpha_n[:]).T
            jk_alpha_n_over_width = 1j*k_alpha_n/width
            #expkAlpha_n_width = (np.exp(1j*kAlpha_n*width))

            y_nr = np.zeros((n_max2, n_max2),dtype=complex)

            for iN in range(1, (n_max2+1)):
                u_max = np.zeros((1, n_max2),dtype=int)
                u_max_element = 0
                for iR in range(1, (n_max2+1)):
                    u_max[0,iR-1] = 0
                    prevTempU = 0
                    err = 1
                    while err>eps:
                        u_max[0,iR-1] = u_max[0,iR-1] + n_max2-1

                        if iR==1:
                            U_u_n = np.zeros((u_max.item((iR-1))+1,n_max2),\
                                dtype=complex)
                        elif u_max[0,iR-1]>u_max_element:
                            U_u_n = np.zeros((u_max.item((iR-1))+1, n_max2),\
                                dtype=complex) 
                        
                        if ((iR==1) or (u_max[0,iR-1]>u_max_element) or \
                            (iFrequencies>previous_frequency)):
                            iU = np.arange(0., round(u_max.item((iR-1)))+1,\
                                dtype=int)[:, np.newaxis]                           
                            uVals = np.tile(iU, (1, n_max2))
                            k_x_u = iU*np.pi/width
                            x_u = np.emath.sqrt(k_frequency**2 - k_x_u**2)

                            CaseA = np.less(abs((k_alpha_n-k_x_u)), 1E-3)
                            CaseA = CaseA.astype(int)
                            CaseB = np.less(abs((k_alpha_n+k_x_u)), 1E-3)
                            CaseB = CaseB.astype(int)
                            CaseC = np.ones(np.shape(CaseA))-(CaseA+CaseB)
                            CaseC = CaseC.astype(int) 
                            if(np.sum(CaseA)>0):                           
                                U_u_n[np.nonzero(CaseA)] = 0.5*np.exp\
                                    (1j*uVals[np.nonzero(CaseA)]*np.pi/2)
                            if(np.sum(CaseB)>0):
                                U_u_n[np.nonzero(CaseB)] = 0.5*np.exp\
                                    (-1j*uVals[np.nonzero(CaseB)]*np.pi/2)
                            tmp = (jk_alpha_n_over_width/\
                                (k_x_u**2-k_alpha_n**2))*((np.exp(1j*k_alpha_n\
                                    *width/2))-((-1)**uVals*(np.exp(-1j\
                                        *k_alpha_n*width/2))))                       
                            U_u_n[np.nonzero(CaseC)] = tmp[np.nonzero(CaseC)]

                        tempU = np.sum([(np.sign(iU)+1)*x_u*np.tanh(1j*x_u*h)\
                            *np.conjugate(U_u_n[iU,iN-1])*U_u_n[iU,iR-1]]\
                                ,axis=1)

                        err = np.abs((prevTempU-tempU)/prevTempU)
                        prevTempU = tempU
                        previous_frequency = iFrequencies                        

                        if u_max_element<u_max.item((iR-1)):
                            u_max_element = u_max.item((iR-1))
                        
                    y_nr[iN-1,iR-1] = tempU*width/(k_frequency*length)

            b = np.sin(phi_0)*(2-(abs(np.sign(n))+1)) + ((y_nr[:,n_max])\
                .reshape(n_max2,1))
            A = np.diagflat(beta_n) -y_nr

            R_n = np.linalg.lstsq(A,b, 1E-6)
            validIds = 1*(np.abs(alpha_n)<=1)
            checksum = np.sum((np.abs(R_n[0][np.nonzero(validIds)])**2)\
                *beta_n[np.nonzero(validIds)]/np.sin(phi_0))
            absError = np.abs(checksum-1)
            print(f'The absolute error is {absError}.')
            s[iFrequencies-1, iPhi-1] = 1-np.abs(R_n[0][n_max])**2
            #print(s)
    return pf.FrequencyData(np.transpose(s), frequency_vector)

