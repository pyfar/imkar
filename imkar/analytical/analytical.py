import numpy as np
import pyfar as pf

def rectangular(frequency_vector, phi_vector, width, length, height, c):
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
        Values between 0°-90° allowed.
    width : double
        Width of the rectangulars in m.
    length : double
        Length of the regtangulars in m.
    height : double
        Height of the profile structure.
    c : double
        Speed of sound with which the scattering coefficient is computed.
    Returns
    -------
    s : FrequencyData
        The analytical solution for the scattering coefficient of a 
        rectangular profile at the given incidences and frequencies.    

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
    #check inputs    
    if not isinstance(frequency_vector, np.ndarray) or\
            not isinstance(frequency_vector.item(0), float):
        raise TypeError("frequency_vector has to be an array of real numbers")
    if frequency_vector.ndim > 1:
        raise ValueError("frequency_vector has to be 1-dimensional")
    if not isinstance(phi_vector, np.ndarray) or \
            not isinstance(phi_vector.item(0), float):
        raise TypeError("phi_vector has to be an array of real numbers")
    if phi_vector.ndim > 1:
        raise ValueError("phi_vector has to be 1-dimensional")
    if (np.any(phi_vector <= 0)) or (np.any(phi_vector >= 90)):
        raise ValueError("phi_vector values have to be between 0° and 90°")
    if not isinstance(width, float) or width <= 0:
        raise TypeError("width has to be a real number >0")
    if not isinstance(length, float) or length <= 0:
        raise TypeError("length has to be a real number >0")
    if not isinstance(height, float) or height <= 0:
        raise TypeError("height_to_length has to be a real number >0")

    # Initialization
    phi_vector = phi_vector*np.pi/180 #from degree to radiant
    #vector of corresponding wavelength in air
    lambda_air = c/frequency_vector
    k = 2*np.pi/lambda_air #vector of wavenumbers
    eps = 1E-3 # stop criterion for infinite sum over u
    n_max = 50 # truncation parameter for numbers of outgoing waves
    n_max2 = 2*n_max+1 #counter from -N to N including zero

    x_u = np.zeros((1, n_max),dtype=complex)
    n = np.arange(-n_max,n_max+1)[:, np.newaxis]
    previous_frequency = 0

    n_frequency = frequency_vector.size # number of computed frequencies
    n_phi = phi_vector.size # number of computed incident angles
    s = np.zeros((n_frequency, n_phi))

    #calculation
    for iFrequencies in range(0, n_frequency):
        for iPhi in range(0, n_phi):
            phi_0 = phi_vector[iPhi]
            alpha_n = np.cos(phi_0) + n*lambda_air[iFrequencies]/length
            beta_n = np.conj(-(1j*np.emath.sqrt(alpha_n**2-1)))            

            #helping variables
            k_frequency = k[iFrequencies]
            k_alpha_n = (k_frequency*alpha_n[:]).T
            jk_alpha_n_over_width = 1j*k_alpha_n/width

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
                            ((iFrequencies+1)>previous_frequency)):
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
                            if np.sum(CaseA) > 0:                           
                                U_u_n[np.nonzero(CaseA)] = 0.5*np.exp\
                                    (1j*uVals[np.nonzero(CaseA)]*np.pi/2)
                            if np.sum(CaseA) > 0:
                                U_u_n[np.nonzero(CaseB)] = 0.5*np.exp\
                                    (-1j*uVals[np.nonzero(CaseB)]*np.pi/2)
                            tmp = (jk_alpha_n_over_width/\
                                (k_x_u**2-k_alpha_n**2))*((np.exp(1j*k_alpha_n\
                                    *width/2))-((-1)**uVals*(np.exp(-1j\
                                        *k_alpha_n*width/2))))                       
                            U_u_n[np.nonzero(CaseC)] = tmp[np.nonzero(CaseC)]

                        tempU = np.sum([(np.sign(iU)+1)*x_u*np.tanh(1j*x_u*height)\
                            *np.conjugate(U_u_n[iU,iN-1])*U_u_n[iU,iR-1]]\
                                ,axis=1)

                        err = np.abs((prevTempU-tempU)/prevTempU)
                        prevTempU = tempU
                        previous_frequency = iFrequencies+1                    

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
            s[iFrequencies, iPhi] = 1-np.abs(R_n[0][n_max])**2
    return pf.FrequencyData(np.transpose(s), frequency_vector)

