# %load SpecWizard_Lines_tom.py
import numpy as np
import specwizard.Phys
constants = specwizard.Phys.ReadPhys()
#
from scipy.signal import convolve
from scipy.special import erf
from astropy.modeling.functional_models import Voigt1D
from scipy.special import voigt_profile as VoigtSciPy
import scipy.interpolate as interpolate
from IPython import embed
import sys


# convolve(in1, in2, mode='full', method='auto')[source]
class Lines:
    ''' Methods to compute optical depth as a function of velocity for a single absorber,
        implementing:
        - a Gaussian profile
        - a Voigt profile
        '''

    def __init__(self, v_kms =0.0, box_kms=-1.0, constants=constants, lambda0_AA=1215.67, f_value=0.4164,
                 naturalwidth_kms=6.06076e-3,verbose=False, periodic=True):

        self.constants    = constants
        self.verbose      = verbose
        self.v_kms        = np.array(v_kms)     # velocity bins to evaluate optical depth [km/s]
        self.pix_kms      = v_kms[1] - v_kms[0] # pixel size
        self.periodic     = periodic
        self.box_kms      = box_kms             # velocity extent of spectrum
        self.npix         = len(self.v_kms)
        self.lambda0      = lambda0_AA * 1e-8  # rest-wavelength           [cm]
        self.f_value      = f_value            # oscillator strength       [dimensionless]
        self.naturalwidth = naturalwidth_kms   # natural line width        [km/s]self.
        self.sigma        = self.constants["c"] * np.sqrt(3*np.pi*self.constants["sigmaT"]/8.) * self.f_value * self.lambda0
    def errfunc(self):
        # tabulated error function
        pix     = 1e-2
        vmax    = 100.
        npix    = int(vmax/pix)
        pix     = vmax / npix
        v       = np.arange(-npix, npix) * pix
        err     = 0.5 * (1 + erf(v))
        return v, err
    
    def IDinterpol(self, x, xp, fp, cumulative=True, stop_flag=False):
        ''' Interpolate the function fx(xp) to the points x, conserving the integral of fp, and thereby the total
            optical depth. This assumes that both x and xp are equally spaced.

            Inputs:
                x (array): Difference in velocity between different bits of material and the sightline pixel.
                xp (array): b*verf, where b is Doppler b parameter (thermal broadening scale) and verf is the
                            array of velocity positions of the error function value.
                            verf corresponds to y in Lukic+15 appendix
                fp (array): The value of the error function at each xp, or velocity separation times the b parameter.
                cumulative (bool): If True, the input fp is the cumulative sum of the error function values.(?)
        '''

        # extend x axis by one element
        dx   = x[-1] - x[-2]
        xnew = np.concatenate((x, [x[-1]+dx]))

        # compute cumulative sum of fp's and extend by one element
        if not cumulative:
            Fc   = np.concatenate(([0], np.cumsum(fp)))
        else:
            Fc   = np.concatenate(([0], fp))
        dX   = xp[-1]-xp[-2]
        Xc   = np.concatenate((xp, [xp[-1]+dX]))

        # interpolate cumulative sum
        fcnew = np.interp(xnew, Xc, Fc)

        # difference to get the per pixel contribution
        fnew  = (np.roll(fcnew, -1) - fcnew)[:-1]

        return fnew    

    def gaussian(self, column_densities = 0, b_kms = 0, vion_tot_kms=0, realspace_quantities=None):

                 #baryon_densities = 0, b_kms = 0, vion_kms=0, vion_tot_kms=0,
                 #baryon_velocities = 0, Tions= 0, baryon_temperatures = 0, ion_metallicities = 0,
                 #ion_X_Carbon = 0):
        """
            Calculate the tau-weighted quantities for a Gaussian line profile, conserving the optical depth over the
            integral. Given the real space pixel quantities, loop over the redshift space pixels and assign tau. Also
            calculates and returns certain optical-depth weighted physical quantities.

            Inputs:
                realspace_quantities (list of lists): List of the real space quantities to be tau-weighted, with
                                                      dimension of Nquantities (rows) times Npixels.
        """
        naturalwidth_kms = self.naturalwidth    # natural line width        [km/s]
        f_value      = self.f_value
        lambda0      = self.lambda0  # rest-wavelength           [cm]
        pix_kms      = self.pix_kms       # pixel size
        periodic     = self.periodic

        # line cross section times speed of light, c [cm^2 * cm/s]
        sigma        = self.sigma # this is the sigma from Theuns et al. 1998. Lorentzian component is excluded.
        
        # generate normalized error function
        verf, erf = self.errfunc()
        
        # full extended sightline (3x the box size) for periodic boundary conditions
        # redshift space velocity position
        pixel_velocity_kms = np.concatenate((self.v_kms - self.box_kms, self.v_kms, self.v_kms + self.box_kms))

        # initialize redshift-space arrays at zero for tau and tau-weighted summation
        tau          = np.zeros_like(pixel_velocity_kms) # add small number to avoid division by zero

        # initialize redshift-space arrays at zero for the tau-weighted extra quantities
        redshift_quantities = np.zeros((len(realspace_quantities), len(pixel_velocity_kms)))

        # loop over real space quantities to find contributions to redshift-space pixels
        for ri in range(len(column_densities)): # real space
            if column_densities[ri] > 0:
                # scale b-parameter
                v_line = b_kms[ri] * verf
                # interpolate, and convert velocity from km/s to cm/s
                g_int = column_densities[ri] * sigma * self.IDinterpol(pixel_velocity_kms - vion_tot_kms[ri], v_line, erf,
                                                                 cumulative=True) / 1e5
                # add
                tau += g_int
                for qi in range(len(realspace_quantities)): # each field
                    redshift_quantities[qi] += realspace_quantities[qi][ri] * g_int
        zspace_output = np.row_stack((tau, redshift_quantities))
        # past the redshift space loop, normalize weighted to pixel size
        zspace_output /= pix_kms
        nint = self.npix

        if periodic:  # sum real space material contributions on 3x skewer segment
            zspace_output = zspace_output[:,0:nint] + zspace_output[:,nint:2*nint] + zspace_output[:,2*nint:3*nint]
            pixel_velocity_kms = pixel_velocity_kms[nint:2*nint]
        else:
            zspace_output = zspace_output[nint:2*nint]
            pixel_velocity_kms = pixel_velocity_kms[nint:2*nint]
        # normalize optical depth weighted quantities
        zspace_output = np.row_stack((zspace_output[0], zspace_output[1:]/zspace_output[0]))

        # compute total column density
        nh_tot = np.cumsum(zspace_output[0])[-1] * 1.e5 * self.pix_kms / sigma
        return zspace_output, nh_tot
    

    def directgauss(self, column_density = 0, b_kms = 0, lambda0_AA=1215.67, f_value=0.4164, naturalwidth_kms=6.06076e-3,periodic=True):
        ''' Direct Gaussian line profile evaluated at centre of each pixel '''

        # cross section [cm^3/s]
        sigma     = self.constants["c"] * np.sqrt(3*np.pi*self.constants["sigmaT"]/8.) * f_value * lambda0 # cross section cm^3/s        
        
        # extent velocity range
        velocity_kms = np.concatenate((self.v_kms - self.box_kms, self.v_kms, self.v_kms + self.box_kms))
        tau          = np.zeros_like(velocity_kms)
        
        #
        pix2         = 0.5*self.pix_kms  # 1/2 of the size of a pixel
        for norm, b, vel in zip(column_density * sigma, b_kms, self.v_kms):
            
            # scale b-parameter
            if norm > 0:
                # By convention, velocity_kms is the start of the pixel. 
                # We evaluate the optical depth at the centre of the pixel
                dv = vel - (velocity_kms + pix2)   

                gauss     = 1./np.sqrt(np.pi*(1e5*b)**2) * np.exp(-(dv)**2/b**2)
                gauss    *= norm
                
                #
                tau      += gauss

        #
                
        # apply periodic boundary conditions if required
        nint = self.npix
        if periodic:
            tau          = tau[0:nint] + tau[nint:2*nint] + tau[2*nint:3*nint]
            velocity_kms = velocity_kms[nint:2*nint]
            
        # compute total column density
        nh_tot = np.cumsum(tau)[-1] * 1.e5 * self.pix_kms / sigma
        
        return velocity_kms, tau, nh_tot

    
    def convolvelorentz(self, phi):
            ''' return convolution of input line profile with Lorentzian
            input: phi(v_z): line profile function as a function of velocity, v_z
            output: phi(v_z) input line profile now convolved with natural line profile
            '''
            # The core of the Lorentzian needs to be sampled to 0.01 km/s to get an accurate
            # convolution. Therefore we interpolate the original profile to a finer velocity grid
            # before performing the convolution.

            # Create velocity bins for interpolation
            dv         = 1e-3                   # pixel size in km/s
            vmin       = np.min(self.v_kms)
            vmax       = np.max(self.v_kms)
            nbins      = np.int((vmax-vmin)/dv)
            dv         = (vmax-vmin)/float(nbins)
            v_convolve = vmin + np.arange(nbins) * dv 

            # Create lorentz profile
            phi_fine    = np.interp(v_convolve, self.v_kms, phi)
            width       = self.naturalwidth
            v_bins      = v_convolve - np.mean(v_convolve)              # centre Lorenz at the centre of the velocity interval
            lorentz     = (1./np.pi) * width / (v_bins**2 + width**2)   
            lorentz     = lorentz / 1e5                                 # convert to units of [s/cm]

            # The integral over the Lorentzian is normalized to unity, so that
            # sum(lorentz) dpix = 1, or the pixel size = 1/np.sum(lorentz)
            phi_fine    = convolve(phi_fine, lorentz, mode='same') / np.sum(lorentz)

            #
            result      = np.interp(self.v_kms, v_convolve, phi_fine)

            return result    #
        
    def SciPyVoigt(self, b_kms=10., v0_kms=0.0, lambda0_AA=1215.67, f_value=0.4164, naturalwidth_kms=6.06076e-3, periodic=True):
        ''' 
        return Voigt line-profile function, Equation 5
        this version uses the SciPy implementation of the Voigt function
        input:
             b_kms (float):     b-parameter [km/s]
             v0_kms (float):    velocity at line centre [km/s]
        output: 
             line profile (float or array) : line shape with unit column density [s/cm]
        
        '''
        # extend velocity array
        velocity_kms = np.concatenate((self.v_kms - self.box_kms, self.v_kms, self.v_kms + self.box_kms))
        u            = velocity_kms / b_kms
        
        # 
        sigma_G     = 1.0 / np.sqrt(2.0)           # variance of Gaussian
        gamma_L     = naturalwidth_kms / b_kms     # Half-width half-maximum of Lorenzian - the parameter "a"
        
        # evaluate Voigt profile
        vnorm       = VoigtSciPy(u, sigma_G, gamma_L, out=None)
        phi         = vnorm / b_kms                # SciPy returns a normalized Voigt profile, which includes the 1/sqrt(pi)

        # impose periodic boundary conditions
        nint = self.npix        
        if periodic:
            phi          = phi[0:nint] + phi[nint:2*nint] + phi[2*nint:3*nint]
            velocity_kms = velocity_kms[nint:2*nint]
        
        return velocity_kms, phi/1e5  # convert to units of [s/cm]
    
    
    
    
    def sigmaHI(self,hnu_eV=13.6):
        ''' Fit from Verner et al ('96) fit to the photo-ionization cross section
        Input: energy of the photon in eV
        Output: photo-ionization cross section in cm^2 '''

        barn   = 1e-24
        sigma0 = 5.475e4 * 1e6 * barn
        E0     = 0.4298
        xa     = 32.88
        P      = 2.963
        #
        energy = np.array(hnu_eV)
        x      = energy/E0
        sigma  = sigma0 * (x-1)**2 * x**(0.5*P-5.5) * (1.+np.sqrt(x/xa))**(-P)
        if isinstance(sigma, (list, tuple, np.ndarray)):
            mask        = energy < 13.6
            sigma[mask] = 0
        else:
            if energy < 13.6:
                sigma = 0.0    
        return sigma

    def convolveLymanLimit(self,tau_Lymanalpha):
        ''' Return Lyman limit optical depth corresponding to input Lyman-alpha optical depth '''
        vel_kms    = self.v_kms           # pixel velocities [km/s]
        pix_kms    = self.pix_kms    # pixel width [km/s]
        npix       = self.npix       # number of pixels
        constants  = self.constants
        lambda0    = 1215.67e-8      # Lya wavelength [cm]
        f_value    = 0.4164          # Lya f-value

        # compute Lya cross section [cm^2 km/s]
        sigma_a    = np.sqrt(3*np.pi*constants["sigmaT"]/8.) * f_value * lambda0 * (constants["c"]) / 1e5

        # generate velocity grid for convolution

        # use finer pixels
        pix     = 0.1 # pixel size in km/s
        npix    = int(np.max(vel_kms) / pix)
        pix     = np.max(vel_kms) / npix
        vel     = np.arange(-npix, npix) * pix
        tau     = np.interp(vel, vel_kms, tau_Lymanalpha)
        #hnu_eV  = 13.6 * (1. - vel * 1e5 / constants["c"])
        hnu_eV  = 13.6 * np.exp(-((vel*1e5)/constants["c"]))
        sigma   = self.sigmaHI(hnu_eV=hnu_eV)
        tau_LL  = convolve(tau/sigma_a, sigma, mode='same')
        tau_LL *= pix

        #
        result  = np.interp(vel_kms, vel, tau_LL)

        #
        return result    
        
