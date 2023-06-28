# %load SpecWizard_Lines_tom.py
import numpy as np
import Phys
constants = Phys.ReadPhys()
#
from scipy.signal import convolve
from scipy.special import erf
from astropy.modeling.functional_models import Voigt1D
from scipy.special import voigt_profile as VoigtSciPy
import scipy.interpolate as interpolate

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
        self.naturalwidth = naturalwidth_kms   # natural line width        [km/s]
        
    def errfunc(self):
        # tabulated error function
        pix     = 1e-2
        vmax    = 100.
        npix    = int(vmax/pix)
        pix     = vmax / npix
        v       = np.arange(-npix, npix) * pix
        err     = 0.5 * (1 + erf(v))
        return v, err
    
    def IDinterpol(self, x, xp, fp, cumulative=True):
        ''' Interpolate the function fx(xp) to the points x, conserving the integral of fp.
            This assumes that both x and xp are equally spaced '''

        # extend x axis by one element
        dx   = x[-1] - x[-2]
        xnew = np.concatenate((x, [x[-1]+dx]))

        # compute culmulative sum of fp's and extend by one element
        if not cumulative:
            Fc   = np.concatenate(([0], np.cumsum(fp)))
        else:
            Fc   = np.concatenate(([0], fp))
        dX   = xp[-1]-xp[-2]
        Xc   = np.concatenate((xp, [xp[-1]+dX]))

        # interpolate cumulative sum
        fcnew = np.interp(xnew, Xc, Fc)

        # difference
        fnew  = (np.roll(fcnew, -1) - fcnew)[:-1]

        return fnew    

    def gaussian(self, column_densities = 0, b_kms = 0,vion_kms=0,Tions= 0):

        naturalwidth_kms = self.naturalwidth    # natural line width        [km/s]
        f_value      = self.f_value
        lambda0      = self.lambda0  # rest-wavelength           [cm]
        pix_kms      = self.pix_kms       # pixel size
        periodic     = self.periodic

        # line cross section times speed of light, c [cm^2 * cm/s]
        sigma        = self.constants["c"] * np.sqrt(3*np.pi*self.constants["sigmaT"]/8.) * f_value * lambda0
        
        # generate normalized error function
        verf, erf = self.errfunc()
        
        # extent velocity range
        pixel_velocity_kms = np.concatenate((self.v_kms - self.box_kms, self.v_kms, self.v_kms + self.box_kms))
        tau          = np.zeros_like(pixel_velocity_kms)
        densities    = np.zeros_like(pixel_velocity_kms)
        velocities   = np.zeros_like(pixel_velocity_kms)
        temperatures = np.zeros_like(pixel_velocity_kms)
        
        for column_density, b, vel,Tion in zip(column_densities, b_kms, vion_kms,Tions):
            if column_density >0:
                # scale b-parameter
                v_line = b * verf

                # interpolate, and convert velocity from km/s to cm/s
                g_int   = column_density * sigma * self.IDinterpol(pixel_velocity_kms - vel, v_line, erf, cumulative=True) / 1e5

                # add
                tau          += g_int
                densities    += g_int * column_density
                velocities   += g_int * vel
                temperatures += g_int * Tion            
        # normalize to pixel size
        tau /= self.pix_kms
        densities /= self.pix_kms
        velocities /= self.pix_kms
        temperatures /= self.pix_kms
        nint = self.npix
        
        if periodic:
            tau          = tau[0:nint] + tau[nint:2*nint] + tau[2*nint:3*nint]
            pixel_velocity_kms = pixel_velocity_kms[nint:2*nint] 
            densities    = densities[0:nint] + densities[nint:2*nint] + densities[2*nint:3*nint]
            velocities   = velocities[0:nint] + velocities[nint:2*nint] + velocities[2*nint:3*nint]
            temperatures = temperatures[0:nint] + temperatures[nint:2*nint] + temperatures[2*nint:3*nint]
            
        else:
            tau   = tau[nint:2*nint]
            pixel_velocity_kms = pixel_velocity_kms[nint:2*nint] 
            densities    = densities[nint:2*nint] 
            velocities   = velocities[nint:2*nint]
            temperatures = temperatures[nint:2*nint]            
        mask = tau > 0 
        #Normalize optical depth quantities 
        print(tau)
        densities[mask]     /=  tau[mask]
        velocities[mask]    /=  tau[mask]
        temperatures[mask]  /=  tau[mask]
        

        # compute total column density
        
        nh_tot = np.cumsum(tau)[-1] * 1.e5 * self.pix_kms / sigma
        spectrum = {'pixel_velocity_kms':pixel_velocity_kms,
            'optical_depth':tau,
            'optical_depth_densities':densities,
            'optical_depth_velocities':velocities,
            'optical_depth_temperatures':temperatures,
            'total_column_density':nh_tot}
        
        return spectrum
    

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
        