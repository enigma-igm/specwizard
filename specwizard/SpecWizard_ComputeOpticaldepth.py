import importlib
import numpy as np
from specwizard.SpecWizard_Elements import Elements
import specwizard.SpecWizard_Lines

specwizard.SpecWizard_Lines = importlib.reload(specwizard.SpecWizard_Lines)
from specwizard.SpecWizard_Lines import Lines
import specwizard.Phys
constants = specwizard.Phys.ReadPhys()
#
import scipy.interpolate as interpolate

class ComputeOpticaldepth:
    ''' Methods to compute optical depth as a function of velocity for a single sight line '''
    def __init__(self, sightlineprojection):
        self.specparams = sightlineprojection
        self.periodic   = self.specparams['extra_parameters']['periodic']
        
        # for each of the ions, determine rest-wavelength and f-value of the first transition
        self.elements = self.specparams["elementparams"]["ElementNames"]

        #
        self.transitions = self.specparams["ionparams"]["transitionparams"]
        
        #
        self.constants  = specwizard.Phys.ReadPhys()
        
        self.ThermEff   = self.specparams['ODParams']['ThermalEffectsOff']
        self.PecVelEff  = self.specparams['ODParams']['PecVelEffectsOff']
        self.VoigtOff   = self.specparams['ODParams']['VoigtOff']
    def MakeAllOpticaldepth(self, projected_los):
        '''
            Apply MakeOpticaldepth to compute optical depths for all desired ionic transitions

            Input:
                    projected_los (dict): Dictionary containing all the projected data for the sightline output by
                                            SightLineProjection.ProjectData().

        '''
        
        projection               = {}
        projection["SightInfo"]  = self.specparams["sightline"]
        projection["Header"]     = self.specparams["Header"]
        projection["Projection"] = projected_los
        
        # header information from the snapshot
        self.header = projection["Header"]
        
        # pixel properties
        pixel_kms = self.ToCGS(self.header, projection["Projection"]["pixel_kms"]) / 1e5  # one pixel in km/s
        pixel     = self.ToCGS(self.header, projection["Projection"]["pixel"]) # size of pixel in cm
        sight_kms = self.ToCGS(self.header, projection["SightInfo"]["sightkms"]) / 1e5 
        projection["SightInfo"]["Sight_kms"] = sight_kms 
        
        self.sightinfo = projection["SightInfo"]
        # add some extra space of length dv_kms to start and end of spectrum
        # Note: first pixel has vel_kms[0]=0, last pixel has vel_kms[-1]=box_kms-pixel_kms
        vel_kms = np.arange(projection["Projection"]["npix"] ) * pixel_kms
        vunit   = self.SetUnit(vardescription='Hubble velocity',
                                       Lunit=1e5, aFact=0, hFact=0)
        npix    = len(vel_kms)
        #
        
        sightparams = {}
        sightparams['sight_kms'] = sight_kms
        sightparams['vel_kms']   = vel_kms
        sightparams['pixel_kms'] = pixel_kms
        sightparams['pixel']     = pixel
             
        Ions        = self.specparams["ionparams"]["Ions"]
        projectionIW  = projection["Projection"]["Ion-weighted"]
        projectionMW  = projection["Projection"]["Mass-weighted"]
        
        extend      = self.specparams['sightline']['ProjectionExtend']
        if extend["extend"]:
            extended_factor  = extend["extendfactor"]
            extended_npix    = npix * extended_factor
            extended_vel_kms = np.arange(extended_npix) * pixel_kms
            start_indx       =  int(0.5 * npix * (extended_factor - 1))
            for ion in projectionIW.keys():
                for key in ['Densities', 'Velocities', 'Temperatures']:
                    temp_array               = np.zeros_like(extended_vel_kms)
                    #nnpix = len(projectionIW[ion][key]['Value'])
                    
                    temp_array[start_indx:start_indx+npix]  = projectionIW[ion][key]['Value'].copy()
                    projectionIW[ion][key]['Value'] = temp_array

            if 'SimIon-weighted' in projection["Projection"].keys():
                projectionSimIon = projection["Projection"]['SimIon-weighted']
                for ion in projectionSimIon.keys():
                    for key in ['Densities', 'Velocities', 'Temperatures']:
                        temp_array               = np.zeros_like(extended_vel_kms)
                        #nnpix = len(projectionIW[ion][key]['Value'])
                        
                        temp_array[start_indx:start_indx+npix]  = projectionSimIon[ion][key]['Value'].copy()
                        projectionSimIon[ion][key]['Value'] = temp_array

            sightparams['vel_kms']   = extended_vel_kms
            sightparams['sight_kms'] = extended_vel_kms.max()
                        
                    
        spectra  = self.WrapSpectra(Ions,projectionIW,projectionMW,sightparams,vel_mod=False,therm_mod=False)
        #return spectra
        if self.ThermEff:

            spectra['ThermalEffectsOff']      = self.WrapSpectra(Ions,projectionIW,projectionMW,sightparams,
                                                                 vel_mod=False,therm_mod=True)

        if self.PecVelEff:
            spectra['PeculiarVelocitiesOff']  = self.WrapSpectra(Ions,projectionIW,projectionMW,sightparams,
                                                                 vel_mod=True,therm_mod=False)
        
        if self.ThermEff and self.PecVelEff:
            spectra['ThermalPecVelOff']       = self.WrapSpectra(Ions,projectionIW,projectionMW,sightparams,
                                                                 vel_mod=True,therm_mod=True)

        DoSimIons   = False
        
        try:
            projectionSIW = projection["Projection"]['SimIon-weighted']
            if extend["extend"]:
                projectionSIW = projectionSimIon
            SimIons       = list(projectionSIW.keys())
            DoSimIons     = True
        except:
            pass
            
        if DoSimIons:

            SimIons   = np.array(SimIons)
            all_ions      = np.array(Ions)[:,1]
            intsc     = np.in1d(all_ions,SimIons)
            SimElmsIons  = np.array(Ions)[intsc]
            SimElmsIons = [tuple(SimElmsIons[i]) for i in range(len(SimElmsIons))]
            spectra["SimIons"] =  self.WrapSpectra(SimElmsIons,projectionSIW,projectionMW,sightparams,vel_mod=False,
                                                   therm_mod=False)
            
            if self.ThermEff:
                spectra["SimIons_ThermalEffectsOff"]  = self.WrapSpectra(SimElmsIons,projectionSIW,projectionMW,
                                                                         sightparams,vel_mod=False,therm_mod=True)
        
            if self.PecVelEff:
                spectra['SimIons_PeculiarVelocitiesOff']  = self.WrapSpectra(SimElmsIons,projectionSIW,projectionMW,
                                                                             sightparams,vel_mod=True,therm_mod=False)

            if self.ThermEff and self.PecVelEff:
                spectra['SimIons_ThermalPecVelOff']   = self.WrapSpectra(SimElmsIons,projectionSIW,projectionMW,
                                                                         sightparams,vel_mod=True,therm_mod=True)
            
        return spectra
    
    def WrapSpectra(self,Ions,projection,projection_mass,sightparams,vel_mod=False,therm_mod=False ):
        """
            Wrapper for the MakeOpticaldepth method.

            Inputs:
                Ions (list): List of ions to include in the computed optical depths.
                projection (dict): Dictionary containing the ion-weighted fields.
                projection_mass (dict): Dictionary containing the mass-weighted fields.
                sightparams (dict): Dictionary containing sightline information.
                vel_mod (bool): If True, set the peculiar velocities to zero.
                therm_mod (bool): If True, set the temperatures to 0.1 K.

            Returns:
                spectra (dict): Dictionary containing the computed optical depths.
        """
        header     = self.header
        spectra    = {}
        vel_kms    = sightparams['vel_kms']
        vunit   = self.SetUnit(vardescription='Hubble velocity',
                               Lunit=1e5, aFact=0, hFact=0)
        for ion in Ions:
            (element_name, ion_name) = ion
            weight  = self.transitions[ion_name]["Mass"] * self.constants["amu"]
            lambda0 = self.transitions[ion_name]["lambda0"]
            f_value = self.transitions[ion_name]["f-value"]
            if lambda0 > 0:
                #
                nions    = self.ToCGS(header, projection[ion_name]["Densities"]) / weight
                vions    = self.ToCGS(header, projection[ion_name]["Velocities"]) / 1e5
                Tions    = self.ToCGS(header, projection[ion_name]["Temperatures"])
                dbary    = self.ToCGS(header, projection_mass["Densities"])
                vbary     = self.ToCGS(header, projection_mass["Velocities"]) / 1e5
                Tbary     = self.ToCGS(header, projection_mass["Temperatures"])
                if vel_mod:
                    vions = np.zeros_like(vions)
                if therm_mod:
                    Tions = np.zeros_like(Tions)+0.1
                spectrum = self.MakeOpticaldepth(
                    sightparams=sightparams,
                    weight=weight, lambda0=lambda0, f_value=f_value,
                    nions=nions, dbary=dbary, vions_kms=vions, vbary=vbary, Tions=Tions, Tbary=Tbary,
                    element_name = element_name)

                spectra[ion]                    = {}
                spectra[ion]["Velocities"]      = {'Value': vel_kms, 'Info': vunit}
                spectra[ion]["Optical depths"]  = spectrum["Optical depths"]
                spectra[ion]["Densities"]       = spectrum["Densities"]
                spectra[ion]["Baryon Densities"] = spectrum["Baryon Densities"]
                spectra[ion]["Velocities"]      = spectrum["Velocities"]
                spectra[ion]["Baryon Velocities"] = spectrum["Baryon Velocities"]
                spectra[ion]["Temperatures"]    = spectrum["Temperatures"]
                spectra[ion]["Baryon Temperatures"] = spectrum["Baryon Temperatures"]
                spectra[ion]["IonColumnDensity"] = spectrum["IonColumnDensity"]
                spectra[ion]["Mass"]            = weight
                spectra[ion]["lambda0"]         = lambda0
                spectra[ion]["f-value"]         = f_value
        return spectra 
    def MakeOpticaldepth(self, sightparams = [0.0,[0.0],1.0,1.0],
                     weight=1.67382335232e-24, lambda0=1215.67, f_value=0.4164, 
                     nions = [0.0], dbary=[0.0], vions_kms = [0.0], vbary=[0.0], Tions = [0.0],
                         Tbary=[0.0], element_name = 'Hydrogen'):

        ''' Compute optical depth for a given transition, given the ionic density, temperature and peculiar velocity

            Inputs:
                sightparams (dict): Dictionary containing sightline information.
                weight (float): Atomic mass of the ion in g.
                lambda0 (float): Rest wavelength of the transition in Angstroms.
                f_value (float): Oscillator strength of the transition.
                nions (array): Array containing the ionic density in ions/cm^3.
                dbary (array): Array containing the baryon density in g/cm^3.
                vions_kms (array): Array containing the peculiar velocity of the ions in km/s.
                vbary (array): Array containing the peculiar velocity of the baryons in km/s.
                Tions (array): Array containing the temperature of the ions in K.
                Tbary (array): Array containing the temperature of the baryons in K.
                element_name (str): Name of the element to which the transition belongs.

            Returns:
                Dictionary containing the computed optical depths, as well as optical-depth weighted averages
                of all of the input arrays.

        '''

        box_kms    = sightparams['sight_kms']
        vel_kms    = sightparams['vel_kms']
        pixel_kms  = sightparams['pixel_kms']
        pixel      = sightparams['pixel']
        npix         = len(vel_kms)
        tau          = np.zeros_like(vel_kms)
        densities    = np.zeros_like(vel_kms)
        baryon_density = np.zeros_like(vel_kms) # tau weighted baryon density (all species)
        velocities   = np.zeros_like(vel_kms)  
        temperatures = np.zeros_like(vel_kms)

        
        # passing the extent of the box in km/s introduces periodic boundary conditions
        lines = Lines(v_kms = vel_kms, box_kms=box_kms, constants = self.constants, verbose=False, 
                 lambda0_AA=lambda0, f_value=f_value,periodic=self.periodic)
            
        # convert from density to column density
        ioncolumns = nions * pixel                                           # in ions/cm^2
        dunit        = self.SetUnit(vardescription="Total ion column density", 
                                             Lunit=1.0, aFact=0.0, hFact=0.0)
        column_density    = {'Value': ioncolumns, "Info": dunit} # in ions/cm^2

        # compute b-parameter
        bions_kms = np.sqrt(2*self.constants["kB"]*Tions/weight) / 1e5
        
        # add Hubble velocity to peculiar velocity
        vHubble_kms    = box_kms * np.arange(len(vions_kms)) / len(vions_kms)
        voffset_kms    = self.specparams['ODParams']['Veloffset']  #Default = 0 
        vions_tot_kms  = vions_kms + vHubble_kms + voffset_kms
        spectrum = lines.gaussian(column_densities = ioncolumns, baryon_densities=dbary, b_kms = bions_kms, vion_kms=vions_kms,
                                  vion_tot_kms=vions_tot_kms,baryon_velocities=vbary,Tions=Tions, baryon_temperatures=Tbary)

        tau        = spectrum['optical_depth']
        densities  = spectrum['optical_depth_densities']
        velocities = spectrum['optical_depth_velocities']# - vHubble_kms # trying subtracting Hubble velocity...
        temperatures = spectrum['optical_depth_temperatures']
        baryon_density = spectrum['optical_depth_baryon_densities']
        baryon_velocities = spectrum['optical_depth_baryon_velocities']
        baryon_temperatures = spectrum['optical_depth_baryon_temperatures']
        pixel_velocities = spectrum['pixel_velocity_kms']

        # optical depth-weighted quantities
        if (not self.VoigtOff) and (element_name=="Hydrogen"):
#            print("this is happening")
            tau = lines.convolvelorentz(tau)
        #
        dunit        = self.SetUnit(vardescription="Tau weighted ion mass density", 
                                             Lunit=1.0, aFact=0.0, hFact=0.0)
        #densities   *= weight # mass density! ccd -> nope, that's not what it is! it's like a mass column density...
        densities    = {'Value': densities, "Info": dunit}

        bdunit        = self.SetUnit(vardescription="Tau weighted baryon mass density",
                                                Lunit=1.0, aFact=0.0, hFact=0.0)
        baryon_density = {'Value': baryon_density, "Info": bdunit}

        vunit        = self.SetUnit(vardescription="Tau weighted ion peculiar velocity",
                                             Lunit=1e5, aFact=0.0, hFact=0.0)
        velocities   = {'Value': velocities, 'Info': vunit}

        vdunit = self.SetUnit(vardescription="Tau weighted baryon peculiar velocity",
                             Lunit=1e5, aFact=0.0, hFact=0.0)
        baryon_velocities = {'Value': baryon_velocities, 'Info': vdunit}

        tunit        = self.SetUnit(vardescription="Tau weighted ion temperature",
                                             Lunit=1, aFact=0.0, hFact=0.0)
        temperatures = {'Value': temperatures, 'Info': tunit}

        tdunit = self.SetUnit(vardescription="Tau weighted baryon temperature",
                             Lunit=1, aFact=0.0, hFact=0.0)
        baryon_temperatures = {'Value': baryon_temperatures, 'Info': tdunit}

        tauunit      = self.SetUnit(vardescription="Ionic optical depth", 
                                             Lunit=1, aFact=0.0, hFact=0.0)
        tau          = {'Value':tau, 'Info':tauunit}

        return {'Optical depths':tau, 'Densities': densities, 'Baryon Densities': baryon_density,
                'Velocities': velocities, 'Baryon Velocities': baryon_velocities,
                'Temperatures': temperatures, 'Baryon Temperatures': baryon_temperatures,
                'IonColumnDensity':column_density}

    def CGSunit(self, header, variable):
        ''' 
        Use the information in the variable to compute the factor needed to convert
        simulation values to proper, h-free cgs units.
        This is of the form
        proper value = simulation value * CGSunit, where
        CGSunit = CGSConversionFactor * h**hexpo * a**aexpo
        '''
        #dependence on expansion factor
        ascale     = (1./(1+header["Cosmo"]["Redshift"]))**variable["Info"]["aexp-scale-exponent"]

        # dependence on hubble parameter
        hscale     = (header["Cosmo"]["HubbleParam"])**variable["Info"]["h-scale-exponent"]
        
        #
        return variable["Info"]["CGSConversionFactor"] * ascale * hscale
        
        
    def ToCGS(self, header, variable):
        ''' 
        return simulations values for this variable in proper cgs units (no h)
        '''
        return variable["Value"] * self.CGSunit(header, variable)

    def SetUnit(self, vardescription = 'text describing variable', Lunit=constants['Mpc'], aFact=1.0, hFact=1.0):
        return {'VarDescription': vardescription, 'CGSConversionFactor':Lunit, 'aexp-scale-exponent' :aFact, 'h-scale-exponent': hFact}
