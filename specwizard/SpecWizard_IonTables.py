#%load SpecWizard_IonTables.py
#%%writefile SpecWizard_IonTables.py

import sys
import h5py
import re
import numpy as np 
import scipy.interpolate as interpolate
from IPython import embed


class IonTables:
    """
    This class reads and interpolates the ionization tables that will be used for the calculation of ion fractions. 

    Parameters
    ----------

    specparams : dictionary
        The Wizard dictionary produced by the BuildInput routine. 
    """

    def __init__(self, specparams = None):
        self.specparams   = specparams
        self.iontable_info = specparams['ionparams']

        
    def ReadIonizationTable(self, ion='H I'):
        
        """
        This will read the ionization tables for the interpolation. The specifics of the table (path,type) are provided by
        initializing the class by providing the output from the Build_Input.
	
        Parameters
            ----------

                ion : str
                  Name of ion to be calculated e.g 'HeII'

        Returns
        -------

                out : tuple
                    tuple with the contents of the ionization table for that ion.
                            if 'specwizard_cloudy' -> ((redshift,Log_temperature,Log_nH),Log_Abundance)
                            if 'hm01_cloudy' -> ((redshift,Log_temperature,Log_nH,Z/Zsol),Log_Abundance)
                            if 'ploeckinger'       -> ((redshift,Log_temperature,Log_(Z/Zsol),Log_nH),Log_Abundance)
        """     
        iontable_info = self.iontable_info
        if iontable_info["table_type"] == 'specwizard_cloudy':
            # check whether ion exists
            if not ion in iontable_info['ions-available']:
                print("Ion",ion," is not found")
                sys.exit(-1)
            fname = iontable_info['iondir'] + '/' + ion + '.hdf5'
            hf    = h5py.File(fname,  "r")
                
            # cloudy parameters
            UVB   = hf.attrs.get("UVB")
            compo = hf.attrs.get("Composition")
            
            # tables
            z            = hf['z'][...]
            LogT         = hf['LogT'][...]
            LognH        = hf['LognH'][...]
            LogAbundance = hf['LogAbundance'][...]
            #
            hf.close()
            #
            return ((z, LogT, LognH), LogAbundance)

        if iontable_info["table_type"] == 'hm01_cloudy':
            # check whether ion exists
            if not ion in iontable_info['ions-available']:
                print("Ion", ion, " is not found")
                sys.exit(-1)

            el_abbr = (ion.split(' ')[0]).lower()
            number = self.reverseRomanNumeral((ion.split(' ')[1]).strip())
            fname = f"{iontable_info['iondir']}/{el_abbr}{number}.hdf5"

            hf = h5py.File(fname, "r")

            # cloudy parameters
            try:
                UVB = hf.attrs.get("UVB")
                compo = hf.attrs.get("Composition")
            except:
                print('No UVB or Composition in the file.')
            # tables
            z = hf['redshift'][...]
            LogT = hf['logt'][...]
            LognH = hf['logd'][...]
            Abundance = hf['ionbal'][...]  # not logged
            #
            hf.close()
            #
            return ((LognH, LogT, z), Abundance)

        if iontable_info["table_type"] == 'ploeckinger':

            file       = iontable_info["iondir"] + '/' + iontable_info["fname"]
            hf         = h5py.File(file,  "r")
            # read temperatre bins
            LogT       = hf["TableBins/TemperatureBins"][...]
            LognH      = hf["TableBins/DensityBins"][...]
            z          = hf["TableBins/RedshiftBins"][...]
            LogZ       = hf["TableBins/MetallicityBins"][...]
            # read and format the correct element
            # 
            element    = (''.join(re.split('I|  |V|X|', ion))).strip()
            
            #print("element to read = {}".format(element))
            # We create a dictionary between the element names (as provided to from the user) and
            # how they are called in the Ploeckinger tables e,g H -> 01hydrogen. 
            srtnames   = np.array(hf['ElementNamesShort'][...],dtype=str)
            ionnames   = np.array(list(hf['Tdep/IonFractions'].keys()),dtype=str)
            tablenames = dict(zip(srtnames,ionnames))            
            tablename  = tablenames[element]
            
 #           print("table name = ", tablename)
            # We use self.RomanNumeral to create a dictionary between Roman and decimal numerals
            # So we can know that the user input O VI -> O 6  
            dec_and_rom = np.array([[i,self.RomanNumeral(i)] for i in range(30)])
            rom2dec_dict = dict(zip(dec_and_rom[:,1],dec_and_rom[:,0]))
            rom_num       = ((ion).replace(element,"")).strip()
            ionlevel     = int(rom2dec_dict[rom_num]) - 1
 #           print("ion = ", ion, 'level =', ionlevel)
            #
            LogAbundance = hf['Tdep/IonFractions/' + tablename][:,:,:,:,ionlevel]
 #           LogAbundance = hf['Tdep/HydrogenFractionsVol'][:,:,:,:,0]
            #
            hf.close()
            #
            return ((z, LogT, LognH, LogZ), LogAbundance)
        

    def reverseRomanNumeral(self, s):
        """
        Converts roman numeral to decimal.

        Parameters
        """

        roman = {'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'VI': 6, 'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10}
        try:
            return roman[s]
        except KeyError:
            print('Requested Roman numeral not found in dictionary')
            sys.exit()


    def RomanNumeral(self, i):
        """
        Converts decimal to roman numerals . 

        Parameters
	----------

        i : int
	Decimal number

	Returns
	-------
	
        out : str
        Roman string e.g 'IV'
         
        """
        result = ''
        # number of Xs
        nX = int(i/10)
        for c in np.arange(nX):
            result += 'X'
        j  = i%10
        if j < 4:
            for c in np.arange(j):
                result += 'I'
        elif j==4:
            result += 'IV'
        elif j == 5:
            result += 'V'
        elif j == 9:
            result += 'IX'
        else:
            k = j - 5
            result += 'V'
            for c in np.arange(k):
                result += 'I'
        return result


    def SetLimitRange(self,values,min_val,max_val):
        """
        Set values < min_val to min_val and values > min_max to max_val. This to avoid extrapolation. 

        Parameters
        ----------

        values : np.array(float)
            Array with the values we want to limit

        min_val : float 
            Minimal value range set by the table. 

        max_val : float 
            Maximal value range set by the table. 

        Returns 
        -------
        out : np.array(float)
            array with values bounded by the table.  
	"""
        values[values<min_val] = min_val
        values[values>max_val] = max_val
        return values
    
    def IonAbundance(self, redshift=0.0, nH_density=1.0, temperature=1e4, Z=1, ion = 'H I', use_Rahmati13=False):
        """ 
        Return the fractional abundance of a given ion. This done by interpolation of the ionization tables with particle data from simulations.

            Parameters
            ----------

                redshift     : float Redshift of the particles.

                density:     : float Proper hydrogen density of gas  particles in $\mathrm{cm}^{-3}$

                temperature  : float  Temperature of  gas particles in $\mathrm{K}$

                metallicity  : float Total metallicity of the particles (only used for ploeckinger tables)

                ionname      : str= Name of ion,e.g. 'H I' for neutral hydrogen

            Returns

                out : float Ion fraction $\log_{10} \mathrm{n_{ion}/n_{element}}$
        """
        iontable_info = self.iontable_info

        if ion == 'H I' and use_Rahmati13:
            # Using the Rahmati et al. 2013 fitting formula for the neutral hydrogen fraction, rather than the tables.
            from enigma.tpecodes._tpecodes import Gamma_ss, get_x_HI

            try:
                # Use the HM01 HI photoionization rate
                if iontable_info["table_type"] == 'hm01_cloudy':
                    fname = iontable_info['iondir'] + '/h1.hdf5'
                    hf = h5py.File(fname, "r")
                    table_Gamma_HI = hf['header']['spectrum']['gammahi'][...] # s^-1
                    table_redshift = hf['header']['spectrum']['redshift'][...]
                    # TODO assumes constant redshift
                    interpd_Gamma_HI = np.interp(redshift[0], table_redshift, table_Gamma_HI)
                    He_fac = 1.0 + (1 - 0.76) / (2 * 0.76)  # assuming X=0.76 for hydrogen

                    # attenuated HI photoionization rate. Array as function of nH_density.
                    Gamma_phot = np.array([Gamma_ss(interpd_Gamma_HI/(1e-12), x) for x in nH_density])
                    x_HI = np.array([get_x_HI(He_fac, x[0], x[1], x[2]) for x in zip(Gamma_phot, nH_density, temperature)])

                    return np.log10(x_HI)
            except:
                embed(header='\nCould not use Rahmati+2013 for HI ionization state. Using the unshielded tables '
                             'instead.\n')

        if iontable_info["table_type"] == 'specwizard_cloudy':
            # read the table
            (table_z, table_LogTs, table_LognHs), table = self.ReadIonizationTable(ion=ion)
            #
            TInterpol  = np.log10(temperature)
            Tinterpol  = self.SetLimitRange(TInterpol,table_LogTs.min(),table_LogTs.max())
            nHInterpol = np.log10(nH_density)
            nHInterpol  = self.SetLimitRange(nHInterpol,table_LognHs.min(),table_LognHs.max())
            zInterpol  = redshift
            pts        = np.column_stack((zInterpol, TInterpol, nHInterpol))
            result     = interpolate.interpn((table_z, table_LogTs, table_LognHs), table, pts,
                                             method='linear', bounds_error=False, fill_value=None)
            return result

        if iontable_info["table_type"] == 'hm01_cloudy':
            # read the table
            (table_LognHs, table_LogTs, table_z), table = self.ReadIonizationTable(ion=ion)
            TInterpol  = np.log10(temperature)
            Tinterpol  = self.SetLimitRange(TInterpol,table_LogTs.min(),table_LogTs.max())
            nHInterpol = np.log10(nH_density)
            nHInterpol  = self.SetLimitRange(nHInterpol,table_LognHs.min(),table_LognHs.max())
            zInterpol  = redshift

            # temps, densities are logged.
            pts        = np.column_stack((nHInterpol, TInterpol, zInterpol))
            result     = interpolate.interpn((table_LognHs, table_LogTs, table_z),
                                             table, pts, method='linear', bounds_error=False, fill_value=None)
            result[result>1.0] = 1.0
            result[result<1e-33] = 1e-33  # for the logging
            return np.log10(result)

        if iontable_info["table_type"] == 'ploeckinger':

            ((table_z, table_LogTs, table_LognHs, table_LogZs), table) = self.ReadIonizationTable(ion=ion)
            # We divide by the Solar metallicity since the tables are in log10(Z/Zsol)
            Zsol       = 0.013371374
            tmpZ          = np.copy(Z).astype('float64')/Zsol
            # The tables treat zero as log10(1e-50)
            tmpZ[tmpZ==0.0]    = 1e-34
            TInterpol  = np.log10(temperature)
            Tinterpol  = self.SetLimitRange(TInterpol,table_LogTs.min(),table_LogTs.max())
            nHInterpol = np.log10(nH_density)
            nHInterpol  = self.SetLimitRange(nHInterpol,table_LognHs.min(),table_LognHs.max())
            zInterpol  = redshift
            Zinterpol  = np.log10(tmpZ)
            Zinterpol[Zinterpol==-34.00] = -50
            pts        = np.column_stack((zInterpol, TInterpol, Zinterpol, nHInterpol))
            result     = interpolate.interpn((table_z, table_LogTs, table_LogZs, table_LognHs), table, pts,
                                             method='linear', bounds_error=False, fill_value=None)
            return result
